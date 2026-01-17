import random
import numpy as np
import torch
import torch.nn.functional as F

def setup_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def timeStepsShuffle(source_data):
    source_data_1 = source_data.clone()
    #retain the last time step
    curTimeStep_1 = source_data_1[:, -1, :]
    # get data of other time steps
    dim_size = source_data[:, :-1, :].size(1)
    # generate a random sequence
    idxs_1 = list(range(dim_size))
    # generate a shuffled sequence
    random.shuffle(idxs_1)
    # get data corresponding to the shuffled sequence
    else_1 = source_data_1[:, idxs_1, :]
    # add the origin last time step
    result_1 = torch.cat([else_1, curTimeStep_1.unsqueeze(1)], dim=1)
    return result_1

# 生成高斯分布样本
def generate_gaussian_samples(n_samples, mean=0.0, std=1.0, input_dim=64):
    return np.random.normal(mean, std, size=(n_samples, input_dim))

# 生成拉普拉斯分布样本
def generate_laplace_samples(n_samples, loc=0.0, scale=1.0, input_dim=64):
    return np.random.laplace(loc, scale, (n_samples, input_dim))

def MMD_Loss_func(num_source=3, sigmas=None):
    if sigmas is None:
        sigmas = [1, 5, 10]
    def loss(e_pred, d_true):
        cost = 0.0
        for i in range(num_source):
            domain_i = e_pred[d_true == i]
            for j in range(i+1, num_source):
                domain_j = e_pred[d_true == j]
                single_res = mmd_two_distribution(domain_i, domain_j, sigmas=sigmas)
                cost += single_res
        return cost/num_source
    return loss

def mmd_two_distribution(source, target, sigmas):
    device = source.device  # Ensure all tensors are on the same device
    sigmas = torch.tensor(sigmas, device=device)  # Move sigmas to the same device
    xy = rbf_kernel(source, target, sigmas)
    xx = rbf_kernel(source, source, sigmas)
    yy = rbf_kernel(target, target, sigmas)
    return xx + yy - 2 * xy

def rbf_kernel(x, y, sigmas):
    # Ensure correct shape for sigmas
    sigmas = sigmas.reshape(-1, 1)  # Reshape sigmas for compatibility
    beta = 1. / (2. * sigmas)
    dist = compute_pairwise_distances(x, y)
    dist_reshaped = dist.reshape(1, -1)  # Ensure proper reshaping for multiplication
    dot = -beta.view(-1, 1) * dist_reshaped
    exp = torch.mean(torch.exp(dot))
    return exp

def compute_pairwise_distances(x, y):
    x_square = torch.sum(x ** 2, dim=1, keepdim=True)
    y_square = torch.sum(y ** 2, dim=1, keepdim=True)
    xy_cross = torch.matmul(x, y.T)
    dist = x_square - 2 * xy_cross + y_square.T
    dist = torch.clamp(dist, min=0.0)  # Ensure non-negative distances
    return dist

def sample_domains(source_data, source_d_labels, num_samples_per_domain=32, num_domains=14):
    """
    从多个域中抽样指定数量的数据点。
    """
    sampled_data = []
    sampled_labels = []

    for domain in range(num_domains):
        # 获取当前域的样本索引
        domain_indices = (source_d_labels == domain).nonzero(as_tuple=True)[0]

        # 如果当前域的样本数量不足，使用随机采样（带替换）
        if len(domain_indices) < num_samples_per_domain:
            sampled_indices = torch.randint(len(domain_indices), (num_samples_per_domain,))
            sampled_indices = domain_indices[sampled_indices]
        else:
            # 如果样本数量足够，使用不带替换的随机抽样
            sampled_indices = domain_indices[torch.randperm(len(domain_indices))[:num_samples_per_domain]]

        # 收集采样结果
        sampled_data.append(source_data[sampled_indices])
        sampled_labels.append(source_d_labels[sampled_indices])

    # 合并结果
    sampled_data = torch.cat(sampled_data, dim=0)
    sampled_labels = torch.cat(sampled_labels, dim=0)

    return sampled_data, sampled_labels

def compute_mcr_loss(fea, cache, tol_factor=0.2, coef_mcr_update=0.99):
    # Flatten embeddings
    emb = torch.flatten(fea, 1)

    # Normalize embeddings
    emb_norm = F.normalize(emb, dim=-1)

    # Compute similarity matrix and SpaceScore
    I = torch.eye(emb_norm.shape[0], device=emb.device)
    emb_all_sim = emb_norm @ emb_norm.T
    space_score = torch.logdet(
        I + emb_norm.shape[0] / emb_norm.shape[1] / 0.01 * emb_all_sim
    )

    # Initialize cache if empty
    if "cache_mcr_all_score" not in cache:
        cache["cache_mcr_all_score"] = torch.tensor([0.0], device=emb.device)
        cache["cache_mcr_all_score_list"] = []

    # Update the score list
    cache["cache_mcr_all_score_list"].append(space_score.item())  # Add the current score to the list

    # Update cached score
    if cache["cache_mcr_all_score"].eq(0).all():
        cache["cache_mcr_all_score"] = torch.tensor(cache["cache_mcr_all_score_list"]).mean().to(emb.device)
    else:
        beta = coef_mcr_update
        cache["cache_mcr_all_score"] = beta * cache["cache_mcr_all_score"] + (1 - beta) * space_score.detach()

    # Compute MCR loss
    loss_mcr = torch.log(
        torch.cosh(tol_factor * (space_score - cache["cache_mcr_all_score"]))
    ) / tol_factor

    return loss_mcr.mean(), cache

# def compute_pairwise_distances(x, y=None, eps=1e-9):
#     """
#     优化后的 pairwise 距离计算，支持归一化和数值稳定性调整。
#     Args:
#         x: Tensor of shape (N, D) - 样本集 X
#         y: Tensor of shape (M, D) - 样本集 Y，默认为 None（即计算 x 与自身的距离）
#         eps: float - 防止数值不稳定的小偏移量
#
#     Returns:
#         Tensor: 距离矩阵，shape 为 (N, M)
#     """
#     if y is None:
#         # 计算样本自身的 pairwise 距离
#         dist = torch.nn.functional.pdist(x, p=2)
#         # 将压缩形式的 pdist 结果还原为方阵
#         n = x.size(0)
#         dist_matrix = torch.zeros(n, n, device=x.device)
#         triu_indices = torch.triu_indices(n, n, offset=1)
#         dist_matrix[triu_indices[0], triu_indices[1]] = dist
#         dist_matrix = dist_matrix + dist_matrix.T
#     else:
#         # 使用 torch.cdist 直接计算 pairwise 距离
#         dist_matrix = torch.cdist(x, y, p=2)
#
#     # 数值稳定性处理，保证非负且加入 eps 防止 sqrt(0) 的情况
#     return torch.clamp(dist_matrix, min=eps)
#
#
# def rbf_kernel(x, y, sigmas):
#     """
#     使用优化的距离计算方法的 RBF 核
#     Args:
#         x: Tensor of shape (N, D) - 样本集 X
#         y: Tensor of shape (M, D) - 样本集 Y
#         sigmas: Tensor of shape (K,) - 不同的 sigma 参数
#
#     Returns:
#         Tensor: 标量值，表示加权平均后的核值
#     """
#     # 将 sigmas 转换为列向量以支持广播
#     sigmas = sigmas.view(-1, 1, 1)  # Shape: (K, 1, 1)
#
#     # 计算 pairwise 距离
#     dist = compute_pairwise_distances(x, y)  # Shape: (N, M)
#
#     # 加入归一化（以特征维度的均值为归一化因子，避免大值溢出）
#     dist = dist / (x.size(1) ** 0.5)  # 正则化距离
#
#     # 增加一个维度以与 sigmas 兼容
#     dist = dist.unsqueeze(0)  # Shape: (1, N, M)
#
#     # 计算 RBF 核
#     beta = 1.0 / (2.0 * sigmas)  # Shape: (K, 1, 1)
#     kernel = torch.exp(-beta * dist)  # Shape: (K, N, M)
#
#     # 对所有 sigma 取平均值，得到最终核矩阵
#     return kernel.mean(dim=0)  # Shape: (N, M)
#
#
# def mmd_two_distribution(source, target, sigmas):
#     """
#     计算两个分布之间的 MMD 值
#     Args:
#         source: Tensor of shape (N, D) - 源域数据
#         target: Tensor of shape (M, D) - 目标域数据
#         sigmas: Tensor of shape (K,) - 不同的 sigma 参数
#
#     Returns:
#         float: MMD 值
#     """
#     device = source.device  # 保证所有数据在同一设备上
#     sigmas = torch.tensor(sigmas, device=device)  # 将 sigmas 移动到同一设备
#     # 计算 RBF 核矩阵
#     xy = rbf_kernel(source, target, sigmas)
#     xx = rbf_kernel(source, source, sigmas)
#     yy = rbf_kernel(target, target, sigmas)
#     return xx.mean() + yy.mean() - 2 * xy.mean()
#
#
# def MMD_Loss_func(num_source=14, sigmas=None):
#     """
#     定义 MMD 损失函数，用于多个源域间的 MMD 计算
#     Args:
#         num_source: int - 源域数量
#         sigmas: list - 不同的 sigma 参数
#
#     Returns:
#         callable: 返回损失函数
#     """
#     if sigmas is None:
#         sigmas = [1, 5, 10]
#
#     def loss(e_pred, d_true):
#         cost = 0.0
#         for i in range(num_source):
#             domain_i = e_pred[d_true == i]
#             for j in range(i + 1, num_source):
#                 domain_j = e_pred[d_true == j]
#                 if len(domain_i) > 0 and len(domain_j) > 0:
#                     # 计算两域间的 MMD
#                     single_res = mmd_two_distribution(domain_i, domain_j, sigmas)
#                     cost += single_res
#         return cost / num_source  # 平均化损失
#
#     return loss
#
