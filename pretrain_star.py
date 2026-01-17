import copy
import os
import time

import numpy as np
import torch
from torch import nn

from model import MMD_AAE, LaplaceDiscriminator, ReverseLayerF, STAR
from utils import MMD_Loss_func, generate_laplace_samples, sample_domains, compute_mcr_loss


def PretrainMMDAAE(log_file, data_loader_dict, optimizer_config_model, optimizer_config_star, device, args, one_subject):
    torch.autograd.set_detect_anomaly(True)

    # log_file, which recording the experiment
    log_file = log_file
    # data of source subjects, which is used as the training set
    source_loader = data_loader_dict['source_loader']
    # The preparation phase
    source_iters = []
    for i in range(len(source_loader)):
        source_iters.append(iter(source_loader[i]))

    shuffle_model = STAR(d_series=args.input_dim, d_core=args.core).to(device)
    model_mmdaae = MMD_AAE(args.cls_classes, args.input_dim, args.hid_dim, args.n_layers, args.output_dim).to(device)

    optimizer = torch.optim.Adam([
        {'params': model_mmdaae.E.parameters(), **optimizer_config_model},  # E 的参数使用模型优化器的配置
        {'params': model_mmdaae.D.parameters(), **optimizer_config_model},  # D 的参数使用模型优化器的配置
        {'params': model_mmdaae.T.parameters(), **optimizer_config_model},  # T 的参数使用模型优化器的配置
        {'params': shuffle_model.parameters(), **optimizer_config_star},  # Star 的参数使用模型优化器的配置
    ])

    mmdLoss = MMD_Loss_func(num_source = 14, sigmas=[0.1, 1, 10])
    taskLoss = nn.CrossEntropyLoss()
    decoderLoss = nn.MSELoss()

    acc_final = 0
    for epoch in range(1, args.pre_epoch+1):
        start_time_train = time.time()
        # 初始化损失列表
        t_loss_list = []
        d_loss_list = []
        mmd_loss_list = []
        total_loss_list = []

        shuffle_model.train()
        model_mmdaae.train()

        data_set_all = 0
        correct_preds = 0  # 用于统计正确预测的数量

        for i in range(1, args.iteration + 1):
            #準備數據
            flag = 0  # 爲了方便拼接
            data, labels, d_labels = None, None, None  # 初始化
            for j in range(len(source_iters)):
                try:
                    data_term, labels_term = next(source_iters[j])
                except StopIteration:
                    source_iters[j] = iter(source_loader[j])
                    data_term, labels_term = next(source_iters[j])

                data_term = torch.autograd.Variable(data_term, requires_grad=False).to(device)
                labels_term = torch.autograd.Variable(labels_term, requires_grad=False).to(device)
                d_labels_term = torch.ones(data_term.size(0)) * j
                d_labels_term = torch.autograd.Variable(d_labels_term, requires_grad=False).to(device)
                if flag == 0:
                    data, labels, d_labels = data_term, labels_term, d_labels_term
                    flag = 1
                else:
                    data, labels, d_labels = (torch.cat((data, data_term), dim=0),
                                              torch.cat((labels, labels_term), dim=0),
                                              torch.cat((d_labels, d_labels_term), dim=0))
            source_data = data.to(device)
            source_label = labels.squeeze(dim=1).to(device)
            source_d_labels = d_labels.to(device)
            data_set_all += len(source_label)

            # ---------------------------------------
            # 更新編碼器,解碼器,分類器 E, D, T, ADV
            # ---------------------------------------
            optimizer.zero_grad()           #梯度清零是每次进行反向传播之前的必要步骤

            # Call the mmdaae model
            source_data_enhance = shuffle_model(source_data)
            e, d, t = model_mmdaae(source_data_enhance, args.time_steps)

            # sampled_data, sampled_labels = sample_domains(e, source_d_labels, num_samples_per_domain=args.num_samples_per_domain, num_domains=14)
            # mmd_loss = mmdLoss(sampled_data, sampled_labels)
            mmd_loss = torch.tensor(0)

            t_loss = taskLoss(t, source_label)
            d_loss = decoderLoss(d, data)

            total_loss = args.w_cls * t_loss + args.w_ae * d_loss + args.w_mmd * mmd_loss
            total_loss.backward()
            optimizer.step()                #调用 .step()，使用优化器更新模型的参数

            # 计算准确率
            _, predicted = torch.max(t, 1)
            correct_preds += (predicted == source_label).sum().item()

            # 将各个损失值存储到列表中
            t_loss_list.append(t_loss.item())
            d_loss_list.append(d_loss.item())
            mmd_loss_list.append(mmd_loss.item())
            total_loss_list.append(total_loss.item())

        print("data set amount: "+str(data_set_all))
        end_time_train = time.time()
        train_epoch_time = end_time_train - start_time_train
        print(f"Epoch: {epoch}","    The time required for one pretraining epoch is：", train_epoch_time, "second")
        print(f"Task Loss Mean: {np.mean(t_loss_list):.4f}, "
              f"Decoder Loss Mean: {np.mean(d_loss_list):.4f}, "
              f"MMD Loss Mean: {np.mean(mmd_loss_list):.4f}, "
              f"Total Loss Mean: {np.mean(total_loss_list):.4f}")
        accuracy = 100.0 * correct_preds / data_set_all
        print(f"Epoch {epoch}/{args.pre_epoch}, Accuracy: {accuracy:.2f}%")

        # ---------------------------------------
        # 評估模型test
        # ---------------------------------------
        shuffle_model.eval()
        model_mmdaae.eval()
        correct, total = 0, 0  # 初始化计数器

        with torch.no_grad():
            for batch_i, (data_, labels_) in enumerate(data_loader_dict["test_loader"]):
                data_, labels_ = data_.cuda(), labels_.cuda()
                # 模型推理
                data_enhence = shuffle_model(data_)
                _, _, t = model_mmdaae(data_enhence, args.time_steps)
                # 获取预测类别
                preds = torch.argmax(t, dim=1).long()
                labels_ = labels_.long().squeeze(dim=1)  # 确保 labels_ 的形状与 preds 一致
                # 统计正确预测数量
                correct += (preds == labels_).sum().item()
                total += labels_.size(0)

        # 检查 total 是否为 0
        acc = correct / total if total > 0 else 0

        print('---------------target_acc:{}-----------------------'.format(acc * 100))
        log_info = f"---------------target_acc:{acc}-----------------------\n"

        if acc > acc_final:
            acc_final = acc
            best_star_model = copy.deepcopy(shuffle_model.state_dict())
            best_mmdaae_model = copy.deepcopy(model_mmdaae.state_dict())


    modelDir = "premodel/" + args.way + "/" + args.session + "/"
    try:
        os.makedirs(modelDir, exist_ok=True)  # 自动忽略已存在的目录
    except:
        pass
    # Save models
    torch.save({'model_state_dict': best_star_model}, f"{modelDir}{str(one_subject)}_star_model.pth")
    print(f"Saved STAR model to {modelDir}{str(one_subject)}_star_model.pth")
    torch.save({'model_state_dict': best_mmdaae_model}, f"{modelDir}{str(one_subject)}_mmdaae_model.pth")
    print(f"Saved mmdaae model to {modelDir}{str(one_subject)}_mmdaae_model.pth")
    return acc_final