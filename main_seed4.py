import os
import numpy as np
import torch
import argparse
from data_process import getDataLoaders
from pretrain_star import PretrainMMDAAE
from train_pro import trainMMDAAE
from utils import setup_seed


def main(log_file, data_loader_dict, optimizer_config_model, optimizer_config_adv, optimizer_config_star, device, args, one_subject):
    if args.dataset_name == 'seed3':
        args.iteration = 12
    elif args.dataset_name == 'seed4':
        args.iteration = 3
    acc = trainMMDAAE(log_file, data_loader_dict, optimizer_config_model, optimizer_config_adv, optimizer_config_star, device, args, one_subject)
    return acc

def pretrain(log_file, data_loader_dict, optimizer_config_model, optimizer_config_star, device, args, one_subject):
    if args.dataset_name == 'seed3':
        args.iteration = 12
    elif args.dataset_name == 'seed4':
        args.iteration = 3
    acc = PretrainMMDAAE(log_file, data_loader_dict, optimizer_config_model, optimizer_config_star, device, args, one_subject)
    return acc

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MMDAAE')

    parser.add_argument('--way', type=str, default='seed-iv', help="experiment message for save model")

    #config of dataset
    parser.add_argument("--dataset_name", type=str, nargs='?', default='seed4', help="the dataset name, supporting seed3 and seed4")
    parser.add_argument("--session", type=str, nargs='?', default='1', help="selected session")
    parser.add_argument("--dim", type=int, default=310, help="dim of input")
    parser.add_argument("--num_workers_train", type=int, default=0, help="Number of subprocesses to use for data loading for train")
    parser.add_argument("--num_workers_test", type=int, default=0, help="Number of subprocesses to use for data loading for testing")
    parser.add_argument("--num_samples_per_domain", type=int, default=64, help="Number of samples to use for compute mmd for train")
    parser.add_argument("--p", type=float, default=0, help="Percentage of enhenced data  to use")

    # 添加權重參數
    parser.add_argument("--w_mmd", type=float, default=0.25, help="weight for MMD loss")  # 0.3
    parser.add_argument("--w_adv", type=float, default=0.25, help="weight for adversarial loss")  # 0.05
    parser.add_argument("--w_ae", type=float, default=0.4, help="weight for autoencoder loss")  # 0.5
    parser.add_argument("--w_cls", type=float, default=2, help="weight for classification loss")  # 0.15
    parser.add_argument("--w_cr", type=float, default=0.1, help="weight for Maximal Coding Rate Reduction loss")

    parser.add_argument("--lr_model", type=int, default=0.0065, help="epoch of calModel")
    parser.add_argument("--lr_adv", type=int, default=0.00012, help="epoch of calModel")
    parser.add_argument("--lr_star", type=int, default=0.00005, help="epoch of calModel")

    # config of MMDAAE
    parser.add_argument("--input_dim", type=int, default=310, help="input dim is the same with sample's last dim")
    parser.add_argument("--hid_dim", type=int, default=64, help="hid dim is for hidden layer of lstm")
    parser.add_argument("--output_dim", type=int, default=310,
                        help="output dim is the same with sample's input dim for autoencoder")
    parser.add_argument("--core", type=int, default=64, help="core is for star model")
    parser.add_argument("--n_layers", type=int, default=1, help="num of layers of lstm")
    parser.add_argument("--weight_decay", type=float, default=0.05, help="weight decay")
    parser.add_argument("--pre_epoch", type=int, default=48, help="epoch of the train phase")
    parser.add_argument("--epoch", type=int, default=128, help="epoch of the train phase")


    args = parser.parse_args()
    args.seed3_path = "./eeg_data/ExtractedFeatures/"
    args.seed4_path = "./eeg_data/eeg_feature_smooth/"

    if args.dataset_name == "seed3":
        args.path = args.seed3_path
        args.cls_classes = 3
        args.time_steps = 30
        args.batch_size = 256  #batch_size,修改時連帶修改iteration
    elif args.dataset_name == "seed4":
        args.path = args.seed4_path
        args.cls_classes = 4
        args.time_steps = 10
        args.batch_size = 128  #batch_size
    else:
        print("need to define the input dataset")

    log_dir = "./experiment"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = "./experiment/log_main2.txt"
    # 每次清空文件内容
    with open(log_file, "w") as file:
        # 写入标题
        file.write("Here records experiment\n")

    setup_seed(1137)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer_config_model, optimizer_config_adv= {"lr": args.lr_model, "weight_decay": args.weight_decay}, {"lr": args.lr_adv, "weight_decay": args.weight_decay}
    optimizer_config_star = {"lr": args.lr_star, "weight_decay": args.weight_decay}

    # pre_optimizer_config_model = {"lr": args.pre_model, "weight_decay": args.weight_decay}
    # pre_optimizer_config_star = {"lr": args.pre_star, "weight_decay": args.weight_decay}
    acc_list = []
    for one_subject in range(1, 16):
        # 1.data preparation
        source_loaders, test_loader = getDataLoaders(one_subject, args)
        data_loader_dict = {"source_loader": source_loaders, "test_loader": test_loader}
        # 2. pretrain
        # acc = pretrain(log_file, data_loader_dict, pre_optimizer_config_model, pre_optimizer_config_star, device, args, one_subject)
        # 3. main
        acc = main(log_file, data_loader_dict, optimizer_config_model, optimizer_config_adv, optimizer_config_star, device, args, one_subject)
        acc_list.append(acc)
    acc_list_str = [str(x) for x in acc_list]
    # 将 acc_list 中的每个张量从 GPU 转移到 CPU，并转换为 NumPy 数组
    acc_list_cpu = [acc.cpu().numpy() if isinstance(acc, torch.Tensor) else acc for acc in acc_list]

    # 计算平均值和标准差
    avg_acc = np.average(acc_list_cpu)
    std_acc = np.std(acc_list_cpu)

    # 打印结果
    print(f"Final experiment accuracy: {acc_list_cpu}")
    print(f"Final experiment accuracy (avg): {avg_acc}")
    print(f"Final experiment accuracy (std): {std_acc}")
