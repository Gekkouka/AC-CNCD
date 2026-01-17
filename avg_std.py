import re
import numpy as np

def process_logs(file_path):
    # 读取日志文件
    with open(file_path, 'r', encoding='utf-8') as f:
        logs = f.readlines()

    # 初始化存储
    subject_acc = {}
    current_sub_id = None

    # 正则表达式匹配
    sub_pattern = re.compile(r"current sub:\s*(\d+),", re.IGNORECASE)
    acc_pattern = re.compile(r"target_acc:\s*([\d.]+)", re.IGNORECASE)

    for line in logs:
        # 去除多余空白符
        line = line.strip()

        # 查找受试者编号
        sub_match = sub_pattern.search(line)
        if sub_match:
            current_sub_id = int(sub_match.group(1))
            # print(f"Sub Match Found: {current_sub_id}")  # 输出匹配内容

        # 查找目标准确率
        acc_match = acc_pattern.search(line)
        if acc_match and current_sub_id is not None:
            target_acc = float(acc_match.group(1))
            # print(f"Acc Match Found: {target_acc}")  # 输出匹配内容

            # 更新最高 target_acc
            if current_sub_id not in subject_acc or target_acc > subject_acc[current_sub_id]:
                subject_acc[current_sub_id] = target_acc

            # 清除当前受试者编号，以便下次匹配时重新设置
            current_sub_id = None

    # 检查是否有匹配到的数据
    if not subject_acc:
        print("No valid data found in the logs!")
        return None, None, None

    # 计算平均值和标准差
    highest_accs = list(subject_acc.values())
    mean_acc = np.mean(highest_accs)
    std_acc = np.std(highest_accs)

    # 打印结果
    print("每个受试者最高的 target_acc:")
    for sub_id, acc in sorted(subject_acc.items()):
        print(f"受试者 {sub_id}: {acc:.4f}")
    print(f"\n最高 target_acc 的平均值: {mean_acc:.4f}")
    print(f"最高 target_acc 的标准差: {std_acc:.4f}")

    return subject_acc, mean_acc, std_acc

# 调用函数
# file_path = r"C:\Users\11537\Desktop\temp.txt"  # 替换为日志文件路径
file_path = "./experiment/log4.txt"
process_logs(file_path)
