import re
import matplotlib.pyplot as plt

def process_logs(file_path, sub_id):
    # 初始化存储数据的列表
    task_losses = []
    decoder_losses = []
    mmd_losses = []
    adversarial_losses = []
    min_coding_rate_losses = []  # 新增 Minimum Coding Rate Loss
    total_losses = []
    target_accs = []

    # 正则表达式匹配
    sub_pattern = re.compile(r"current sub:\s*(\d+),", re.IGNORECASE)
    task_loss_pattern = re.compile(r"task loss Mean:\s*([\d.]+)", re.IGNORECASE)
    decoder_loss_pattern = re.compile(r"decoder loss Mean:\s*([\d.]+)", re.IGNORECASE)
    mmd_loss_pattern = re.compile(r"mmd loss Mean:\s*([\d.]+)", re.IGNORECASE)
    adversarial_loss_pattern = re.compile(r"adversarial loss Mean:\s*([\d.]+)", re.IGNORECASE)
    min_coding_rate_loss_pattern = re.compile(r"Minimum Coding Rate Loss Mean:\s*([\d.]+)", re.IGNORECASE)  # 新增
    total_loss_pattern = re.compile(r"total loss Mean:\s*([\d.]+)", re.IGNORECASE)
    acc_pattern = re.compile(r"target_acc:\s*([\d.]+)", re.IGNORECASE)

    current_sub_id = None

    # 读取日志文件
    with open(file_path, 'r', encoding='utf-8') as f:
        logs = f.readlines()

    for line in logs:
        # 去除多余空白符
        line = line.strip()

        # 查找受试者编号
        sub_match = sub_pattern.search(line)
        if sub_match:
            current_sub_id = int(sub_match.group(1))

        # 如果当前行属于目标受试者
        if current_sub_id == sub_id:
            # 匹配各个数据
            task_loss_match = task_loss_pattern.search(line)
            decoder_loss_match = decoder_loss_pattern.search(line)
            mmd_loss_match = mmd_loss_pattern.search(line)
            adversarial_loss_match = adversarial_loss_pattern.search(line)
            min_coding_rate_loss_match = min_coding_rate_loss_pattern.search(line)  # 新增
            total_loss_match = total_loss_pattern.search(line)
            acc_match = acc_pattern.search(line)

            if task_loss_match:
                task_losses.append(float(task_loss_match.group(1)))
            if decoder_loss_match:
                decoder_losses.append(float(decoder_loss_match.group(1)))
            if mmd_loss_match:
                mmd_losses.append(float(mmd_loss_match.group(1)))
            if adversarial_loss_match:
                adversarial_losses.append(float(adversarial_loss_match.group(1)))
            if min_coding_rate_loss_match:  # 新增
                min_coding_rate_losses.append(float(min_coding_rate_loss_match.group(1)))
            if total_loss_match:
                total_losses.append(float(total_loss_match.group(1)))
            if acc_match:
                target_accs.append(float(acc_match.group(1)))

    # 如果没有找到相关数据
    if not task_losses:
        print(f"No data found for sub ID {sub_id}.")
        return

    # 确保所有数据的长度一致
    epochs = range(len(task_losses))

    # 绘制图表
    plt.figure(figsize=(14, 14))  # 增加图表的高度

    # 绘制所有 Loss 在一张图中
    plt.subplot(3, 1, 1)  # 修改为3行1列的布局
    plt.plot(epochs[:len(task_losses)], task_losses, label="Task Loss", marker='o')
    plt.plot(epochs[:len(decoder_losses)], decoder_losses, label="Decoder Loss", marker='o')
    plt.plot(epochs[:len(mmd_losses)], mmd_losses, label="MMD Loss", marker='o')
    plt.plot(epochs[:len(adversarial_losses)], adversarial_losses, label="Adversarial Loss", marker='o')
    # plt.plot(epochs[:len(min_coding_rate_losses)], min_coding_rate_losses, label="Minimum Coding Rate Loss", marker='o')  # 新增
    # plt.plot(epochs[:len(total_losses)], total_losses, label="Total Loss", marker='o')
    plt.title(f"Sub {sub_id} - Losses")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # 绘制 Accuracy 在另一张图中
    plt.subplot(3, 1, 2)  # 修改为3行1列的布局
    plt.plot(epochs, target_accs, label="Target Accuracy", color="cyan", marker='o')
    plt.title(f"Sub {sub_id} - Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# 调用函数
# file_path = r"C:\Users\11537\Desktop\temp.txt"  # 替换为日志文件路径
file_path = "./experiment/log4.txt"

for i in range(1, 16):
    sub_id = i  # 替换为目标受试者编号
    process_logs(file_path, sub_id)
