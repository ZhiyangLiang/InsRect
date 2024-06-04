# 导入需要的模块
import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description="hybrid", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--metric", type=str, choices=["fpr95", "auroc"])
parser.add_argument("--table", type=int, choices=[7, 8])
args = parser.parse_args()

# 随机生成两组数据
if args.table == 7:
    if args.metric == "fpr95":
        data1 = np.array([4.66, 22.37])
        data2 = np.array([5.22, 28.38])
        data3 = np.array([7.53, 32.25])
    elif args.metric == "auroc":
        data1 = np.array([98.91, 94.23])
        data2 = np.array([98.26, 93.48])
        data3 = np.array([98.30, 92.60])
elif args.table == 8:
    if args.metric == "fpr95":
        data1 = np.array([4.66, 22.37])
        data2 = np.array([7.12, 36.89])
        data3 = np.array([5.56, 31.56])
    elif args.metric == "auroc":
        data1 = np.array([98.91, 94.23])
        data2 = np.array([98.30, 92.30])
        data3 = np.array([98.65, 92.43])


# 设置标签
labels = ['CIFAR-10', 'CIFAR-100']

# 设置柱子的宽度和位置
width = 0.5
x1 = np.arange(len(labels))
x2 = x1 + width / 3
x3 = x1 + width / 3 * 2

# 绘制柱状图
plt.figure(figsize=(10, 8))
if args.table == 7:
    plt.bar(x1, data1, width / 3, color='royalblue', label='regret')
    plt.bar(x2, data2, width / 3, color='forestgreen', label='risk')
    plt.bar(x3, data3, width / 3, color='lightcoral', label='random')
elif args.table == 8:
    plt.bar(x1, data1, width / 3, color='royalblue', label='IDS')
    plt.bar(x2, data2, width / 3, color='forestgreen', label='ADV')
    plt.bar(x3, data3, width / 3, color='lightcoral', label='EMB')

plt.xlabel('Datasets', family='Times New Roman', fontweight='bold', fontsize=30)
if args.metric == "fpr95":
    plt.ylabel('FPR95', family='Times New Roman', fontweight='bold', fontsize=30)
elif args.metric == "auroc":
    plt.ylabel('AUROC', family='Times New Roman', fontweight='bold', fontsize=30)

plt.xticks(x1 + width / 3, labels, family='Times New Roman', fontweight='bold', fontsize=30)
if args.metric == "fpr95":
    plt.yticks(family='Times New Roman', fontweight='bold', fontsize=30)
elif args.metric == "auroc":
    plt.yticks(family='Times New Roman', fontweight='bold', fontsize=30)

if args.table == 7:
    if args.metric == "fpr95":
        plt.ylim([0, 40])
    elif args.metric == "auroc":
        plt.ylim([80, 100])
elif args.table == 8:
    if args.metric == "fpr95":
        plt.ylim([0, 40])
    elif args.metric == "auroc":
        plt.ylim([90, 100])
plt.rcParams.update({'font.size': 30, 'font.weight': 'bold', 'font.family': 'Times New Roman'})
plt.legend(loc='upper right') 
# plt.show()
if args.table == 7:
    if args.metric == "fpr95":
        plt.savefig("./wdoe_bar1.jpg") # 0-40
        # plt.savefig("./regret_risk_random_fpr95.svg")
    elif args.metric == "auroc":
        plt.savefig("./wdoe_bar2.jpg") # 80-100
        # plt.savefig("./regret_risk_random_auroc.svg")
elif args.table == 8:
    if args.metric == "fpr95":
        plt.savefig("./wdoe_bar3.jpg") # 0-40
        # plt.savefig("./ids_adv_emb_fpr95.svg")
    elif args.metric == "auroc":
        plt.savefig("./wdoe_bar4.jpg") # 90-100
        # plt.savefig("./ids_adv_emb_auroc.svg")
