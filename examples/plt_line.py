# 导入需要的模块
import numpy as np
import matplotlib.pyplot as plt

# 随机生成两组数据
cifar10_fpr95 = np.array([11.79, 13.33, 16.22, 18.66, 19.18, 19.57, 19.64, 19.97, 19.79, 19.51])
cifar10_auroc = np.array([97.41, 96.76, 95.90, 95.22, 95.04, 94.93, 94.89, 94.85, 94.87, 94.89])
cifar100_fpr95 = np.array([37.00, 27.25, 26.74, 28.66, 31.61, 31.01, 32.99, 34.36, 35.51, 36.00])
cifar100_auroc = np.array([90.62, 92.93, 93.06, 92.52, 90.85, 91.57, 90.96, 90.60, 90.91, 89.70])

# 设置标签
labels = ['5', '10', '15', '20', '25', '30', '35', '40', '45', '50']

# 绘制柱状图
plt.figure(figsize=(15, 10))
plt.plot(labels, cifar10_fpr95, color='gold', marker='o', label='FPR95(CIFAR10)')
plt.plot(labels, cifar100_fpr95, color='darkorange', marker='*', label='FPR95(CIFAR100)')
# plt.plot(labels, cifar10_auroc, color='lightseagreen', marker='o', label='AUROC(CIFAR10)')
# plt.plot(labels, cifar100_auroc, color='mediumspringgreen', marker='*', label='AUROC(CIFAR100)')
# plt.title('Sensitivity analysis of scaling hyperparameter')
plt.xlabel('Percent', fontsize=30)
plt.ylabel('FPR95', fontsize=30)
# plt.ylabel('AUROC', fontsize=15)
plt.ylim(0, 40)
# plt.ylim(85, 100)
plt.tick_params(labelsize=30)
plt.rcParams.update({'font.size': 30})
plt.legend(loc="lower right")

for x, y in zip(labels, cifar10_fpr95):
    if int(x) % 10 == 0:
        continue
    plt.text(x, y, str(y), ha='center', va='top', fontsize=25)
for x, y in zip(labels, cifar100_fpr95):
    if int(x) % 10 == 0:
        continue
    plt.text(x, y, str(y), ha='center', va='top', fontsize=25)

# for x, y in zip(labels, cifar10_auroc):
#     plt.text(x, y, str(y), ha='center', va='top', fontsize=15)
# for x, y in zip(labels, cifar100_auroc):
#     plt.text(x, y, str(y), ha='center', va='top', fontsize=15)

# plt.show()
plt.savefig("./test_fpr.png")
# plt.savefig("./cifar_auroc.svg")
# plt.savefig("./cifar_both.svg")
