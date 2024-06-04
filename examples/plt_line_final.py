import matplotlib.pyplot as plt
import numpy as np
# plt.style.use('seaborn-v0_8-darkgrid')
plt.style.use('seaborn-v0_8-whitegrid')
fig,axes = plt.subplots(2,2,figsize=(8, 6),sharex=False,sharey=False, dpi=400)
Percent = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
xticks = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
yticks_f10 = [9, 12, 15, 18, 21, 24]
yticks_f100 = [25, 28, 31, 34, 37, 40]
yticks = [88, 90, 92, 94, 96, 98]
cifar10_fpr95 = np.array([11.79, 13.33, 16.22, 18.66, 19.18, 19.57, 19.64, 19.97, 19.79, 19.51])
cifar10_auroc = np.array([97.41, 96.76, 95.90, 95.22, 95.04, 94.93, 94.89, 94.85, 94.87, 94.89])
cifar100_fpr95 = np.array([37.00, 27.25, 26.74, 28.66, 31.61, 31.01, 32.99, 34.36, 35.51, 36.00])
cifar100_auroc = np.array([90.62, 92.93, 93.06, 92.52, 90.85, 91.57, 90.96, 90.60, 90.91, 89.70])

axes[0,0].plot(Percent,cifar10_fpr95,linewidth=1.0,marker='o',alpha=0.8,color='royalblue')
# axes[0,0].set_title(r'CIFAR-10',fontsize=18)
axes[0,0].set_xlabel(r'Percent',fontsize=18)
axes[0,0].set_ylabel('FPR95',fontsize=18)
axes[0,0].set_xticks(xticks)
axes[0,0].set_yticks(yticks_f10)
axes[0,0].tick_params(labelsize=18)

axes[0,1].plot(Percent,cifar10_auroc,linewidth=1.0,marker='o', alpha=0.8,color='lightseagreen')
# axes[0,1].set_title(r'CIFAR-10',fontsize=18)
axes[0,1].set_xlabel(r'Percent',fontsize=18)
axes[0,1].set_ylabel('AUROC',fontsize=18)
axes[0,1].set_xticks(xticks)
axes[0,1].set_yticks(yticks)
axes[0,1].tick_params(labelsize=18)

axes[1,0].plot(Percent,cifar100_fpr95,linewidth=1.0, marker='o',alpha=0.8,color='royalblue')
# axes[1,0].set_title(r'CIFAR-100',fontsize=18)
axes[1,0].set_xlabel(r'Percent',fontsize=18)
axes[1,0].set_xticks(xticks)
axes[1,0].set_yticks(yticks_f100)
axes[1,0].set_ylabel('FPR95',fontsize=18)
axes[1,0].tick_params(labelsize=18)

axes[1,1].plot(Percent,cifar100_auroc,linewidth=1.0, marker='o',alpha=0.8,color='lightseagreen')
# axes[1,1].set_title(r'CIFAR-100',fontsize=18)
axes[1,1].set_xticks(xticks)
axes[1,1].set_yticks(yticks)
axes[1,1].set_xlabel(r'Percent',fontsize=18)
axes[1,1].set_ylabel('AUROC',fontsize=18)
axes[1,1].tick_params(labelsize=18)

plt.tight_layout()

# axes[0,0].grid(True)
# axes[0,1].grid(True)
# axes[1,0].grid(True)
# axes[1,1].grid(True)

plt.savefig('new_scaling_percent.svg')
