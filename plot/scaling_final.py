import matplotlib.pyplot as plt
import numpy as np

# plt.rc('font', family='Times New Roman')
# plt.style.use('seaborn-v0_8-darkgrid')
plt.style.use('seaborn-v0_8-whitegrid')
fig,axes = plt.subplots(2,2,figsize=(8, 6),sharex=False,sharey=False, dpi=400)

Percent = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
xticks = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
cifar10_fpr95 = np.array([11.98, 13.15, 15.88, 17.90, 19.14, 19.18, 19.40, 19.54, 19.63, 19.69])
cifar10_auroc = np.array([97.36, 96.73, 95.91, 95.32, 95.00, 94.98, 94.92, 94.89, 94.87, 94.85])
cifar100_fpr95 = np.array([36.86, 27.63, 27.52, 28.35, 29.78, 31.17, 32.63, 33.91, 35.14, 36.24])
cifar100_auroc = np.array([90.76, 92.84, 92.88, 92.48, 92.01, 91.49, 91.02, 90.56, 90.11, 89.68])

axes[0,0].plot(Percent,cifar10_fpr95,linewidth=1.0, marker='o',alpha=0.8,color='firebrick')
# axes[0,0].set_title(r'CIFAR-10',fontsize=18)
axes[0,0].set_xlabel('Percent',fontsize=18, family='Times New Roman', fontweight='bold')
axes[0,0].set_ylabel('FPR95',fontsize=18, family='Times New Roman', fontweight='bold')
axes[0,0].set_xticks(xticks)
axes[0,0].set_yticks([9, 12, 15, 18, 21, 24])
axes[0,0].tick_params(labelsize=18)

axes[0,1].plot(Percent,cifar10_auroc,linewidth=1.0,marker='o', alpha=0.8,color='darkviolet')
# axes[0,1].set_title(r'CIFAR-10',fontsize=18)
axes[0,1].set_xlabel('Percent',fontsize=18, family='Times New Roman', fontweight='bold')
axes[0,1].set_ylabel('AUROC',fontsize=18, family='Times New Roman', fontweight='bold')
axes[0,1].set_xticks(xticks)
axes[0,1].set_yticks([88, 90, 92, 94, 96, 98])
axes[0,1].tick_params(labelsize=18)

axes[1,0].plot(Percent,cifar100_fpr95,linewidth=1.0, marker='o',alpha=0.8,color='firebrick')
# axes[1,0].set_title(r'CIFAR-100',fontsize=18)
axes[1,0].set_xlabel('Percent',fontsize=18, family='Times New Roman', fontweight='bold')
axes[1,0].set_ylabel('FPR95',fontsize=18, family='Times New Roman', fontweight='bold')
axes[1,0].set_xticks(xticks)
axes[1,0].set_yticks([25, 28, 31, 34, 37, 40])
axes[1,0].tick_params(labelsize=18)

axes[1,1].plot(Percent,cifar100_auroc,linewidth=1.0, marker='o',alpha=0.8,color='darkviolet')
# axes[1,1].set_title(r'CIFAR-100',fontsize=18)
axes[1,1].set_xlabel('Percent',fontsize=18, family='Times New Roman', fontweight='bold')
axes[1,1].set_ylabel('AUROC',fontsize=18, family='Times New Roman', fontweight='bold')
axes[1,1].set_xticks(xticks)
axes[1,1].set_yticks([88, 90, 92, 94, 96, 98])
axes[1,1].tick_params(labelsize=18)

plt.sca(axes[0, 0])
plt.xticks(fontname='Times New Roman', fontweight='bold')
plt.yticks(fontname='Times New Roman', fontweight='bold')
plt.sca(axes[0, 1])
plt.xticks(fontname='Times New Roman', fontweight='bold')
plt.yticks(fontname='Times New Roman', fontweight='bold')
plt.sca(axes[1, 0])
plt.xticks(fontname='Times New Roman', fontweight='bold')
plt.yticks(fontname='Times New Roman', fontweight='bold')
plt.sca(axes[1, 1])
plt.xticks(fontname='Times New Roman', fontweight='bold')
plt.yticks(fontname='Times New Roman', fontweight='bold')

plt.tight_layout()
# plt.savefig('new_scaling_percent.png')
plt.savefig('new_scaling_percent.svg')
