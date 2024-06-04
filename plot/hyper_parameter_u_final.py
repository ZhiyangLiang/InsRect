import matplotlib.pyplot as plt
import numpy as np
import torchvision

# plt.rc('font', family='Times New Roman')
# plt.style.use('seaborn-v0_8-darkgrid')
plt.style.use('seaborn-v0_8-whitegrid')
fig,axes = plt.subplots(2,2,figsize=(8, 6),sharex=False,sharey=False, dpi=400)

u = [50, 55, 60, 65, 70, 75, 80, 85]
xticks = [50, 55, 60, 65, 70, 75, 80, 85]
cifar10_fpr95 = np.array([12.15, 12.26, 12.27, 11.98, 12.30, 12.11, 12.12, 12.30])
cifar10_auroc = np.array([97.33, 97.34, 97.34, 97.36, 97.34, 97.28, 97.28, 97.25])
cifar100_fpr95 = np.array([28.17, 28.34, 27.97, 27.52, 28.16, 29.28, 29.04, 28.82])
cifar100_auroc = np.array([92.61, 92.73, 92.78, 92.88, 92.60, 92.38, 92.45, 92.49])

# axes[0,0].plot(u,cifar10_fpr95,linewidth=1.0, marker='o',alpha=0.8,color='royalblue')
axes[0,0].plot(u,cifar10_fpr95,linewidth=1.0, marker='o',alpha=0.8,color='c')
# axes[0,0].set_title(r'CIFAR-10',fontsize=18)
axes[0,0].set_xlabel('u',fontsize=18, family='Times New Roman', fontweight='bold')
axes[0,0].set_ylabel('FPR95',fontsize=18, family='Times New Roman', fontweight='bold')
axes[0,0].set_xticks(xticks)
axes[0,0].set_yticks([10, 11, 12, 13, 14, 15])
axes[0,0].tick_params(labelsize=18)

# axes[0,1].plot(u,cifar10_auroc,linewidth=1.0,marker='o', alpha=0.8,color='lightseagreen')
axes[0,1].plot(u,cifar10_auroc,linewidth=1.0,marker='o', alpha=0.8,color='limegreen')
# axes[0,1].set_title(r'CIFAR-10',fontsize=18)
axes[0,1].set_xlabel('u',fontsize=18, family='Times New Roman', fontweight='bold')
axes[0,1].set_ylabel('AUROC',fontsize=18, family='Times New Roman', fontweight='bold')
axes[0,1].set_xticks(xticks)
axes[0,1].set_yticks([94, 95, 96, 97, 98, 99])
axes[0,1].tick_params(labelsize=18)

# axes[1,0].plot(u,cifar100_fpr95,linewidth=1.0, marker='o',alpha=0.8,color='royalblue')
axes[1,0].plot(u,cifar100_fpr95,linewidth=1.0, marker='o',alpha=0.8,color='c')
# axes[1,0].set_title(r'CIFAR-100',fontsize=18)
axes[1,0].set_xlabel('u',fontsize=18, family='Times New Roman', fontweight='bold')
axes[1,0].set_ylabel('FPR95',fontsize=18, family='Times New Roman', fontweight='bold')
axes[1,0].set_xticks(xticks)
axes[1,0].set_yticks([25, 26, 27, 28, 29, 30])
axes[1,0].tick_params(labelsize=18)

# axes[1,1].plot(u,cifar100_auroc,linewidth=1.0, marker='o',alpha=0.8,color='lightseagreen')
axes[1,1].plot(u,cifar100_auroc,linewidth=1.0, marker='o',alpha=0.8,color='limegreen')
# axes[1,1].set_title(r'CIFAR-100',fontsize=18)
axes[1,1].set_xlabel('u',fontsize=18, family='Times New Roman', fontweight='bold')
axes[1,1].set_ylabel('AUROC',fontsize=18, family='Times New Roman', fontweight='bold')
axes[1,1].set_xticks(xticks)
axes[1,1].set_yticks([90, 91, 92, 93, 94, 95])
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
# plt.savefig('hyper_parameter_u.png')
plt.savefig('hyper_parameter_u.svg')

