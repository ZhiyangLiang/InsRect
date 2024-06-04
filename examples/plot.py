import matplotlib.pyplot as plt
import numpy as np
plt.style.use('seaborn-v0_8-darkgrid')
fig,axes = plt.subplots(2,2,figsize=(8, 6),sharex=False,sharey=False, dpi=400)
T = 2*np.arange(0,9)
xticks = 2*np.arange(0,9)
y1 = np.array([29.16,
39.52,
46.60,
48.02,
48.20,
48.16,
50.20,
51.62,
48.80])
y2 = np.array([77.84,
80.06,
80.20,
79.88,
79.32,
79.06,
77.20,
77.26,
76.98])
y3 = np.array([57.04,
61.32,
61.06,
60.32,
59.24,
58.68,
55.40,
55.26,
54.62])
y4 = np.array([69.9,
70.68,
70.54,
69.86,
69.58,
69.32,
67.16,
67.44,
67.52])
axes[0,0].plot(T,y1,linewidth=1.0,marker='o',alpha=0.8,color='royalblue')
axes[0,0].set_title(r'Cora (Asymmetric $\phi=0.5$)',fontsize=12)
axes[0,0].set_xlabel(r'$T$',fontsize=12)
axes[0,0].set_ylabel('Accuracy',fontsize=12)
axes[0,0].set_xticks(xticks)
axes[0,1].plot(T,y2,linewidth=1.0,marker='o', alpha=0.8,color='royalblue')
axes[ 0,1].set_title(r'Cora (Symmetric $\phi=0.4$)',fontsize=12)
axes[ 0,1].set_xlabel(r'$T$',fontsize=12)
axes[0,1].set_ylabel('Accuracy',fontsize=12)
axes[0,1].set_xticks(xticks)
axes[1,0].plot(T,y3,linewidth=1.0, marker='o',alpha=0.8,color='royalblue')
axes[1, 0].set_title(r'CiteSeer (Asymmetric $\phi=0.4$)',fontsize=12)
axes[1,0].set_xlabel(r'$T$',fontsize=12)
axes[1,0].set_xticks(xticks)
axes[1,0].set_ylabel('Accuracy',fontsize=12)
axes[1,1].plot(T,y4,linewidth=1.0, marker='o',alpha=0.8,color='royalblue')
axes[1, 1].set_title(r'CiteSeer (Symmetric $\phi=0.4$)',fontsize=12)
axes[1,1].set_xticks(xticks)
axes[1,1].set_xlabel(r'$T$',fontsize=12)
axes[1,1].set_ylabel('Accuracy',fontsize=12)
axes[0,0].grid(True)
axes[0,1].grid(True)
axes[1,0].grid(True)
axes[1,1].grid(True)

plt.tight_layout()
plt.savefig('ablation_T.png', dpi=400)
