# 导入需要的模块
import numpy as np
import matplotlib.pyplot as plt

def PCA_visualize(X,Y,save_path='',epoch=0,num_classes=7):
    save_path = os.path.join(save_path,f'pca_{epoch}.svg')
    pca = PCA(n_components=2)
    pca_results = pca.fit_transform(X)
    fig, ax = plt.subplots(figsize=(7, 5), sharey=True, sharex=True, dpi=400)
    plt.scatter(pca_results[:,0],pca_results[:,1],c=Y,cmap='tab10',s=5)
    fig.tight_layout()
    plt.xticks([])
    plt.yticks([])
    fig.savefig(save_path)
    plt.clf()
    plt.close()

# 生成两组数据
PCA_ID_x = np.random.normal(0, 1, 100) # 生成100个服从正态分布的x坐标
PCA_ID_y = np.random.normal(0, 1, 100) # 生成100个服从正态分布的y坐标
PCA_OOD_x = np.random.normal(0, 1, 100) # 生成100个服从正态分布的x坐标
PCA_OOD_y = np.random.normal(0, 1, 100) # 生成100个服从正态分布的y坐标
NMF_ID_x = np.random.uniform(-1, 1, 100) # 生成100个服从均匀分布的x坐标
NMF_ID_y = np.random.uniform(-1, 1, 100) # 生成100个服从均匀分布的y坐标
NMF_OOD_x = np.random.uniform(-1, 1, 100) # 生成100个服从均匀分布的x坐标
NMF_OOD_y = np.random.uniform(-1, 1, 100) # 生成100个服从均匀分布的y坐标
ICA_ID_x = np.random.exponential(1, 100) # 生成100个服从指数分布的随机数
ICA_ID_y = np.random.exponential(1, 100) # 生成100个服从指数分布的随机数
ICA_OOD_x = np.random.exponential(1, 100) # 生成100个服从指数分布的随机数
ICA_OOD_y = np.random.exponential(1, 100) # 生成100个服从指数分布的随机数


# 绘制散点图
plt.figure(figsize=(8, 6))
plt.scatter(PCA_ID_x, PCA_ID_y, s=10, c='firebrick', marker='o', label='PCA_ID')
plt.scatter(PCA_OOD_x, PCA_OOD_y, s=10, c='lightcoral', marker='*', label='PCA_OOD')
plt.scatter(NMF_ID_x, NMF_ID_y, s=10, c='slateblue', marker='o', label='NMF_ID')
plt.scatter(NMF_OOD_x, NMF_OOD_y, s=10, c='lightskyblue', marker='*', label='NMF_OOD')
plt.scatter(ICA_ID_x, ICA_ID_y, s=10, c='forestgreen', marker='o', label='ICA_ID')
plt.scatter(ICA_OOD_x, ICA_OOD_y, s=10, c='lightgreen', marker='*', label='ICA_OOD')
plt.title('scatter_graph')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
# plt.show()
plt.savefig("./scatter.png")
