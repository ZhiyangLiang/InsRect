# 导入需要的模块
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pdb

# 生成三组数据
PCA_ID_x = np.random.normal(0, 1, 100) # 生成100个服从正态分布的x坐标
PCA_ID_y = np.random.normal(0, 1, 100) # 生成100个服从正态分布的y坐标
PCA_ID_z = np.random.normal(0, 1, 100) # 生成100个服从正态分布的y坐标
PCA_OOD_x = np.random.normal(0, 1, 100) # 生成100个服从正态分布的x坐标
PCA_OOD_y = np.random.normal(0, 1, 100) # 生成100个服从正态分布的y坐标
PCA_OOD_z = np.random.normal(0, 1, 100) # 生成100个服从正态分布的y坐标

# NMF_ID_x = np.random.uniform(-1, 1, 100) # 生成100个服从均匀分布的x坐标
# NMF_ID_y = np.random.uniform(-1, 1, 100) # 生成100个服从均匀分布的y坐标
# NMF_ID_z = np.random.uniform(-1, 1, 100) # 生成100个服从均匀分布的y坐标
# NMF_OOD_x = np.random.uniform(-1, 1, 100) # 生成100个服从均匀分布的x坐标
# NMF_OOD_y = np.random.uniform(-1, 1, 100) # 生成100个服从均匀分布的y坐标
# NMF_OOD_z = np.random.uniform(-1, 1, 100) # 生成100个服从均匀分布的y坐标

NMF_ID_x = np.random.normal(0, 1, 100) # 生成100个服从均匀分布的x坐标
NMF_ID_y = np.random.normal(0, 1, 100) # 生成100个服从均匀分布的y坐标
NMF_ID_z = np.random.normal(0, 1, 100) # 生成100个服从均匀分布的y坐标
NMF_OOD_x = np.random.normal(0, 1, 100) # 生成100个服从均匀分布的x坐标
NMF_OOD_y = np.random.normal(0, 1, 100) # 生成100个服从均匀分布的y坐标
NMF_OOD_z = np.random.normal(0, 1, 100) # 生成100个服从均匀分布的y坐标

# ICA_ID_x = np.random.exponential(1, 100) # 生成100个服从指数分布的随机数
# ICA_ID_y = np.random.exponential(1, 100) # 生成100个服从指数分布的随机数
# ICA_ID_z = np.random.exponential(1, 100) # 生成100个服从指数分布的随机数
# ICA_OOD_x = np.random.exponential(1, 100) # 生成100个服从指数分布的随机数
# ICA_OOD_y = np.random.exponential(1, 100) # 生成100个服从指数分布的随机数
# ICA_OOD_z = np.random.exponential(1, 100) # 生成100个服从指数分布的随机数

ICA_ID_x = np.random.normal(0, 1, 100) # 生成100个服从指数分布的随机数
ICA_ID_y = np.random.normal(0, 1, 100) # 生成100个服从指数分布的随机数
ICA_ID_z = np.random.normal(0, 1, 100) # 生成100个服从指数分布的随机数
ICA_OOD_x = np.random.normal(0, 1, 100) # 生成100个服从指数分布的随机数
ICA_OOD_y = np.random.normal(0, 1, 100) # 生成100个服从指数分布的随机数
ICA_OOD_z = np.random.normal(0, 1, 100) # 生成100个服从指数分布的随机数

# 设置三维散点图的画布
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# 绘制三维散点图fontdict={s=20, c='orangered', marker='o', label='PCA_ID'}
ax.scatter(PCA_ID_x, PCA_ID_y, PCA_ID_z, c='firebrick', marker='o', label='PCA_ID', s=10)
ax.scatter(PCA_OOD_x, PCA_OOD_y, PCA_OOD_z, c='lightcoral', marker='*', label='PCA_OOD', s=10)
ax.scatter(NMF_ID_x, NMF_ID_y, NMF_ID_z, c='slateblue', marker='o', label='NMF_ID', s=10)
ax.scatter(NMF_OOD_x, NMF_OOD_y, NMF_OOD_z, c='lightskyblue', marker='*', label='NMF_OOD', s=10)
ax.scatter(ICA_ID_x, ICA_ID_y, ICA_ID_z, c='forestgreen', marker='o', label='ICA_ID', s=10)
ax.scatter(ICA_OOD_x, ICA_OOD_y, ICA_OOD_z, c='lightgreen', marker='*', label='ICA_OOD', s=10)

# 设置标题和坐标轴标签
ax.set_title('three_dimensional_scatterplot')

# 显示图例
ax.legend()

# 显示图形
# plt.show()
plt.savefig("./three_dimensional_scatterplot.png")
