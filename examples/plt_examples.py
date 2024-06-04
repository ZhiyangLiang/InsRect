# 导入需要的模块
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 随机生成五组数据
data1 = np.random.normal(0, 1, 100)
data2 = np.random.uniform(-1, 1, 100)
data3 = np.random.exponential(1, 100)
data4 = np.random.poisson(5, 100)
data5 = np.random.binomial(10, 0.5, 100)

# 设置colormap
# cmap = plt.get_cmap('Set3')
cmap = plt.get_cmap('twilight')

# 绘制小提琴图
plt.figure(figsize=(8, 6))
# plt.violinplot([data1, data2, data3, data4, data5], colors=cmap.colors[:5])
plt.violinplot([data1, data2, data3, data4, data5])
plt.title('violin_diagram')
plt.xlabel('data')
plt.ylabel('value')
# plt.show()
plt.savefig("./violin_diagram.png")

# 绘制箱线图
plt.figure(figsize=(8, 6))
# plt.boxplot([data1, data2, data3, data4, data5])
plt.boxplot([data1, data2, data3, data4, data5], patch_artist=True, boxprops=dict(facecolor=cmap.colors[5]), medianprops=dict(color=cmap.colors[6]), whiskerprops=dict(color=cmap.colors[7]), capprops=dict(color=cmap.colors[8]), flierprops=dict(markerfacecolor=cmap.colors[9]))
plt.title('box_plot')
plt.xlabel('data')
plt.ylabel('value')
# plt.show()
plt.savefig("./box_plot.png")

# 绘制三维散点图
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data1, data2, data3, c=data3, cmap=cmap)
ax.set_title('three_dimensional_scatterplot')
ax.set_xlabel('data1')
ax.set_ylabel('data2')
ax.set_zlabel('data3')
# plt.show()
plt.savefig("./three_dimensional_scatterplot.png")

# 绘制折线图
plt.figure(figsize=(8, 6))
plt.plot(data4, label='data4', color=cmap.colors[0])
plt.plot(data5, label='data5', color=cmap.colors[1])
plt.title('line_graph')
plt.xlabel('data')
plt.ylabel('value')
plt.legend()
# plt.show()
plt.savefig("./line_graph.png")
