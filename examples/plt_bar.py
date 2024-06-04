# 导入需要的模块
import numpy as np
import matplotlib.pyplot as plt

# 随机生成两组数据
data1 = np.random.randint(10, 100, 10)
data2 = np.random.randint(10, 100, 10)

# 设置标签
labels = ['5', '10', '15', '20', '25', '30', '35', '40', '45', '50']

# 设置柱子的宽度和位置
width = 0.2
x1 = np.arange(len(labels))
x2 = x1 + width

# 绘制柱状图
plt.figure(figsize=(8, 6))
plt.bar(x1, data1, width, color='powderblue', label='data1')
plt.bar(x2, data2, width, color='mediumturquoise', label='data2')
plt.title('bar_chart')
plt.xlabel('data')
plt.ylabel('value')
plt.xticks(x1 + width / 2, labels)
plt.legend()
# plt.show()
plt.savefig("./bar_chart.png")
