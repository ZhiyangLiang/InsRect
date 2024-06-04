import numpy as np
import matplotlib.pyplot as plt

# 生成两组数据
data1 = np.random.randint(10, 100, 10) # 10 random integers between 10 and 100
data2 = np.random.randint(10, 100, 10) # 10 random integers between 10 and 100

# 设置标签
labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']

# 绘制折线图
plt.figure(figsize=(8, 6))
plt.plot(labels, data1, color='blue', label='Data 1')
plt.plot(labels, data2, color='green', label='Data 2')
plt.title('Line Chart')
plt.xlabel('Category')
plt.ylabel('Value')
plt.legend()

# 标注每个数据点的数值
for x, y in zip(labels, data1):
    plt.text(x, y + 1, str(y), ha='center', va='bottom', fontsize=10)
for x, y in zip(labels, data2):
    plt.text(x, y - 1, str(y), ha='center', va='top', fontsize=10)

# 显示图表
plt.show()