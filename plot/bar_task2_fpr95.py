import matplotlib.pyplot as plt

# 模型名称
# models = ['MobileNet', 'ResNet18', 'ResNet50', 'WideResNet', 'DenseNet161']
models = ['net1', 'net2', 'net3', 'net4', 'net5']

# 方法名称
# methods = ['MSP', 'MaxLogit', 'FES', 'GradNorm', 'MAH', 'ASH', 'KNN']
methods = ['MSP', 'FES', 'GradNorm', 'MAH', 'ASH', 'KNN']

# 每个模型对应的数值，共5个模型，每个模型对应7种方法的数值

# cifar10
values = [
    [77.94, 74.76, 75.42, 61.73, 74.97, 85.48],
    [51.61, 38.37, 45.04, 19.76, 38.24, 45.24],
    [47.09, 34.66, 41.31, 36.76, 42.80, 49.99],
    [44.34, 25.63, 37.03, 20.12, 28.87, 82.65],
    [52.06, 32.00, 40.47, 0.42, 32.17, 71.91]
]

# cifar100
# values = [
#     [77.73, 69.79, 71.25, 71.94, 74.59, 88.79],
#     [65.81, 62.09, 61.70, 84.46, 61.43, 67.47],
#     [57.95, 39.82, 41.22, 81.75, 48.87, 63.34],
#     [59.43, 45.35, 46.19, 68.78, 47.81, 66.20],
#     [50.47, 34.49, 35.27, 58.36, 40.47, 89.74]
# ]


# 颜色对应的 RGB 值
colors = [
    (54, 81, 170),
    (0, 132, 132),
    (64, 193, 39),
    (254, 224, 139),
    (255, 191, 0),
    (255, 121, 4),
    (255, 0, 4)
]

markers = ['o', 's', 'd', '^', '*', 'p', 'h']

def rgb_to_hex(rgb):
    r, g, b = rgb
    return "#{:02x}{:02x}{:02x}".format(r, g, b)

# # 绘制折线图
# fig, ax = plt.subplots()

# for i in range(len(methods)):
#     color = rgb_to_hex(colors[i])
#     ax.plot(models, [value[i] for value in values], label=methods[i], color=color, marker=markers[i], linewidth=1.5)

# # 添加标题和标签
# ax.set_xlabel('Backbones', fontsize=14)
# ax.set_ylabel('FPR95', fontsize=14)
# ax.tick_params(axis='x', labelsize=12)
# ax.tick_params(axis='y', labelsize=12)
# ax.set_title('Different backbones for CIFAR-100', fontsize=16)
# ax.legend()

# ax.grid(ls='--', lw=0.5)

# plt.savefig('./task2_cifar100_fpr95.svg', dpi=1024)
# plt.show()

# 绘制折线图
fig, ax = plt.subplots(figsize=(5 * 1.4,3.5 * 1.4))

[x.set_linewidth(1.5 * 2) for x in ax.spines.values()]

for i in range(len(methods)):
    color = rgb_to_hex(colors[i])
    ax.plot(models, [value[i] for value in values], label=methods[i], color=color, marker=markers[i], linewidth=2 * 2, markersize=13)

# 添加标题和标签
ax.set_xlabel('Datasets', fontsize=25)
ax.set_ylabel('FPR95', fontsize=25)
ax.tick_params(axis='x', labelsize=25)
ax.tick_params(axis='y', labelsize=25)
ax.legend(fontsize=22, loc='upper right')
# ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=3, fancybox=True, shadow=True,fontsize=20)

ax.grid(ls='--', lw=1.5 * 2)
plt.tight_layout() 
plt.savefig('./task2_cifar10_fpr95.svg', dpi=1024)
plt.show()

