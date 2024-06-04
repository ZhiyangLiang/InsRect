import matplotlib.pyplot as plt

# 数据集名称
datasets = ['Texture', 'Places', 'LSUN', 'iSUN', 'SVHN']

# 方法名称
# methods = ['MSP', 'MaxLogit', 'FES', 'GradNorm', 'MAH', 'ASH', 'KNN']
methods = ['MSP', 'FES', 'GradNorm', 'MAH', 'ASH', 'KNN']

# 每个模型对应的数值，共5个模型，每个模型对应7种方法的数值
# cifar10
values = [
    [49.80, 40.46, 41.72, 32.41, 45.39, 39.59],
    [43.68, 34.88, 37.82, 30.35, 42.77, 84.07],
    [48.92, 34.24, 41.12, 36.50, 40.17, 29.43],
    [52.80, 35.69, 43.56, 49.41, 42.94, 39.79],
    [50.30, 32.24, 42.31, 46.74, 40.67, 57.06]
]

# values = [
#     [49.80, 43.68, 48.92, 52.80, 50.30],
#     [41.10, 35.05, 35.57, 37.09, 33.77],
#     [40.46, 34.88, 34.24, 35.69, 32.24],
#     [41.72, 37.82, 41.12, 43.56, 42.31],//
#     [32.41, 30.35, 36.50, 49.41, 46.74],
#     [45.39, 42.77, 40.17, 42.94, 40.67],
#     [39.59, 84.07, 29.43, 39.79, 57.06]
# ]

# cifar100
# values = [
#     [61.28, 52.09, 48.92, 46.54, 64.61, 49.38, 45.14],
#     [54.42, 43.81, 39.83, 38.95, 73.18, 44.65, 94.20],
#     [58.57, 43.50, 37.54, 38.95, 82.88, 42.89, 65.75],
#     [62.34, 49.86, 43.93, 44.26, 94.36, 52.45, 86.52],
#     [57.79, 42.95, 37.15, 37.41, 89.79, 43.84, 25.12]
# ]

# values = [
#     [61.28, 54.42, 58.57, 62.34, 57.79],
#     [52.09, 43.81, 43.50, 49.86, 42.95],
#     [48.92, 39.83, 37.54, 43.93, 37.15],
#     [46.54, 38.95, 38.95, 44.26, 37.41],//
#     [64.61, 73.18, 82.88, 94.36, 89.79],
#     [49.38, 44.65, 42.89, 52.45, 43.84],
#     [45.14, 94.20, 65.75, 86.52, 25.12]
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

# for i in range(len(datasets)):
#     color = rgb_to_hex(colors[i])
#     ax.plot(methods, [value[i] for value in values], label=datasets[i], color=color, marker=markers[i], linewidth=1.5)

# # 添加标题和标签
# ax.set_xlabel('Datasets', fontsize=14)
# ax.set_ylabel('FPR95', fontsize=14)
# ax.tick_params(axis='x', labelsize=12)
# ax.tick_params(axis='y', labelsize=12)
# ax.set_title('Different Datasets for CIFAR-100', fontsize=16)
# ax.legend()

# ax.grid(ls='--', lw=0.5)

# plt.savefig('./new_task4_cifar100_fpr95.svg', dpi=1024)
# plt.show()

# 绘制折线图
fig, ax = plt.subplots(figsize=(5 * 1.4,3.5 * 1.4))

[x.set_linewidth(1.5 * 2) for x in ax.spines.values()]

for i in range(len(methods)):
    color = rgb_to_hex(colors[i])
    ax.plot(datasets, [value[i] for value in values], label=methods[i], color=color, marker=markers[i], linewidth=2 * 2, markersize=13)

# 添加标题和标签
ax.set_xlabel('Datasets', fontsize=25)
ax.set_ylabel('FPR95', fontsize=25)
ax.tick_params(axis='x', labelsize=25)
ax.tick_params(axis='y', labelsize=25)
ax.legend(fontsize=22, loc='upper right')
# ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=3, fancybox=True, shadow=True,fontsize=20)

ax.grid(ls='--', lw=1.5 * 2)
plt.tight_layout() 
plt.savefig('./new_task4_cifar10_fpr95.svg', dpi=1024)
plt.show()
