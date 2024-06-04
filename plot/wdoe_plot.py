import matplotlib.pyplot as plt

# 模型名称
models = ['CIFAR-10', 'CIFAR-100']

# 方法名称
methods = ['regret', 'risk', 'random']

# 每个模型对应的数值，共5个模型，每个模型对应7种方法的数值

# cifar10
values = [
    [4.66, 5.22, 7.53],
    [22.37, 28.38, 32.25],
]


# 颜色对应的 RGB 值
colors = [
    (54, 81, 170),
    (0, 132, 132),
    (64, 193, 39),
    # (254, 224, 139),
    # (255, 191, 0),
    # (255, 121, 4),
    # (255, 0, 4)
]

markers = ['o', 's', 'd']

def rgb_to_hex(rgb):
    r, g, b = rgb
    return "#{:02x}{:02x}{:02x}".format(r, g, b)

# 绘制折线图
fig, ax = plt.subplots(figsize=(5 * 1.4, 3.5 * 1.4))

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

ax.grid(ls='--', lw=1.5 * 2)
plt.tight_layout() 
plt.savefig('./test.png', dpi=1024)
plt.show()
