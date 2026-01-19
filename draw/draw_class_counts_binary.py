# 现有8个类别，每个类别都分为0.3，0.2，0.1三种设置，每种设置都有图片数量和前景类占比两个数据。现在给这些数据绘图，要求x轴为8个类别，y轴主要坐标轴为图片数量，y轴次要坐标轴为前景类占比。
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import matplotlib
# Specify a font that supports Chinese characters
'''
SimHei 黑体 常用于标题。
SimSun 宋体 常用于正文。
Microsoft YaHei 微软雅黑 微软开发的现代中文字体 适合屏幕显示
'''
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # SimHei is a commonly used Chinese font
matplotlib.rcParams['axes.unicode_minus'] = False  # This line is used to ensure that negative signs are displayed correctly

# Specify the path of your CSV file
file_path = 'data_xj_l8.csv'
df = pd.read_csv(file_path)
# Print the DataFrame
print(df)
print(df.columns)

categories = df['类别']
print(categories)
settings = ['30%', '20%', '10%', '5%', '1%', '0.1%', '0%']
image_counts = ['0.3', '0.2', '0.1', '0.05', '0.01', '0.001', '0']
foreground_ratios = ['10.3', '10.2', '10.1', '10.05', '10.01', '10.001', '10']

fig, ax1 = plt.subplots(figsize=(16,5))
# fig, ax1 = plt.subplots()
# Set the title of the figure
fig.suptitle('新疆 Landsat8 二分类数据集概况')

# Create a color gradient
colors = plt.cm.viridis(np.linspace(0.1, 0.4, len(settings)))
alphas = np.linspace(0.2, 0.8, len(settings))

x = np.arange(len(categories))
len_settings = len(settings)
width = 0.1
# draw bar
for i, (setting, image_count) in enumerate(zip(settings, image_counts)):
    ax1.bar(x - len_settings*width/2 + i*width, df[image_count], width, label=setting, color='blue', alpha=alphas[i])

# ax1.set_xlabel('类别')
ax1.set_ylabel('训练集图片数量')
ax1.set_xticks(x)
ax1.set_xticklabels(categories)
ax1.legend()

# draw plot
ax2 = ax1.twinx()
for i, category in enumerate(categories):
    ax2.plot([i - len_settings*width/2 + j*width for j in range(len(settings))], df[df['类别'] == category][foreground_ratios].values.tolist()[0], marker='o', label=category, color='orange')

ax2.set_ylabel('前景类占比')
# ax2.legend()

# Set the y-axis to display percentages
# fmt = '%.0f%%' # Format you want the ticks, e.g. '40%'
# yticks = mtick.FormatStrFormatter(fmt)
# plt.gca().yaxis.set_minor_formatter(yticks)

fig.tight_layout()
plt.savefig("./xj_l8_binary_percent_16x5.png")
# plt.show()
