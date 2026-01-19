import matplotlib.pyplot as plt
def draw_scatter(x, y, title, label='feature map', size_point=50):
    plt.scatter(x, y, label=label, s=size_point)
    # title
    plt.title(title)
    plt.legend()
    plt.savefig(label + '.png')

x = [1, 2, 3]
y = [1, 2, 3]
title = 'test'
draw_scatter(x, y, title, 'test1', size_point=100)

x2 = [50, 10, 13]
y2 = [10, 20, 30]
draw_scatter(x2, y2, title, 'test2', size_point=200)

x3 = [100, 200, 350]
y3 = [110, 220, 320]
draw_scatter(x3, y3, title, 'test3', size_point=300)

print('done')