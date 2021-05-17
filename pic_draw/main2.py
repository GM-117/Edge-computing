import matplotlib.pyplot as plt

task_num = []
for i in range(20):
    task_num.append(1 - (i / 20))

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['KaiTi']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
plt.rcParams['savefig.dpi'] = 300  # 图片像素
plt.rcParams['figure.dpi'] = 300  # 分辨率

fig = plt.figure()
plt.plot(task_num)
plt.legend()
fig.show()
