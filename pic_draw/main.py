import matplotlib.pyplot as plt

task_num = [5, 10, 15]

ptr_total = [2.9585023, 2.8204627, 2.7238255]
ptr_time = [2.5426497, 2.446725, 2.379773]
ptr_priority = [0.22678986, 0.21673748, 0.2175376]
ptr_timeout = [0.1890625, 0.15699999, 0.12651515]

greed_total = [3.0094125, 2.9776312500000004, 2.9433310376492194]
greed_time = [2.5549999999999997, 2.5294999999999996, 2.4700757575757573]
greed_priority = [0.2519125, 0.24813125, 0.2630280073461892]
greed_timeout = [0.20250000000000004, 0.2, 0.2102272727272727]

rand_total = [3.1028156249999994, 3.0482124999999995, 3.0654929981634527]
rand_time = [2.61775, 2.5869999999999997, 2.56780303030303]
rand_priority = [0.24881562500000004, 0.25121249999999995, 0.26284148301193755]
rand_timeout = [0.23625000000000002, 0.21000000000000002, 0.23484848484848486]

queue_total = [3.1794265625000007, 3.0308187500000003, 2.9053661616161612]
queue_time = [2.6189999999999998, 2.5170000000000003, 2.4132575757575756]
queue_priority = [0.23980156249999998, 0.23506875000000002, 0.24400252525252525]
queue_timeout = [0.32062500000000005, 0.27875000000000005, 0.2481060606060606]

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['KaiTi']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
plt.rcParams['savefig.dpi'] = 300  # 图片像素
plt.rcParams['figure.dpi'] = 300  # 分辨率

fig = plt.figure()
plt.plot(task_num, ptr_total, c='red', label=u'指针网络', marker='o')
plt.plot(task_num, greed_total, c='blue', label=u'贪心算法', marker='>')
plt.plot(task_num, rand_total, c='green', label=u'随机算法', marker='*')
plt.plot(task_num, queue_total, c='orange', label=u'多级反馈队列算法', marker='D')
plt.title(u"reward:综合目标")
plt.xlabel('服务器数m(个)')
plt.ylabel('reward')
plt.legend()
fig.show()

fig = plt.figure()
plt.plot(task_num, ptr_time, c='red', label=u'指针网络', marker='o')
plt.plot(task_num, greed_time, c='blue', label=u'贪心算法', marker='>')
plt.plot(task_num, rand_time, c='green', label=u'随机算法', marker='*')
plt.plot(task_num, queue_time, c='orange', label=u'多级反馈队列算法', marker='D')
plt.title(u"目标1:运行时间")
plt.xlabel('服务器数m(个)')
plt.ylabel('reward1 / n')
plt.legend()
fig.show()

fig = plt.figure()
plt.plot(task_num, ptr_priority, c='red', label=u'指针网络', marker='o')
plt.plot(task_num, greed_priority, c='blue', label=u'贪心算法', marker='>')
plt.plot(task_num, rand_priority, c='green', label=u'随机算法', marker='*')
plt.plot(task_num, queue_priority, c='orange', label=u'多级反馈队列算法', marker='D')
plt.title(u"目标2:优先级")
plt.xlabel('服务器数m(个)')
plt.ylabel('reward2 / n')
plt.legend()
fig.show()

fig = plt.figure()
plt.plot(task_num, ptr_timeout, c='red', label=u'指针网络', marker='o')
plt.plot(task_num, greed_timeout, c='blue', label=u'贪心算法', marker='>')
plt.plot(task_num, rand_timeout, c='green', label=u'随机算法', marker='*')
plt.plot(task_num, queue_timeout, c='orange', label=u'多级反馈队列算法', marker='D')
plt.title(u"目标3:超时率")
plt.xlabel('服务器数m(个)')
plt.ylabel('reward3')
plt.legend()
fig.show()
