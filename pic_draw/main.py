import matplotlib.pyplot as plt

task_num = [20, 40, 60, 80, 100]

ptr_cpu = [3.8933017, 3.660663, 3.7238526, 3.6954377, 3.5574005]
ga_cpu = [3.861599102751587, 3.7685912314353787, 3.8123521862091265, 3.82755257592937, 3.8327217207791064]
rand_cpu = [4.211168310018303, 4.100700997321953, 4.090138930086191, 4.030134237915362, 4.02309572207618]

ptr_io = [3.9408202, 3.610669, 3.6317906, 3.6487236, 3.6003392]
ga_io = [3.9408461154426613, 3.8128113140864017, 3.7924782425085453, 3.800755405565075, 3.8379869256545347]
rand_io = [4.182014690752087, 4.078542756262037, 4.070440591519777, 4.024984253168297, 4.046583615087014]

ptr_bandwidth = [3.816117, 3.6688144, 3.731749, 3.6512916, 3.618073]
ga_bandwidth = [3.908574295566651, 3.7371051114383844, 3.8038561171019714, 3.8250426197841696, 3.829901924197536]
rand_bandwidth = [4.204406607778505, 4.137617878456583, 4.045189280990888, 4.084844307317222, 4.025836633347128]

ptr_memory = [3.7174535, 3.7357903, 3.7113655, 3.7529635, 3.6251538]
ga_memory = [3.7709813142571966, 3.7916554955483917, 3.8216328449920995, 3.8636772573941136, 3.8535522613603788]
rand_memory = [4.2064746842053715, 4.103123649782947, 4.11583031222254, 4.035002232606965, 4.038638176214173]

ptr_priority = [0.78945315, 0.8017969, 0.7539601, 0.85239945, 0.77382594]
ga_priority = [0.8100156249999999, 0.8960625, 0.8901866319444445, 0.9034541015625001, 0.907294375]
rand_priority = [2.9397825520833334, 2.847948752170139, 2.766136869855967, 2.6243834838867186, 2.7074515562499997]

ptr_timeout = [0.015625, 0.0074999994, 0.023125, 0.0032421877, 0.014687499]
ga_timeout = [0.01515625, 0.02453125, 0.031770833333333345, 0.034765625, 0.030406250000000017]
rand_timeout = [0.0625, 0.051953125, 0.05260416666666667, 0.044628906249999996, 0.05414062500000001]

ptr_time = [2.721092700958252, 4.453427076339722, 4.94178318977356, 8.522253513336182, 10.860414743423462]
ga_time = [7.616686820983887, 14.371088743209839, 17.452942848205566, 27.72668766975403, 34.46977353096008]

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['KaiTi']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
plt.rcParams['savefig.dpi'] = 300  # 图片像素
plt.rcParams['figure.dpi'] = 300  # 分辨率

fig = plt.figure()
plt.plot(task_num, ptr_cpu, c='red', label=u'指针网络', marker='o')
plt.plot(task_num, ga_cpu, c='blue', label=u'遗传算法', marker='>')
plt.plot(task_num, rand_cpu, c='green', label=u'随机算法', marker='*')
plt.title(u"目标1.1：CPU")
plt.xlabel('任务数(个)')
plt.ylabel('reward-1(cpu)')
plt.legend()
fig.show()

fig = plt.figure()
plt.plot(task_num, ptr_io, c='red', label=u'指针网络', marker='o')
plt.plot(task_num, ga_io, c='blue', label=u'遗传算法', marker='>')
plt.plot(task_num, rand_io, c='green', label=u'随机算法', marker='*')
plt.title(u"目标1.2：I/O")
plt.xlabel('任务数(个)')
plt.ylabel('reward-1(i/o)')
plt.legend()
fig.show()

fig = plt.figure()
plt.plot(task_num, ptr_bandwidth, c='red', label=u'指针网络', marker='o')
plt.plot(task_num, ga_bandwidth, c='blue', label=u'遗传算法', marker='>')
plt.plot(task_num, rand_bandwidth, c='green', label=u'随机算法', marker='*')
plt.title(u"目标1.3：带宽")
plt.xlabel('任务数(个)')
plt.ylabel('reward-1(bandwidth)')
plt.legend()
fig.show()

fig = plt.figure()
plt.plot(task_num, ptr_memory, c='red', label=u'指针网络', marker='o')
plt.plot(task_num, ga_memory, c='blue', label=u'遗传算法', marker='>')
plt.plot(task_num, rand_memory, c='green', label=u'随机算法', marker='*')
plt.title(u"目标1.4：内存")
plt.xlabel('任务数(个)')
plt.ylabel('reward-1(memory)')
plt.legend()
fig.show()

fig = plt.figure()
plt.plot(task_num, ptr_priority, c='red', label=u'指针网络', marker='o')
plt.plot(task_num, ga_priority, c='blue', label=u'遗传算法', marker='>')
plt.plot(task_num, rand_priority, c='green', label=u'随机算法', marker='*')
plt.title(u"目标2：任务优先级")
plt.xlabel('任务数(个)')
plt.ylabel('reward-2')
plt.legend()
fig.show()

fig = plt.figure()
plt.plot(task_num, ptr_timeout, c='red', label=u'指针网络', marker='o')
plt.plot(task_num, ga_timeout, c='blue', label=u'遗传算法', marker='>')
plt.plot(task_num, rand_timeout, c='green', label=u'随机算法', marker='*')
plt.title(u"目标3：超时率")
plt.xlabel('任务数(个)')
plt.ylabel('reward-3')
plt.legend()
fig.show()

fig = plt.figure()
plt.plot(task_num, ptr_time, c='red', label=u'指针网络', marker='o')
plt.plot(task_num, ga_time, c='blue', label=u'遗传算法', marker='>')
plt.title(u"运行时间对比")
plt.xlabel('任务数(个)')
plt.ylabel('时间(秒)')
plt.legend()
fig.show()
