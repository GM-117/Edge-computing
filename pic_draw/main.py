import matplotlib.pyplot as plt

task_num = [5, 6, 7, 8, 9]

ptr_total = [2.51305,2.1186851,1.9074863,1.8241217,1.6733654]
greed_total = [2.6853593750000003,  2.30845,2.0959859375000002,1.897428125,1.7323472222222223]
rand_total = [2.7469765625000003, 2.3492296874999997, 2.0949906250000002,  1.9636968749999997,1.8011078125000002]
queue_total = [2.7396890625000004, 2.2979421875,2.0979718749999997,2.0110296874999998,1.7818463541666667]

ptr_time = [2.1130623, 1.7222499,1.5624374,1.452328,1.3136041]
greed_time = [2.2281249999999993, 1.8674999999999998,1.6881250000000002,1.486796875, 1.3534722222222222]
rand_time = [2.2574999999999997, 1.8725, 1.645625,1.5274999999999999,1.390625]
queue_time = [2.18375, 1.799375,1.59125, 1.5153124999999998,1.3272916666666668]

ptr_priority = [0.22005, 0.22518519,0.19623625,0.2320436,0.23026128]
greed_priority = [0.251609375, 0.260325, 0.2528609375,0.25125625,0.24512499999999998]
rand_priority = [0.25385156250000007, 0.26110468750000004,0.25436562500000004,0.248071875,0.24923281249999998]
queue_priority = [0.23656406249999998,0.23656406249999998,0.24234687500000002,0.24071718749999998,0.24142968750000002]

ptr_timeout = [0.1799375,0.17125002, 0.1488125,0.13975,0.1295]
greed_timeout = [ 0.205625,0.18062499999999998,0.15499999999999997,0.159375,0.13375]
rand_timeout = [0.23562499999999992,0.215625,0.195,0.18812499999999996, 0.16124999999999998]
queue_timeout = [0.319375,0.2799999999999999,0.2643750000000001,0.255,0.213125]

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
plt.xlabel('服务器负载α')
plt.ylabel('reward')
plt.legend()
fig.show()

fig = plt.figure()
plt.plot(task_num, ptr_time, c='red', label=u'指针网络', marker='o')
plt.plot(task_num, greed_time, c='blue', label=u'贪心算法', marker='>')
plt.plot(task_num, rand_time, c='green', label=u'随机算法', marker='*')
plt.plot(task_num, queue_time, c='orange', label=u'多级反馈队列算法', marker='D')
plt.title(u"目标1:运行时间")
plt.xlabel('服务器负载α)')
plt.ylabel('reward1 / n')
plt.legend()
fig.show()

fig = plt.figure()
plt.plot(task_num, ptr_priority, c='red', label=u'指针网络', marker='o')
plt.plot(task_num, greed_priority, c='blue', label=u'贪心算法', marker='>')
plt.plot(task_num, rand_priority, c='green', label=u'随机算法', marker='*')
plt.plot(task_num, queue_priority, c='orange', label=u'多级反馈队列算法', marker='D')
plt.title(u"目标2:优先级")
plt.xlabel('服务器负载α')
plt.ylabel('reward2 / n')
plt.legend()
fig.show()

fig = plt.figure()
plt.plot(task_num, ptr_timeout, c='red', label=u'指针网络', marker='o')
plt.plot(task_num, greed_timeout, c='blue', label=u'贪心算法', marker='>')
plt.plot(task_num, rand_timeout, c='green', label=u'随机算法', marker='*')
plt.plot(task_num, queue_timeout, c='orange', label=u'多级反馈队列算法', marker='D')
plt.title(u"目标3:超时率")
plt.xlabel('服务器负载α')
plt.ylabel('reward3')
plt.legend()
fig.show()
