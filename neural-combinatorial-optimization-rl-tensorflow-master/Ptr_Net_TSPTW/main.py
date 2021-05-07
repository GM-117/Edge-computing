#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

from Ptr_Net_TSPTW.dataset import DataGenerator
from Ptr_Net_TSPTW.actor import Actor
from Ptr_Net_TSPTW.config import get_config, print_config
from Ptr_Net_TSPTW.rand import do_rand
from Ptr_Net_TSPTW.multy import do_multy


# Model: Decoder inputs = Encoder outputs Critic design (state value function approximator) = RNN encoder last hidden
# state (c) (parametric baseline ***) + 1 glimpse over (c) at memory states + 2 FFN layers (ReLu),
# w/o moving_baseline (init_value = 7 for TSPTW20) Penalty: Discrete (counts) with beta = +3 for one constraint /
# beta*sqrt(N) for N constraints violated (concave **0.5) No Regularization Decoder Glimpse = on Attention_g (mask -
# current) Residual connections 01

# NEW data generator (wrap config.py)
# speed1000 model: n20w100
# speed10 model: s10_k5_n20w100 (fine tuned w/ feasible kNN datagen)
# Benchmark: Dumas n20w100 instances


def main():
    # Get running configuration
    config, _ = get_config()
    print_config()

    # Build tensorflow graph from config
    print("Building graph...")
    actor = Actor(config)

    predictions = []
    time_used = []
    task_priority = []
    ns_ = []

    training_set = DataGenerator(config)
    tasks_input_batch, servers_input_batch, server_allocate = training_set.train_batch()

    with tf.Session() as sess:
        # Run initialize op
        sess.run(tf.global_variables_initializer())

        # 训练
        if not config.inference_mode:

            print("Starting training...")
            for i in tqdm(range(config.nb_epoch)):
                # Get feed dict
                feed = {actor.input_: tasks_input_batch, actor.server_input_: servers_input_batch}
                # Forward pass & train step

                result, time_use, task_priority_sum, ns_prob, summary, train_step1, train_step2 = sess.run(
                    [actor.reward, actor.time_use, actor.task_priority_sum, actor.ns_prob, actor.merged,
                     actor.train_step1, actor.train_step2],
                    feed_dict=feed)

                time_use = time_use / 10
                ns_prob = ns_prob / 5
                result = time_use + task_priority_sum + ns_prob
                reward_mean = np.mean(result)
                time_mean = np.mean(time_use)
                task_priority_mean = np.mean(task_priority_sum)
                ns_mean = np.mean(ns_prob)

                predictions.append(reward_mean)
                time_used.append(time_mean)
                task_priority.append(task_priority_mean)
                ns_.append(ns_mean)

            print("Training COMPLETED !")

    ga_result = ga_time_result = ga_task_priority_result = ga_ns_result = []

    # 解决中文显示问题
    plt.rcParams['font.sans-serif'] = ['KaiTi']  # 指定默认字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

    fig = plt.figure()
    plt.plot(list(range(len(predictions))), predictions, c='red', label=u'指针网络')
    plt.plot(list(range(len(ga_result))), ga_result, c='blue', label=u'遗传算法')
    plt.title(u"效果曲线")
    plt.xlabel('轮数')
    plt.legend()
    fig.show()

    fig = plt.figure()
    plt.plot(list(range(len(time_used))), time_used, c='red', label=u'指针网络')
    plt.plot(list(range(len(ga_time_result))), ga_time_result, c='blue', label=u'遗传算法')
    plt.title(u"目标1：运行时间")
    plt.xlabel('轮数')
    plt.legend()
    fig.show()

    fig = plt.figure()
    plt.plot(list(range(len(task_priority))), task_priority, c='red', label=u'指针网络')
    plt.plot(list(range(len(ga_task_priority_result))), ga_task_priority_result, c='blue', label=u'遗传算法')
    plt.title(u"目标2：任务优先级")
    plt.xlabel('轮数')
    plt.legend()
    fig.show()

    fig = plt.figure()
    plt.plot(list(range(len(ns_))), ns_, c='red', label=u'指针网络')
    plt.plot(list(range(len(ga_ns_result))), ga_ns_result, c='blue', label=u'遗传算法')
    plt.title(u"目标3：超时率")
    plt.xlabel('轮数')
    plt.legend()
    fig.show()

    rand_result, rand_time_result, rand_task_priority_result, rand_ns_result = do_rand(tasks_input_batch,
                                                                                       servers_input_batch, 0)
    greed_result, greed_1_result, greed_2_result, greed_3_result = do_rand(tasks_input_batch, servers_input_batch, 1)
    multy_result, multy_1_result, multy_2_result, multy_3_result = do_multy(tasks_input_batch, servers_input_batch)

    print('task:', config.task_num)
    print('server:', config.server_num)
    print('capacity:', config.server_capacity)
    print('ptr')
    print('综合效果', np.mean(predictions[-10:]))
    print('目标1：运行时间', np.mean(time_used[-10:]))
    print('目标2：任务优先级', np.mean(task_priority[-10:]))
    print('目标3：超时率', np.mean(ns_[-10:]))
    print('greed')
    print('综合效果', np.mean(greed_result[-10:]))
    print('目标1：运行时间', np.mean(greed_1_result[-10:]))
    print('目标2：任务优先级', np.mean(greed_2_result[-10:]))
    print('目标3：超时率', np.mean(greed_3_result[-10:]))
    print('rand')
    print('综合效果', np.mean(rand_result[-10:]))
    print('目标1：运行时间', np.mean(rand_time_result[-10:]))
    print('目标2：任务优先级', np.mean(rand_task_priority_result[-10:]))
    print('目标3：超时率', np.mean(rand_ns_result[-10:]))
    print('multy')
    print('综合效果', np.mean(multy_result[-10:]))
    print('目标1：运行时间', np.mean(multy_1_result[-10:]))
    print('目标2：任务优先级', np.mean(multy_2_result[-10:]))
    print('目标3：超时率', np.mean(multy_3_result[-10:]))


if __name__ == "__main__":
    main()
