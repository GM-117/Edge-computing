#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
from numpy import *
import matplotlib.pyplot as plt
from tqdm import tqdm

from Ptr_Net_TSPTW.dataset import DataGenerator
from Ptr_Net_TSPTW.actor import Actor
from Ptr_Net_TSPTW.config import get_config, print_config
from Ptr_Net_TSPTW.ga import do_ga
from Ptr_Net_TSPTW.rand import do_rand


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

    # Saver to save & restore all the variables.
    variables_to_save = [v for v in tf.global_variables() if 'Adam' not in v.name]
    saver = tf.train.Saver(var_list=variables_to_save, keep_checkpoint_every_n_hours=1.0)

    predictions = []
    cpu = []
    io = []
    bandwidth = []
    memory = []
    task_priority = []
    ns_ = []

    training_set = DataGenerator(config)
    input_batch = training_set.train_batch()

    with tf.Session() as sess:
        # Run initialize op
        sess.run(tf.global_variables_initializer())

        # Restore variables from disk.
        if config.restore_model == True:
            saver.restore(sess, "save/" + config.restore_from + "/actor.ckpt")
            print("Model restored.")

        # 训练
        if not config.inference_mode:

            # Summary writer
            writer = tf.summary.FileWriter(config.log_dir, sess.graph)

            print("Starting training...")
            for i in tqdm(range(config.nb_epoch)):
                # Get feed dict
                feed = {actor.input_: input_batch}

                # Forward pass & train step
                reward, cpu_sum, io_sum, bandwidth_sum, memory_sum, task_priority_sum, ns, summary, train_step1, train_step2, train_step1_task, train_step2_task, train_step1_time, train_step2_time = sess.run(
                    [actor.result, actor.cpu_sum, actor.io_sum, actor.bandwidth_sum, actor.memory_sum,
                     actor.task_priority_sum, actor.ns_prob, actor.merged,
                     actor.train_step1, actor.train_step2,
                     actor.train_step1_task, actor.train_step2_task, actor.train_step1_time, actor.train_step2_time],
                    feed_dict=feed)

                # reward, cpu_sum, io_sum, bandwidth_sum, memory_sum, task_priority_sum, ns, summary, train_step1, train_step2 = sess.run(
                #     [actor.reward, actor.cpu_sum, actor.io_sum, actor.bandwidth_sum, actor.memory_sum,
                #      actor.task_priority_sum, actor.ns_prob, actor.merged,
                #      actor.train_step1, actor.train_step2,
                #      ],
                #     feed_dict=feed)

                reward_mean = np.mean(reward)
                cpu_mean = np.mean(cpu_sum)
                io_mean = np.mean(io_sum)
                bandwidth_mean = np.mean(bandwidth_sum)
                memory_mean = np.mean(memory_sum)
                task_priority_mean = np.mean(task_priority_sum)
                ns_mean = np.mean(ns)

                predictions.append(reward_mean)
                cpu.append(cpu_mean)
                io.append(io_mean)
                bandwidth.append(bandwidth_mean)
                memory.append(memory_mean)
                task_priority.append(task_priority_mean)
                ns_.append(ns_mean)

                if i % 1000 == 0:
                    writer.add_summary(summary, i)

                # Save the variables to disk
                if i % 1000 == 0 and i != 0:
                    save_path = saver.save(sess, "save/" + config.save_to + "/tmp.ckpt", global_step=i)
                    print("\n Model saved in file: %s" % save_path)

            print("Training COMPLETED !")
            saver.save(sess, "save/" + config.save_to + "/actor.ckpt")

        # 测试
        else:
            predictions = []
            # Get feed_dict (single input)
            feed = {actor.input_: input_batch}

            # Sample solutions
            reward, cpu_sum, io_sum, bandwidth_sum, memory_sum, task_priority_sum, ns, summary, train_step1, train_step2, train_step1_task, train_step2_task, train_step1_time, train_step2_time = sess.run(
                [actor.result, actor.cpu_sum, actor.io_sum, actor.bandwidth_sum, actor.memory_sum,
                 actor.task_priority_sum, actor.ns_prob, actor.merged,
                 actor.train_step1, actor.train_step2,
                 actor.train_step1_task, actor.train_step2_task, actor.train_step1_time, actor.train_step2_time],
                feed_dict=feed)

            reward_mean = np.mean(reward)
            cpu_mean = np.mean(cpu_sum)
            io_mean = np.mean(io_sum)
            bandwidth_mean = np.mean(bandwidth_sum)
            memory_mean = np.mean(memory_sum)
            task_priority_mean = np.mean(task_priority_sum)
            ns_mean = np.mean(ns)

            predictions.append(reward_mean)
            cpu.append(cpu_mean)
            io.append(io_mean)
            bandwidth.append(bandwidth_mean)
            memory.append(memory_mean)
            task_priority.append(task_priority_mean)
            ns_.append(ns_mean)

    ga_result, ga_cpu_result, ga_io_result, ga_bandwidth_result, ga_memory_result, ga_task_priority_result, ga_ns_result \
        = do_ga(input_batch)

    # ga_result = ga_cpu_result = ga_io_result = ga_bandwidth_result = ga_memory_result = ga_task_priority_result = ga_ns_result = []
    rand_result, rand_server_ratio_result, rand_task_priority_result, rand_ns_result = do_rand(input_batch)

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
    plt.plot(list(range(len(cpu))), cpu, c='red', label=u'指针网络')
    plt.plot(list(range(len(ga_cpu_result))), ga_cpu_result, c='blue', label=u'遗传算法')
    plt.title(u"目标1.1：CPU")
    plt.xlabel('轮数')
    plt.legend()
    fig.show()

    fig = plt.figure()
    plt.plot(list(range(len(io))), io, c='red', label=u'指针网络')
    plt.plot(list(range(len(ga_io_result))), ga_io_result, c='blue', label=u'遗传算法')
    plt.title(u"目标1.2：I/O")
    plt.xlabel('轮数')
    plt.legend()
    fig.show()

    fig = plt.figure()
    plt.plot(list(range(len(bandwidth))), bandwidth, c='red', label=u'指针网络')
    plt.plot(list(range(len(ga_bandwidth_result))), ga_bandwidth_result, c='blue', label=u'遗传算法')
    plt.title(u"目标1.3：带宽")
    plt.xlabel('轮数')
    plt.legend()
    fig.show()

    fig = plt.figure()
    plt.plot(list(range(len(memory))), memory, c='red', label=u'指针网络')
    plt.plot(list(range(len(ga_memory_result))), ga_memory_result, c='blue', label=u'遗传算法')
    plt.title(u"目标1.4：内存")
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

    print('task:', config.max_length)
    print('gen_num:', config.gen_num)
    print('nb_epoch:', config.nb_epoch)
    print('ptr')
    print('目标1.1：CPU', mean(cpu[-10:]))
    print('目标1.2：I/O', mean(io[-10:]))
    print('目标1.3：带宽', mean(bandwidth[-10:]))
    print('目标1.4：内存', mean(memory[-10:]))
    print('目标2：任务优先级', mean(task_priority[-10:]))
    print('目标3：超时率', mean(ns_[-10:]))
    print('ga')
    print('目标1.1：CPU', mean(ga_cpu_result[-10:]))
    print('目标1.2：I/O', mean(ga_io_result[-10:]))
    print('目标1.3：带宽', mean(ga_bandwidth_result[-10:]))
    print('目标1.4：内存', mean(ga_memory_result[-10:]))
    print('目标2：任务优先级', mean(ga_task_priority_result[-10:]))
    print('目标3：超时率', mean(ga_ns_result[-10:]))


if __name__ == "__main__":
    main()
