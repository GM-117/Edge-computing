#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import time
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
    time_used = []
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
                input_batch = training_set.train_batch()
                # Forward pass & train step

                result, time_use, task_priority_sum, ns_prob, summary, train_step1, train_step2, train_step1_task, train_step2_task, train_step1_time, train_step2_time = sess.run(
                    [actor.result, actor.time_use, actor.task_priority_sum, actor.ns_prob, actor.merged,
                     actor.train_step1, actor.train_step2,
                     actor.train_step1_task, actor.train_step2_task, actor.train_step1_time, actor.train_step2_time],
                    feed_dict=feed)

                reward_mean = np.mean(result)
                time_mean = np.mean(time_use)
                task_priority_mean = np.mean(task_priority_sum)
                ns_mean = np.mean(ns_prob)

                predictions.append(reward_mean)
                time_used.append(time_mean)
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
            time_start = time.time()
            # Sample solutions

            result, time_used, task_priority_sum, ns_prob, summary, train_step1, train_step2, train_step1_task, train_step2_task, train_step1_time, train_step2_time = sess.run(
                [actor.result, actor.time_used, actor.task_priority_sum, actor.ns_prob, actor.merged,
                 actor.train_step1, actor.train_step2,
                 actor.train_step1_task, actor.train_step2_task, actor.train_step1_time, actor.train_step2_time],
                feed_dict=feed)

            time_end = time.time()
            print("ptr: ", time_end - time_start)

            reward_mean = np.mean(result)
            time_mean = np.mean(time_used)
            task_priority_mean = np.mean(task_priority_sum)
            ns_mean = np.mean(ns_prob)

            predictions.append(reward_mean)
            time_used.append(time_mean)
            task_priority.append(task_priority_mean)
            ns_.append(ns_mean)

    ga_result, ga_time_result, ga_task_priority_result, ga_ns_result = do_ga(input_batch)

    # ga_result = ga_cpu_result = ga_io_result = ga_bandwidth_result = ga_memory_result = ga_task_priority_result = ga_ns_result = []

    rand_result, rand_time_result, rand_task_priority_result, rand_ns_result = do_rand(input_batch)

    # rand_result = rand_cpu_result = rand_io_result = rand_bandwidth_result = rand_memory_result = rand_task_priority_result = rand_ns_result = []

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

    print('task:', config.max_length)
    print('gen_num:', config.gen_num)
    print('nb_epoch:', config.nb_epoch)
    print('ptr')
    print('目标1：运行时间', mean(time_used[-10:]))
    print('目标2：任务优先级', mean(task_priority[-10:]))
    print('目标3：超时率', mean(ns_[-10:]))
    print('ga')
    print('目标1：运行时间', mean(ga_time_result[-10:]))
    print('目标2：任务优先级', mean(ga_task_priority_result[-10:]))
    print('目标3：超时率', mean(ga_ns_result[-10:]))
    print('rand')
    print('目标1：运行时间', mean(rand_time_result[-10:]))
    print('目标2：任务优先级', mean(rand_task_priority_result[-10:]))
    print('目标3：超时率', mean(rand_ns_result[-10:]))


if __name__ == "__main__":
    main()
