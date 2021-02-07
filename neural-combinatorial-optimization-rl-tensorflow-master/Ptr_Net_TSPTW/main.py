#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from dataset import DataGenerator
from actor import Actor
from config import get_config, print_config


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

    print("Starting session...")
    with tf.Session() as sess:
        # Run initialize op
        sess.run(tf.global_variables_initializer())

        # Restore variables from disk.
        if config.restore_model == True:
            saver.restore(sess, "save/" + config.restore_from + "/actor.ckpt")
            print("Model restored.")

        # Initialize data generator
        training_set = DataGenerator(config)

        # Training mode
        if not config.inference_mode:

            # Summary writer
            writer = tf.summary.FileWriter(config.log_dir, sess.graph)

            print("Starting training...")
            for i in tqdm(range(config.nb_epoch)):
                # Get feed dict
                input_batch = training_set.train_batch()
                feed = {actor.input_: input_batch}

                # Forward pass & train step
                summary, train_step1, train_step2 = sess.run([actor.merged, actor.train_step1, actor.train_step2],
                                                             feed_dict=feed)
                if i % 100 == 0:
                    writer.add_summary(summary, i)

                # Save the variables to disk
                if i % max(1, int(config.nb_epoch / 5)) == 0 and i != 0:
                    save_path = saver.save(sess, "save/" + config.save_to + "/tmp.ckpt", global_step=i)
                    print("\n Model saved in file: %s" % save_path)

            print("Training COMPLETED !")
            saver.save(sess, "save/" + config.save_to + "/actor.ckpt")


        # Inference mode
        else:

            targets = []
            predictions_length = []
            predictions_delay = []
            predictions_length_w2opt = []
            predictions_2opt = []
            no_predictions_length = []

            # load benchmark instances
            dataset = training_set.load_Dumas(dir_=config.dir_)
            for file_name in dataset:

                # Get feed_dict
                print(file_name)
                or_sequence, tw_open, tw_close = dataset[file_name]['sequence'], dataset[file_name]['tw_open'], \
                                                 dataset[file_name]['tw_close']
                feed = {actor.input_: np.tile(dataset[file_name]['input_'], (config.batch_size, 1, 1))}

                # Initial tour length
                init_tour_length = training_set.get_tour_length(or_sequence)
                no_predictions_length.append(init_tour_length / 100)

                # Solve to optimality
                targets.append(dataset[file_name]['optimal_length'])

                # Sample solutions
                permutations, reward, circuit_length, delay, delivery_time, attending, pointing = sess.run(
                    [actor.positions, actor.reward, actor.distances, actor.delay, actor.constrained_delivery_time,
                     actor.attending, actor.pointing], feed_dict=feed)

                # Find best solution
                j = np.argmin(reward)
                best_permutation = permutations[j][:-1]
                if delay[j] > 0:  # fail
                    print('err2 (Model)', file_name)
                    predictions_length.append(init_tour_length / 100)
                else:
                    predictions_length.append(training_set.get_tour_length(or_sequence[best_permutation]) / 100)
                predictions_delay.append(delay[j])

                # Improve tour with 2 opt
                two_opt_input = np.concatenate(
                    (or_sequence[best_permutation], tw_open[best_permutation], tw_close[best_permutation]), axis=1)
                two_opt_output, two_opt_length = training_set.loop2opt(two_opt_input, speed=1.0)
                if two_opt_length > 100000000:
                    print('err3 (Model + 2 opt)', file_name)
                    predictions_length_w2opt.append(init_tour_length / 100)
                else:
                    predictions_length_w2opt.append(two_opt_length / 100)

                # 2 opt alone
                two_opt_input_ = np.concatenate((or_sequence[::-1], tw_open[::-1], tw_close[::-1]), axis=1)
                two_opt_output_, two_opt_length_ = training_set.loop2opt(two_opt_input_, speed=1.0)
                if two_opt_length_ > 100000000:
                    print('err4 (2 opt)', file_name)
                    predictions_2opt.append(init_tour_length / 100)
                else:
                    predictions_2opt.append(two_opt_length_ / 100)

                # print, plot corresponding tour
                if delay[j]>0:  # delay[j]>0:
                    # training_set.visualize_sampling(permutations)
                    print('\n Model tour length: ', training_set.get_tour_length(or_sequence[best_permutation]) / 100,
                          '(delay:', delay[j], ')')
                    print('\n w/ 2opt: ', two_opt_length / 100)
                    # print(' * permutation: \n', best_permutation)
                    # print(' * delivery time: \n', np.rint(100*(delivery_time[j]-delivery_time[j][0]))-1)
                    print('\n Optimal tour length: \n', dataset[file_name]['optimal_length'])
                    # training_set.visualize_attention(attending[j])
                    # training_set.visualize_attention(pointing[j])
                    # training_set.visualize_2D_trip(or_sequence[::-1], tw_open[::-1], tw_close[::-1]) # Input
                    training_set.visualize_2D_trip(or_sequence[best_permutation], tw_open[best_permutation],
                                                   tw_close[best_permutation])  # Model
                    training_set.visualize_2D_trip(two_opt_output[:, :2], np.expand_dims(two_opt_output[:, 2], axis=1),
                                                   np.expand_dims(two_opt_output[:, 3], axis=1))  # Model + 2 opt
                    training_set.visualize_2D_trip(dataset[file_name]['optimal_sequence'],
                                                   dataset[file_name]['optimal_tw_open'],
                                                   dataset[file_name]['optimal_tw_close'])  # Optimal

            # Average tour length
            targets = np.asarray(targets)
            predictions_delay = np.asarray(predictions_delay)
            predictions_length = np.asarray(predictions_length)
            predictions_length_w2opt = np.asarray(predictions_length_w2opt)
            predictions_2opt = np.asarray(predictions_2opt)
            no_predictions_length = np.asarray(no_predictions_length)
            print('\n Mean delay:', np.mean(predictions_delay))
            print(' Mean length:', np.mean(predictions_length))
            print(' Mean length w/ 2opt:', np.mean(predictions_length_w2opt))
            print(' Mean length 2opt alone:', np.mean(predictions_2opt))
            print(' Init length:', np.mean(no_predictions_length))
            print(' Target length:', np.mean(targets))

            # Tour lenth ratio
            ratio1 = predictions_length / targets
            ratio2 = predictions_length_w2opt / targets
            ratio3 = predictions_2opt / targets
            ratio4 = no_predictions_length / targets
            print('\n Average deviation (Model): \n', np.mean(ratio1))
            print('\n Average deviation2 (Model+2opt): \n', np.mean(ratio2))
            print('\n Average deviation3 (2opt): \n', np.mean(ratio3))
            print('\n Average deviation4 (None): \n', np.mean(ratio4))

            # Histogram
            n1, bins1, patches1 = plt.hist(ratio1, 50, facecolor='b', alpha=0.75)
            n2, bins2, patches2 = plt.hist(ratio2, 50, facecolor='g', alpha=0.75)
            plt.xlabel('Prediction/target')
            plt.ylabel('Counts')
            plt.title('Comparison to Google OR tools')
            plt.axis([0.9, 1.5, 0, 250])
            plt.grid(True)
            plt.show()


if __name__ == "__main__":
    main()
