#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from dataset import DataGenerator
from actor import Actor
from config import get_config, print_config
from tsp_with_ortools import Solver


# Model: Critic (state value function approximator) = slim mean Attentive (parametric baseline ***)
#        w/ moving baseline (init_value default = 7 for TSP20, 20 for TSP40)
#        Encoder = w/ FFN ([3] num_stacks / [16] num_heads / inner_FFN = 4*hidden_dim / [0.1] dropout_rate)
#        Decoder init_state = train, mean(enc)
#        Decoder inputs = Encoder outputs
#        Decoder Glimpse = Attention_g on ./(mask - first) + Residual connection


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
        solver = Solver(actor.max_length)  ###### ######
        training_set = DataGenerator(solver)

        # Training mode
        if not config.inference_mode:

            # Summary writer
            writer = tf.summary.FileWriter(config.log_dir, sess.graph)

            print("Starting training...")
            for i in tqdm(range(config.nb_epoch)):
                # Get feed dict
                input_batch = training_set.train_batch(actor.batch_size, actor.max_length, actor.input_dimension)
                feed = {actor.input_: input_batch}
                # Forward pass & train step
                summary, permutation, distances = sess.run([actor.merged, actor.positions, actor.distances],
                                                           feed_dict=feed)
                print(' Permutation \n', permutation)
                print(' Tour length \n', distances)

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
            predictions = []

            for __ in tqdm(range(1000)):  # num of examples

                # Get feed_dict (single input)
                seed_ = 1 + __
                input_batch, or_sequence = training_set.test_batch(actor.batch_size, actor.max_length,
                                                                   actor.input_dimension,
                                                                   seed=seed_)  # seed=0 means None
                feed = {actor.input_: input_batch}

                # Solve instance (OR tools)
                opt_trip, opt_length = training_set.solve_instance(or_sequence)
                targets.append(opt_length / 100)
                # print('\n Optimal length:',opt_length/100)

                ################################### UMPA LOOOOP HERE ###################################    nb_loop / temperature

                # Sample solutions
                permutations, circuit_length = sess.run([actor.positions, actor.distances], feed_dict=feed)
                # training_set.visualize_sampling(permutations)

                # Find best solution
                # print(circuit_length)
                j = np.argmin(circuit_length)
                best_permutation = permutations[j][:-1]
                predictions.append(circuit_length[j])

                ################################### UMPA LOOOOP HERE ###################################

                # print('\n Best tour length:',circuit_length[j])
                # print(' * permutation:', best_permutation)

                # plot corresponding tour
                # training_set.visualize_2D_trip(opt_trip)
                # training_set.visualize_2D_trip(or_sequence[best_permutation])

            predictions = np.asarray(predictions)
            targets = np.asarray(targets)

            print(' Mean length:', np.mean(predictions))
            ratio = np.asarray(predictions) / np.asarray(targets)
            print('\n Average deviation: \n', np.mean(ratio))

            n, bins, patches = plt.hist(ratio, 50, facecolor='r', alpha=0.75)

            plt.xlabel('Prediction/target')
            plt.ylabel('Counts')
            plt.title('Comparison to Google OR tools')
            plt.axis([0.9, 1.4, 0, 500])
            plt.grid(True)
            plt.show()


if __name__ == "__main__":
    main()
