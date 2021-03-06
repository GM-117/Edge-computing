import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell, MultiRNNCell, DropoutWrapper
import copy
import numpy as np
from tqdm import tqdm

from Ptr_Net_TSPTW.dataset import DataGenerator
from Ptr_Net_TSPTW.decoder import Pointer_decoder
from Ptr_Net_TSPTW.critic import Critic
from Ptr_Net_TSPTW.config import get_config, print_config


# Tensor summaries for TensorBoard visualization
def variable_summaries(name, var, with_max_min=False):
    with tf.name_scope(name):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        if with_max_min:
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))


class Actor(object):

    def __init__(self, config):
        self.config = config

        # Data config
        self.batch_size = config.batch_size  # batch size
        self.max_length = config.task_num  # input sequence length (number of tasks)
        self.server_num = config.server_num
        self.server_allocate = config.server_allocate
        self.input_dimension = config.input_dimension  # dimension of a task (coordinates)

        # Network config
        self.input_embed = config.input_embed  # dimension of embedding space
        self.num_neurons = config.hidden_dim  # dimension of hidden states (LSTM cell)
        self.initializer = tf.contrib.layers.xavier_initializer()  # variables initializer

        # Reward config
        self.alpha = config.alpha
        self.beta = config.beta
        self.gama = config.gama

        self.alpha_c = config.alpha_c
        self.alpha_o = config.alpha_o
        self.alpha_b = config.alpha_b
        self.alpha_m = config.alpha_m

        # Training config (actor)
        self.global_step = tf.Variable(0, trainable=False, name="global_step")  # global step
        self.global_step_task = tf.Variable(0, trainable=False, name="global_step_task")  # global step
        self.global_step_time = tf.Variable(0, trainable=False, name="global_step_time")  # global step
        self.lr1_start = config.lr1_start  # initial learning rate
        self.lr1_decay_rate = config.lr1_decay_rate  # learning rate decay rate
        self.lr1_decay_step = config.lr1_decay_step  # learning rate decay step
        self.is_training = not config.inference_mode

        # Training config (critic)
        self.global_step2 = tf.Variable(0, trainable=False, name="global_step2")  # global step
        self.global_step2_task = tf.Variable(0, trainable=False, name="global_step2_task")  # global step
        self.global_step2_time = tf.Variable(0, trainable=False, name="global_step2_time")  # global step
        self.lr2_start = config.lr1_start  # initial learning rate
        self.lr2_decay_rate = config.lr1_decay_rate  # learning rate decay rate
        self.lr2_decay_step = config.lr1_decay_step  # learning rate decay step

        # Tensor block holding the input sequences [Batch Size, Sequence Length, Features]
        self.input_ = tf.placeholder(tf.float32, [self.batch_size, self.max_length, self.input_dimension],
                                     name="input_raw")
        self.server_input_ = tf.placeholder(tf.float32, [self.batch_size, self.server_num, 4],
                                            name="server_input_raw")

        self.weight_list = self.build_weight_list()
        self.batch_idx = 0

        self.build_permutation()
        self.build_critic()
        self.build_reward()
        self.build_optim()
        self.merged = tf.summary.merge_all()

    def build_weight_list(self):
        weight_list = []
        for idx in range(self.max_length):
            weight_list.append(1 - (idx / self.max_length))
        return tf.constant(weight_list, shape=[self.max_length, 1])

    def build_permutation(self):
        with tf.variable_scope("encoder"):
            with tf.variable_scope("embedding"):
                # Embed input sequence
                W_embed = tf.get_variable("weights", [1, self.input_dimension, self.input_embed],
                                          initializer=self.initializer)
                embedded_input = tf.nn.conv1d(self.input_, W_embed, 1, "VALID", name="embedded_input")
                # Batch Normalization
                embedded_input = tf.layers.batch_normalization(embedded_input, axis=2, training=self.is_training,
                                                               name='layer_norm', reuse=None)

            with tf.variable_scope("dynamic_rnn"):
                # Encode input sequence
                cell1 = LSTMCell(self.num_neurons,
                                 initializer=self.initializer)  # BNLSTMCell(self.num_neurons, self.training) or cell1 = DropoutWrapper(cell1, output_keep_prob=0.9)
                # Return the output activations [Batch size, Sequence Length, Num_neurons] and last hidden state as tensors.
                encoder_output, encoder_state = tf.nn.dynamic_rnn(cell1, embedded_input, dtype=tf.float32)

        with tf.variable_scope('decoder'):
            # Ptr-net returns permutations (self.positions), with their log-probability for backprop
            self.ptr = Pointer_decoder(encoder_output, self.config)
            self.positions, self.log_softmax, self.attending, self.pointing = self.ptr.loop_decode(encoder_state)
            variable_summaries('log_softmax', self.log_softmax, with_max_min=True)

    def build_critic(self):
        with tf.variable_scope("critic"):
            # Critic predicts reward (parametric baseline for REINFORCE)
            self.critic = Critic(self.config)
            self.critic.predict_rewards(self.input_)
            # ??????self.critic.predictions
            variable_summaries('predictions', self.critic.predictions, with_max_min=True)

    def cond_2(self, min_task_idx, server_run_map, server_remain):
        min_time = tf.cond(tf.equal(min_task_idx, tf.constant(-1)),
                           lambda: tf.constant(10, dtype=tf.float32), lambda: server_run_map[min_task_idx][-1])
        return tf.less_equal(min_time, tf.constant(0, dtype=tf.float32))

    def body_2(self, min_task_idx, server_run_map, server_remain):
        min_task = server_run_map[min_task_idx]
        min_need = min_task[:4]
        server_remain += min_need  # ??????????????????

        part1 = server_run_map[:min_task_idx]
        part2 = server_run_map[min_task_idx + 1:]
        server_run_map = tf.concat([part1, part2], axis=0)  # ????????????

        min_task_idx = tf.cond(tf.equal(tf.shape(server_run_map)[0], tf.constant(0)),
                               lambda: tf.constant(-1, dtype=tf.int64), lambda: tf.argmin(server_run_map, axis=0)[-1])
        min_task_idx = tf.cast(min_task_idx, dtype=tf.int32)  # ????????????????????????

        return min_task_idx, server_run_map, server_remain

    def cond(self, server_remain, need, time_used, server_run_map):
        c_ = tf.less(server_remain, need)
        c_ = tf.logical_or(tf.logical_or(c_[0], c_[1]), tf.logical_or(c_[2], c_[3]))
        return c_

    def body(self, server_remain, need, time_used, server_run_map):
        min_task_idx = tf.argmin(server_run_map, axis=0)[-1]
        min_task_idx = tf.cast(min_task_idx, dtype=tf.int32)  # ????????????????????????

        min_time = server_run_map[min_task_idx][-1]

        time_used += min_time  # ????????????

        server_run_map = tf.transpose(server_run_map, [1, 0])
        server_run_map = tf.unstack(server_run_map)
        server_run_map[-1] -= min_time
        server_run_map = tf.stack(server_run_map)
        server_run_map = tf.transpose(server_run_map, [1, 0])

        min_task_idx, \
        server_run_map, \
        server_remain = tf.while_loop(self.cond_2, self.body_2,
                                      [min_task_idx, server_run_map, server_remain],
                                      shape_invariants=[min_task_idx.get_shape(),
                                                        tf.TensorShape([None, self.input_dimension]),
                                                        server_remain.get_shape()])

        return server_remain, need, time_used, server_run_map

    def f1(self, timeout_count, time_used, server_remain, need, server_run_map, task):
        timeout_count += 1
        return timeout_count, time_used, server_remain, need, server_run_map, task

    def f2(self, timeout_count, time_used, server_remain, need, server_run_map, task):
        server_remain, \
        need, \
        time_used, \
        server_run_map = tf.while_loop(self.cond, self.body,
                                       [server_remain, need, time_used, server_run_map],
                                       shape_invariants=[server_remain.get_shape(),
                                                         need.get_shape(),
                                                         time_used.get_shape(),
                                                         tf.TensorShape([None, self.input_dimension])])
        server_run_map = tf.concat([server_run_map, task], axis=0)
        server_remain -= need  # ???????????????????????????
        return timeout_count, time_used, server_remain, need, server_run_map, task

    def build_reward(self):
        with tf.name_scope('permutations'):
            # Reorder input % tour
            # [256, 20, 2]
            self.permutations = tf.stack(
                [
                    tf.tile(tf.expand_dims(tf.range(self.batch_size, dtype=tf.int32), 1), [1, self.max_length]),
                    self.positions
                ],
                2
            )
            # ???input??????
            self.ordered_input_ = tf.gather_nd(self.input_, self.permutations)
            self.ordered_input_trans = tf.transpose(self.ordered_input_, [2, 1,
                                                                          0])  # [batch size, seq length , features] to [features, seq length, batch_size]
            # ????????? ??
            task_priority = self.ordered_input_trans[4]

        with tf.name_scope('environment'):
            time_used = [[tf.constant(0, dtype=tf.float32)] * self.server_num] * self.batch_size  # [batch_size, server_num]  # ????????????
            time_used_result = [None] * self.batch_size
            timeout_count = [tf.constant(0, dtype=tf.float32)] * self.batch_size  # [batch_size]  # ?????????
            server_input_ = tf.unstack(self.server_input_)
            for batch_idx, instance in enumerate(tf.unstack(self.ordered_input_)):
                instance = instance  # self.batch_idx:[max_length, input_dimension]
                server_run_map = [None] * self.server_num  # ????????????????????????????????????
                server_remain = server_input_[self.batch_idx]  # ?????????????????????
                server_remain = tf.unstack(server_remain)
                # ??????????????????????????????
                for task_idx, task in enumerate(tf.unstack(instance)):
                    server_idx = self.server_allocate[task_idx]
                    task = task  # [input_dimension]
                    need = task[:4]
                    time_out = task[5]
                    time_need = task[6]
                    task = tf.stack([task])

                    if server_run_map[server_idx] is None:
                        server_run_map[server_idx] = task
                        server_remain[server_idx] -= need
                        continue

                    timeout_count[batch_idx], time_used[batch_idx][server_idx], \
                    server_remain[server_idx], need, server_run_map[server_idx], task = tf.cond(
                        tf.less(time_out, time_used[batch_idx][server_idx] + time_need),
                        lambda: self.f1(timeout_count[batch_idx], time_used[batch_idx][server_idx],
                                        server_remain[server_idx], need, server_run_map[server_idx], task),
                        lambda: self.f2(timeout_count[batch_idx], time_used[batch_idx][server_idx],
                                        server_remain[server_idx], need, server_run_map[server_idx], task))

                # ????????????????????????
                server_time_used = tf.constant(0, dtype=tf.float32)
                for server_i in range(self.server_num):
                    max_time = tf.reduce_max(tf.stack(server_run_map[server_i]), axis=0)[-1]
                    server_time_used += time_used[batch_idx][server_i] + max_time
                time_used_result[batch_idx] = server_time_used / self.server_num

            self.time_use = 10 * tf.stack(time_used_result)/ self.max_length  # ??????
            self.ns_prob = 10 * tf.stack(timeout_count) / self.max_length  # ?????????

            priority_max = tf.reduce_max(task_priority, axis=0)  # ?????????????????????????????
            task_priority = tf.divide(task_priority, priority_max)  # ?????????
            task_priority_weight = tf.multiply(task_priority, self.weight_list)  # ??????
            self.task_priority_sum = tf.reduce_sum(task_priority_weight, axis=0)  # [batch_size]# ??????
            self.task_priority_sum = 2 * self.task_priority_sum / self.max_length

            self.reward_1 = tf.cast(self.time_use, tf.float32)  # ????????????
            self.reward_2 = tf.cast(self.task_priority_sum, tf.float32)  # ???????????????
            self.reward_3 = tf.cast(self.ns_prob, tf.float32)  # ?????????
            self.reward = self.reward_1 + self.reward_2 + self.reward_3
            variable_summaries('reward', self.reward, with_max_min=True)
            variable_summaries('reward_1', self.reward_1, with_max_min=True)
            variable_summaries('reward_2', self.reward_2, with_max_min=True)
            variable_summaries('reward_3', self.reward_3, with_max_min=True)

    def build_optim(self):
        # Update moving_mean and moving_variance for batch normalization layers
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            with tf.name_scope('reinforce'):
                # Actor learning rate
                self.lr1 = tf.train.exponential_decay(self.lr1_start, self.global_step, self.lr1_decay_step,
                                                      self.lr1_decay_rate, staircase=False, name="learning_rate1")
                # Optimizer
                self.opt1 = tf.train.AdamOptimizer(learning_rate=self.lr1, beta1=0.9, beta2=0.99, epsilon=0.0000001)
                # Discounted reward
                # ??????reward????????????reward?????????
                self.reward_baseline = tf.stop_gradient(self.reward - self.critic.predictions)  # [Batch size, 1]
                variable_summaries('reward_baseline', self.reward_baseline, with_max_min=True)
                # Loss
                # ?????????????????????
                self.loss1 = tf.reduce_mean(self.reward_baseline * self.log_softmax, 0)
                tf.summary.scalar('loss1', self.loss1)
                # Minimize step
                gvs = self.opt1.compute_gradients(self.loss1)
                capped_gvs = [(tf.clip_by_norm(grad, 1.), var) for grad, var in gvs if grad is not None]  # L2 clip
                self.train_step1 = self.opt1.apply_gradients(capped_gvs, global_step=self.global_step)

            with tf.name_scope('state_value'):
                # Critic learning rate
                self.lr2 = tf.train.exponential_decay(self.lr2_start, self.global_step2, self.lr2_decay_step,
                                                      self.lr2_decay_rate, staircase=False, name="learning_rate1")
                # Optimizer
                self.opt2 = tf.train.AdamOptimizer(learning_rate=self.lr2, beta1=0.9, beta2=0.99, epsilon=0.0000001)
                # Loss
                self.loss2 = tf.losses.mean_squared_error(self.reward, self.critic.predictions, weights=1.0)
                tf.summary.scalar('loss2', self.loss2)
                # Minimize step
                gvs2 = self.opt2.compute_gradients(self.loss2)
                capped_gvs2 = [(tf.clip_by_norm(grad, 1.), var) for grad, var in gvs2 if grad is not None]
                self.train_step2 = self.opt1.apply_gradients(capped_gvs2, global_step=self.global_step2)


if __name__ == "__main__":
    # get config
    config, _ = get_config()

    # Build Model and Reward from config
    actor = Actor(config)

    print("Starting training...")
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        print_config()

        solver = []  # Solver(actor.max_length)
        training_set = DataGenerator(solver)

        nb_epoch = 2
        for i in tqdm(range(nb_epoch)):  # epoch i

            # Get feed_dict
            input_batch = training_set.train_batch()
            feed = {actor.input_: input_batch}
            # print(' Input \n', input_batch)

            # permutation, distances, ordered_tw_open_, ordered_tw_close_, time_at_cities, constrained_delivery_time, delay, reward = sess.run([actor.positions, actor.distances, actor.ordered_tw_open_, actor.ordered_tw_close_, actor.time_at_cities, actor.constrained_delivery_time, actor.delay, actor.reward],feed_dict=feed)
            # print(' Permutation \n',permutation)
            # print(' Tour length \n',distances)
            # print(' Ordered tw open \n',ordered_tw_open_)
            # print(' Ordered tw close \n',ordered_tw_close_)
            # print(' Time at cities \n',time_at_cities)
            # print(' Constrained delivery \n',constrained_delivery_time)
            # print(' Delay \n',delay)
            # print(' Reward \n',reward)

        variables_names = [v.name for v in tf.global_variables() if 'Adam' not in v.name]
        values = sess.run(variables_names)
        for k, v in zip(variables_names, values):
            print("Variable: ", k, "Shape: ", v.shape)
