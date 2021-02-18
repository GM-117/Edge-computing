import tensorflow as tf

timeout = tf.constant([10, 20, 15, 12, 10, 40, 30, 20, 10, 15, 30, 20], shape=[4, 3])
time_use = tf.constant([7, 5, 2, 8, 4, 7, 10, 8, 6, 6, 3, 5], shape=[4, 3])

ns = 0
t = tf.constant([0 for i in range(3)], dtype=tf.int32, shape=[1, 3])
for to, tu in zip(tf.unstack(timeout), tf.unstack(time_use)):
    t = tf.add(t, tu)
    with tf.Session():
        print(t.eval())

# with tf.Session():
#
