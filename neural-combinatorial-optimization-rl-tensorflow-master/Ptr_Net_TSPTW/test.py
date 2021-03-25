import tensorflow as tf

server_run = tf.constant([[1, 1, 1, 1],
                          [2, 2, 2, 2],
                          [3, 3, 3, 3],
                          [4, 4, 4, 4]])
min_time = tf.constant(1)

server_run = tf.transpose(server_run, [1, 0])
server_run = tf.unstack(server_run)
server_run[-1] -= min_time
server_run = tf.stack(server_run)
server_run = tf.transpose(server_run, [1, 0])

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    res = sess.run(server_run)
    print(res)
