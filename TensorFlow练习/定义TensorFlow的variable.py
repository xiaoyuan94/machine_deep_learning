import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def setTfVariable():
    # w = tf.Variable([[1.0, 2.0]])
    # x = tf.Variable([[1.0], [2.0]])
    w = tf.Variable([[0.5, 1.0]])
    x = tf.Variable([[2.0], [1.0]])

    result = tf.matmul(w, x)

    init_op = tf.global_variables_initializer

    with tf.Session() as sess:
        sess.run(init_op)
        print(result.eval())


if __name__ == "__main__":
    setTfVariable()
