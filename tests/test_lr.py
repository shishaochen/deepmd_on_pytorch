import numpy as np
import unittest

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

from deepmd.utils import learning_rate

from deepmd_pt.learning_rate import LearningRateExp


class TestLearningRate(unittest.TestCase):

    def setUp(self):
        self.start_lr = 0.001
        self.stop_lr = 3.51e-8
        self.decay_steps = 500
        self.stop_steps = 1600

    def test_consistency(self):
        base_lr = learning_rate.LearningRateExp(self.start_lr, self.stop_lr, self.decay_steps)
        g = tf.Graph()
        with g.as_default():
            global_step = tf.placeholder(shape=[], dtype=tf.int32)
            t_lr = base_lr.build(global_step, self.stop_steps)

        my_lr = LearningRateExp(self.start_lr, self.stop_lr, self.decay_steps, self.stop_steps)
        with tf.Session(graph=g) as sess:
            for step_id in range(50):
                base_val = sess.run(t_lr, feed_dict={global_step: step_id})
                my_val = my_lr.value(step_id)
                self.assertTrue(np.allclose(base_val, my_val))


if __name__ == '__main__':
    unittest.main()
