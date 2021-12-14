import numpy as np
import os
import torch
import unittest

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

from deepmd.env import op_module

from deepmd_pt import my_random
from deepmd_pt.dataset import DeepmdDataSet
from deepmd_pt.descriptor import SmoothDescriptor
from deepmd_pt.env import GLOBAL_NP_FLOAT_PRECISION


CUR_DIR = os.path.dirname(__file__)


def base_se_a(rcut, rcut_smth, sel, batch, mean, stddev):
    g = tf.Graph()
    with g.as_default():
        coord = tf.placeholder(GLOBAL_NP_FLOAT_PRECISION, [None, None])
        box = tf.placeholder(GLOBAL_NP_FLOAT_PRECISION, [None, None])
        atype = tf.placeholder(tf.int32, [None, None])
        natoms_vec = tf.placeholder(tf.int32, [None])
        default_mesh = tf.placeholder(tf.int32, [None])
        stat_descrpt, descrpt_deriv, rij, nlist \
            = op_module.prod_env_mat_a(coord,
                                        atype,
                                        natoms_vec,
                                        box,
                                        default_mesh,
                                        tf.constant(mean),
                                        tf.constant(stddev),
                                        rcut_a=-1.,
                                        rcut_r=rcut,
                                        rcut_r_smth=rcut_smth,
                                        sel_a=sel,
                                        sel_r=[0, 0])

        net_deriv_reshape = tf.ones_like(stat_descrpt)
        force = op_module.prod_force_se_a(net_deriv_reshape,
                                          descrpt_deriv,
                                          nlist,
                                          natoms_vec,
                                          n_a_sel=sum(sel),
                                          n_r_sel=0)

    with tf.Session(graph=g) as sess:
        return sess.run([stat_descrpt, force], feed_dict={
            coord: batch['coord'],
            box: batch['box'],
            natoms_vec: batch['natoms_vec'],
            atype: batch['type'],
            default_mesh: np.array([0, 0, 0, 2, 2, 2])
        })


class TestSeA(unittest.TestCase):

    def setUp(self):
        my_random.seed(10)
        ds = DeepmdDataSet([
            os.path.join(CUR_DIR, 'water/data/data_0'),
            os.path.join(CUR_DIR, 'water/data/data_1'),
            os.path.join(CUR_DIR, 'water/data/data_2')
        ], 2, ['O', 'H'])
        self.batch = ds.get_batch()
        self.rcut = 6.
        self.rcut_smth = 0.5
        self.sel = [46, 92]

        self.sec = np.cumsum(self.sel)
        self.ntypes = len(self.sel)
        self.nnei = sum(self.sel)

    def test_consistency(self):
        avg_zero = np.zeros([self.ntypes, self.nnei*4]).astype(GLOBAL_NP_FLOAT_PRECISION)
        std_ones = np.ones([self.ntypes, self.nnei*4]).astype(GLOBAL_NP_FLOAT_PRECISION)
        deriv_std_ones = np.ones([self.ntypes, self.nnei, 4, 3]).astype(GLOBAL_NP_FLOAT_PRECISION)

        base_d, base_force = base_se_a(
            rcut=self.rcut,
            rcut_smth=self.rcut_smth,
            sel=self.sel,
            batch=self.batch,
            mean=avg_zero,
            stddev=std_ones
        )
        pt_coord = torch.from_numpy(self.batch['coord'])
        pt_coord.requires_grad_(True)
        my_d = SmoothDescriptor.apply(
            pt_coord,
            self.batch['type'],
            self.batch['natoms_vec'],
            self.batch['box'],
            avg_zero.reshape([-1, self.nnei, 4]),
            std_ones.reshape([-1, self.nnei, 4]),
            deriv_std_ones,
            self.rcut,
            self.rcut_smth,
            self.sec
        )
        my_d.sum().backward()
        my_force = pt_coord.grad.detach().numpy()
        self.assertTrue(np.allclose(base_d, my_d.detach().numpy()))
        self.assertTrue(np.allclose(base_force, my_force))


if __name__ == '__main__':
    unittest.main()
