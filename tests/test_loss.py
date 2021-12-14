import numpy as np
import os
import torch
import unittest

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

from deepmd_pt.dataset import DeepmdDataSet
from deepmd.loss.ener import EnerStdLoss

from deepmd_pt.loss import EnergyStdLoss


CUR_DIR = os.path.dirname(__file__)


class TestLearningRate(unittest.TestCase):

    def setUp(self):
        self.start_lr = 1.1
        self.start_pref_e = 0.02
        self.limit_pref_e = 1.
        self.start_pref_f = 1000.
        self.limit_pref_f = 1.

        self.batch_size = 3
        self.dataset = DeepmdDataSet([
            os.path.join(CUR_DIR, 'water/data/data_0'),
            os.path.join(CUR_DIR, 'water/data/data_1'),
            os.path.join(CUR_DIR, 'water/data/data_2')
        ], self.batch_size, ['O', 'H'])

    def test_consistency(self):
        base = EnerStdLoss(self.start_lr, self.start_pref_e, self.limit_pref_e, self.start_pref_f, self.limit_pref_f)
        g = tf.Graph()
        with g.as_default():
            t_cur_lr = tf.placeholder(shape=[], dtype=tf.float64)
            t_natoms = tf.placeholder(shape=[None], dtype=tf.int32)
            t_penergy = tf.placeholder(shape=[None, 1], dtype=tf.float64)
            t_pforce = tf.placeholder(shape=[None, None], dtype=tf.float64)
            t_pvirial = tf.placeholder(shape=[None, 9], dtype=tf.float64)
            t_patom_energy = tf.placeholder(shape=[None, None], dtype=tf.float64)
            t_lenergy = tf.placeholder(shape=[None, 1], dtype=tf.float64)
            t_lforce = tf.placeholder(shape=[None, None], dtype=tf.float64)
            t_lvirial = tf.placeholder(shape=[None, 9], dtype=tf.float64)
            t_latom_energy = tf.placeholder(shape=[None, None], dtype=tf.float64)
            t_atom_pref = tf.placeholder(shape=[None, None], dtype=tf.float64)
            find_energy = tf.constant(1., dtype=tf.float64)
            find_force = tf.constant(1., dtype=tf.float64)
            find_virial = tf.constant(0., dtype=tf.float64)
            find_atom_energy = tf.constant(0., dtype=tf.float64)
            find_atom_pref = tf.constant(0., dtype=tf.float64)
            model_dict = {
                'energy': t_penergy,
                'force': t_pforce,
                'virial': t_pvirial,
                'atom_ener': t_patom_energy
            }
            label_dict = {
                'energy': t_lenergy,
                'force': t_lforce,
                'virial': t_lvirial,
                'atom_ener': t_latom_energy,
                'atom_pref': t_atom_pref,
                'find_energy': find_energy,
                'find_force': find_force,
                'find_virial': find_virial,
                'find_atom_ener': find_atom_energy,
                'find_atom_pref': find_atom_pref
            }
            t_loss = base.build(t_cur_lr, t_natoms, model_dict, label_dict, '')

        batch = self.dataset.get_batch()
        mine = EnergyStdLoss(self.start_lr, self.start_pref_e, self.limit_pref_e, self.start_pref_f, self.limit_pref_f)
        cur_lr = 1.2
        natoms = batch['natoms_vec']
        l_energy = batch['energy']
        l_force = batch['force']
        p_energy = np.ones_like(l_energy)
        p_force = np.ones_like(l_force)
        nloc = natoms[0]
        virial = np.zeros(shape=[self.batch_size, 9])
        atom_energy = np.zeros(shape=[self.batch_size, nloc])
        atom_pref = np.zeros(shape=[self.batch_size, nloc*3])

        with tf.Session(graph=g) as sess:
            base_loss, _ = sess.run(t_loss, feed_dict={
                t_cur_lr: cur_lr,
                t_natoms: natoms,
                t_penergy: p_energy,
                t_pforce: p_force,
                t_pvirial: virial,
                t_patom_energy: atom_energy,
                t_lenergy: l_energy,
                t_lforce: l_force,
                t_lvirial: virial,
                t_latom_energy: atom_energy,
                t_atom_pref: atom_pref
            })
        my_loss = mine(
            cur_lr,
            natoms,
            torch.from_numpy(p_energy),
            torch.from_numpy(p_force),
            torch.from_numpy(l_energy),
            torch.from_numpy(l_force)
        ).detach()
        self.assertTrue(np.allclose(base_loss, my_loss.numpy()))


if __name__ == '__main__':
    unittest.main()
