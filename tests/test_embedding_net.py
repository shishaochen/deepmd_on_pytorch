import numpy as np
import os
import re
import torch
import unittest

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

from deepmd.descriptor import DescrptSeA

from deepmd_pt import my_random
from deepmd_pt.dataset import DeepmdDataSet
from deepmd_pt.embedding_net import EmbeddingNet
from deepmd_pt.env import GLOBAL_NP_FLOAT_PRECISION


CUR_DIR = os.path.dirname(__file__)


def gen_key(worb, depth, elemid):
    return (worb, depth, elemid)


def base_se_a(descriptor, coord, atype, natoms, box):
    g = tf.Graph()
    with g.as_default():
        name_pfx = 'd_sea_'
        t_coord = tf.placeholder(GLOBAL_NP_FLOAT_PRECISION, [None, None], name = name_pfx+'t_coord')
        t_atype = tf.placeholder(tf.int32, [None, None], name=name_pfx+'t_type')
        t_natoms = tf.placeholder(tf.int32, [descriptor.ntypes+2], name=name_pfx+'t_natoms')
        t_box = tf.placeholder(GLOBAL_NP_FLOAT_PRECISION, [None, None], name = name_pfx+'t_box')
        t_default_mesh = tf.placeholder(tf.int32, [None], name=name_pfx+'t_mesh')
        t_embedding = descriptor.build(t_coord, t_atype, t_natoms, t_box, t_default_mesh, input_dict={})
        fake_energy = tf.reduce_sum(t_embedding)
        t_force = descriptor.prod_force_virial(fake_energy, t_natoms)[0]
        t_vars = {}
        for var in tf.global_variables():
            ms = re.findall(r'([a-z]+)_(\d)_(\d)', var.name)
            if len(ms) == 1:
                m = ms[0]
                key = gen_key(worb=m[0], depth=int(m[1]), elemid=int(m[2]))
                t_vars[key] = var
        init_op = tf.global_variables_initializer()

    with tf.Session(graph=g) as sess:
        sess.run(init_op)
        embedding, force, values = sess.run([t_embedding, t_force, t_vars], feed_dict={
            t_coord: coord,
            t_atype: atype,
            t_natoms: natoms,
            t_box: box,
            t_default_mesh: np.array([0, 0, 0, 2, 2, 2])
        })
        return embedding, force, values


class TestSeA(unittest.TestCase):

    def setUp(self):
        my_random.seed(0)
        ds = DeepmdDataSet([
            os.path.join(CUR_DIR, 'water/data/data_0'),
            os.path.join(CUR_DIR, 'water/data/data_1'),
            os.path.join(CUR_DIR, 'water/data/data_2')
        ], 2, ['O', 'H'])
        self.batch = ds.get_batch()
        self.rcut = 6.
        self.rcut_smth = 0.5
        self.sel = [46, 92]
        self.filter_neuron = [25, 50, 100]
        self.axis_neuron = 16

    def test_consistency(self):
        dp_d = DescrptSeA(
            rcut=self.rcut,
            rcut_smth=self.rcut_smth,
            sel=self.sel,
            neuron=self.filter_neuron,
            axis_neuron=self.axis_neuron,
            seed=1
        )
        dp_embedding, dp_force, dp_vars = base_se_a(
            descriptor=dp_d,
            coord=self.batch['coord'],
            atype=self.batch['type'],
            natoms=self.batch['natoms_vec'],
            box=self.batch['box'],
        )

        # Reproduced
        embedding_net = EmbeddingNet(
            self.rcut, self.rcut_smth, self.sel,
            self.filter_neuron, self.axis_neuron
        )
        for name, param in embedding_net.named_parameters():
            ms = re.findall(r'(\d)\.deep_layers\.(\d)\.([a-z]+)', name)
            if len(ms) == 1:
                m = ms[0]
                key = gen_key(worb=m[2], depth=int(m[1])+1, elemid=int(m[0]))
                var = dp_vars[key]
                with torch.no_grad():
                    # Keep parameter value consistency between 2 implentations
                    param.data.copy_(torch.from_numpy(var))
        pt_coord = torch.from_numpy(self.batch['coord'])
        pt_coord.requires_grad_(True)
        env_embedding = embedding_net(
            pt_coord,
            self.batch['type'],
            self.batch['natoms_vec'],
            self.batch['box']
        )
        my_embedding = env_embedding.detach().numpy()
        fake_energy = torch.sum(env_embedding)
        fake_energy.backward()
        my_force = pt_coord.grad.numpy()

        # Check
        self.assertTrue(np.allclose(dp_embedding, my_embedding))
        self.assertTrue(np.allclose(dp_force, my_force))


if __name__ == '__main__':
    unittest.main()
