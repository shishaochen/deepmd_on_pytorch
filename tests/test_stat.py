import numpy as np
import os
import torch
import unittest

from deepmd.descriptor.se_a import DescrptSeA
from deepmd.fit.ener import EnerFitting
from deepmd.model.model_stat import make_stat_input as dp_make, merge_sys_stat as dp_merge
from deepmd.utils.data_system import DeepmdDataSystem
from deepmd.utils import random as dp_random

from deepmd_pt import my_random
from deepmd_pt.dataset import DeepmdDataSet
from deepmd_pt.embedding_net import EmbeddingNet
from deepmd_pt.stat import make_stat_input as my_make, merge_sys_stat as my_merge, compute_output_stats


CUR_DIR = os.path.dirname(__file__)


def compare(ut, base, given):
    if isinstance(base, list):
        ut.assertEqual(len(base), len(given))
        for idx in range(len(base)):
            compare(ut, base[idx], given[idx])
    elif isinstance(base, np.ndarray):
        ut.assertTrue(np.allclose(base, given))
    else:
        ut.assertEqual(base, given)


class TestDataset(unittest.TestCase):

    def setUp(self):
        self.systems = [
            os.path.join(CUR_DIR, 'water/data/data_0'),
            os.path.join(CUR_DIR, 'water/data/data_1'),
            os.path.join(CUR_DIR, 'water/data/data_2')
        ]
        self.batch_size = 3
        self.rcut = 6.
        self.data_stat_nbatch = 2
        self.rcut = 6.
        self.rcut_smth = 0.5
        self.sel = [46, 92]
        self.filter_neuron = [25, 50, 100]
        self.axis_neuron = 16
        self.n_neuron = [240, 240, 240]

        dp_random.seed(10)
        dp_dataset = DeepmdDataSystem(self.systems, self.batch_size, 1, self.rcut)
        dp_dataset.add('energy', 1, atomic=False, must=False, high_prec=True)
        dp_dataset.add('force',  3, atomic=True,  must=False, high_prec=False)
        self.dp_sampled = dp_make(dp_dataset, self.data_stat_nbatch, False)
        self.dp_merged = dp_merge(self.dp_sampled)
        self.dp_mesh = self.dp_merged.pop('default_mesh')
        self.dp_d = DescrptSeA(
            rcut=self.rcut,
            rcut_smth=self.rcut_smth,
            sel=self.sel,
            neuron=self.filter_neuron,
            axis_neuron=self.axis_neuron
        )

    def test_stat_input(self):
        my_random.seed(10)
        my_dataset = DeepmdDataSet(self.systems, self.batch_size, ['O', 'H'])
        my_sampled = my_make(my_dataset, self.data_stat_nbatch)
        my_merged = my_merge(my_sampled)
        dp_keys = set(self.dp_merged.keys())
        my_keys = set(my_merged.keys())
        self.assertEqual(len(dp_keys - my_keys), 0)
        self.assertEqual(len(my_keys - dp_keys), 0)
        for key in dp_keys:
            compare(self, self.dp_merged[key], my_merged[key])

    def test_stat_output(self):
        energy = self.dp_sampled['energy']
        natoms = self.dp_sampled['natoms_vec']
        dp_fn = EnerFitting(self.dp_d, self.n_neuron)
        dp_fn.compute_output_stats(self.dp_sampled)
        bias_atom_e = compute_output_stats(energy, natoms)
        self.assertTrue(np.allclose(dp_fn.bias_atom_e, bias_atom_e))

    def test_descriptor(self):
        coord = self.dp_merged['coord']
        atype = self.dp_merged['type']
        natoms = self.dp_merged['natoms_vec']
        box = self.dp_merged['box']
        self.dp_d.compute_input_stats(coord, box, atype, natoms, self.dp_mesh, {})
        my_en = EmbeddingNet(self.rcut, self.rcut_smth, self.sel, self.filter_neuron, self.axis_neuron)
        my_en.compute_input_stats(coord, atype, natoms, box)
        self.assertTrue(np.allclose(self.dp_d.davg.reshape([-1]), my_en.mean.reshape([-1])))
        self.assertTrue(np.allclose(self.dp_d.dstd.reshape([-1]), my_en.stddev.reshape([-1])))


if __name__ == '__main__':
    unittest.main()
