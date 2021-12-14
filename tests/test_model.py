import collections
import numpy as np
import os
import torch
import unittest

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

from deepmd import op
from deepmd.common import data_requirement
from deepmd.descriptor import DescrptSeA
from deepmd.fit import EnerFitting
from deepmd.loss import EnerStdLoss
from deepmd.model import EnerModel
from deepmd.utils.data_system import DeepmdDataSystem
from deepmd.utils.learning_rate import LearningRateExp

from deepmd_pt.dataset import DeepmdDataSet
from deepmd_pt.learning_rate import LearningRateExp as MyLRExp
from deepmd_pt.loss import EnergyStdLoss
from deepmd_pt.model import EnergyModel


CUR_DIR = os.path.dirname(__file__)
kDataSystems = [
    os.path.join(CUR_DIR, 'water/data/data_0'),
    os.path.join(CUR_DIR, 'water/data/data_1'),
    os.path.join(CUR_DIR, 'water/data/data_2')
]

VariableState = collections.namedtuple('VariableState', ['value', 'gradient'])


def torch2tf(torch_name):
    fields = torch_name.split('.')
    element_id = int(fields[2])
    if fields[0] == 'embedding_net':
        layer_id = int(fields[4]) + 1
        weight_type = fields[5]
        return 'filter_type_all/%s_%d_%d:0' % (weight_type, layer_id, element_id)
    elif fields[3] == 'deep_layers':
        layer_id = int(fields[4])
        weight_type = fields[5]
        return 'layer_%d_type_%d/%s:0' % (layer_id, element_id, weight_type)
    elif fields[3] == 'final_layer':
        weight_type = fields[4]
        return 'final_layer_type_%d/%s:0' % (element_id, weight_type)
    else:
        raise RuntimeError('Unexpected parameter name: %s' % torch_name)


class DpTrainer(object):

    def __init__(self):
        self.batch_size = 3
        self.type_map = ['O', 'H']
        self.ntypes = len(self.type_map)
        self.rcut = 6.
        self.rcut_smth = 0.5
        self.sel = [46, 92]
        self.filter_neuron = [25, 50, 100]
        self.axis_neuron = 16
        self.n_neuron = [32, 32, 32]
        self.data_stat_nbatch = 3
        self.start_lr = 1.1
        self.stop_lr = 3.51e-8
        self.decay_steps = 500
        self.stop_steps = 1600
        self.start_pref_e = 0.02
        self.limit_pref_e = 1.
        self.start_pref_f = 1000.
        self.limit_pref_f = 1.

    def get_intermediate_state(self, num_steps=1):
        dp_model = self._get_dp_model()
        dp_loss = self._get_dp_loss()
        dp_lr = self._get_dp_lr()
        dp_ds = self._get_dp_dataset()
        dp_model.data_stat(dp_ds)

        # Build graph
        g = tf.Graph()
        with g.as_default():
            place_holders = self._get_dp_placeholders(dp_ds)
            model_pred = dp_model.build(
                coord_=place_holders['coord'],
                atype_=place_holders['type'],
                natoms=place_holders['natoms_vec'],
                box=place_holders['box'],
                mesh=place_holders['default_mesh'],
                input_dict=place_holders
            )
            global_step = tf.train.get_or_create_global_step()
            learning_rate = dp_lr.build(global_step, self.stop_steps)
            l2_l, _ = dp_loss.build(
                learning_rate=learning_rate,
                natoms=place_holders['natoms_vec'],
                model_dict=model_pred,
                label_dict=place_holders,
                suffix='test'
            )
            t_vars = tf.trainable_variables()
            optimizer = tf.train.AdamOptimizer(learning_rate)
            t_grad_and_vars = optimizer.compute_gradients(l2_l, t_vars)
            train_op = optimizer.apply_gradients(t_grad_and_vars, global_step)
            init_op = tf.global_variables_initializer()
            t_heads = {
                'loss': l2_l,
                'energy': model_pred['energy'],
                'force': model_pred['force']
            }

        # Get statistics of each component
        stat_dict = {
            'descriptor.mean': dp_model.descrpt.davg,
            'descriptor.stddev': dp_model.descrpt.dstd,
            'fitting_net.bias_atom_e': dp_model.fitting.bias_atom_e
        }

        # Get variables and their gradients
        with tf.Session(graph=g) as sess:
            sess.run(init_op)
            for _ in range(num_steps - 1):
                batch = dp_ds.get_batch()
                feeds = self._get_feed_dict(batch, place_holders)
                sess.run(train_op, feed_dict=feeds)

            batch = dp_ds.get_batch()
            feeds = self._get_feed_dict(batch, place_holders)
            grads_and_vars, head_dict = sess.run([t_grad_and_vars, t_heads], feed_dict=feeds)
            vs_dict = {}
            for idx, one in enumerate(t_vars):
                grad, var = grads_and_vars[idx]
                vs_dict[one.name] = VariableState(var, grad)

        # Used for reproducing
        return batch, head_dict, stat_dict, vs_dict

    def _get_dp_dataset(self):
        data = DeepmdDataSystem(
            systems=kDataSystems,
            batch_size=self.batch_size,
            test_size=1,
            rcut=self.rcut,
            type_map=self.type_map,
            trn_all_set=True
        )
        data.add_dict(data_requirement)
        return data

    def _get_dp_model(self):
        dp_descrpt = DescrptSeA(
            rcut=self.rcut,
            rcut_smth=self.rcut_smth,
            sel=self.sel,
            neuron=self.filter_neuron,
            axis_neuron=self.axis_neuron
        )
        dp_fitting = EnerFitting(
            descrpt=dp_descrpt,
            neuron=self.n_neuron
        )
        return EnerModel(
            descrpt=dp_descrpt,
            fitting=dp_fitting,
            type_map=self.type_map,
            data_stat_nbatch=self.data_stat_nbatch
        )

    def _get_dp_loss(self):
        return EnerStdLoss(
            starter_learning_rate=self.start_lr,
            start_pref_e=self.start_pref_e,
            limit_pref_e=self.limit_pref_e,
            start_pref_f=self.start_pref_f,
            limit_pref_f=self.limit_pref_f
        )

    def _get_dp_lr(self):
        return LearningRateExp(
            start_lr=self.start_lr,
            stop_lr=self.stop_lr,
            decay_steps=self.decay_steps
        )

    def _get_dp_placeholders(self, dataset):
        place_holders = {}
        data_dict = dataset.get_data_dict()
        for kk in data_dict.keys():
            if kk == 'type':
                continue
            prec = tf.float64
            place_holders[kk] = tf.placeholder(prec, [None], name = 't_' + kk)
            place_holders['find_'+kk] = tf.placeholder(tf.float32, name = 't_find_' + kk)
        place_holders['type'] = tf.placeholder(tf.int32, [None], name='t_type')
        place_holders['natoms_vec'] = tf.placeholder(tf.int32, [self.ntypes+2], name='t_natoms')
        place_holders['default_mesh'] = tf.placeholder(tf.int32, [None], name='t_mesh')
        place_holders['is_training'] = tf.placeholder(tf.bool)
        return place_holders

    def _get_feed_dict(self, batch, place_holders):
        feed_dict = {}
        for kk in batch.keys():
            if kk == 'find_type' or kk == 'type':
                continue
            if 'find_' in kk:
                feed_dict[place_holders[kk]] = batch[kk]
            else:
                feed_dict[place_holders[kk]] = np.reshape(batch[kk], [-1])
        for ii in ['type']:
            feed_dict[place_holders[ii]] = np.reshape(batch[ii], [-1])
        for ii in ['natoms_vec', 'default_mesh']:
            feed_dict[place_holders[ii]] = batch[ii]
        feed_dict[place_holders['is_training']] = True
        return feed_dict


class TestEnergy(unittest.TestCase):

    def setUp(self):
        self.dp_trainer = DpTrainer()
        self.wanted_step = 3
        for key in dir(self.dp_trainer):
            if not key.startswith('_') or key == 'get_intermediate_state':
                value = getattr(self.dp_trainer, key)
                setattr(self, key, value)

    def test_consistency(self):
        batch, head_dict, stat_dict, vs_dict = self.dp_trainer.get_intermediate_state(self.wanted_step+1)

        # Build DeePMD graph
        my_ds = DeepmdDataSet(kDataSystems, self.batch_size, self.type_map)
        my_model = EnergyModel(
            model_params={
                'descriptor': {
                    'type': 'se_e2_a',
                    'sel': self.sel,
                    'rcut_smth': self.rcut_smth,
                    'rcut': self.rcut,
                    'neuron': self.filter_neuron,
                    'axis_neuron': self.axis_neuron,
                },
                'fitting_net': {
                    'neuron': self.n_neuron
                },
                'data_stat_nbatch': self.data_stat_nbatch
            },
            training_data=my_ds
        )
        my_lr = MyLRExp(self.start_lr, self.stop_lr, self.decay_steps, self.stop_steps)
        my_loss = EnergyStdLoss(
            starter_learning_rate=self.start_lr,
            start_pref_e=self.start_pref_e,
            limit_pref_e=self.limit_pref_e,
            start_pref_f=self.start_pref_f,
            limit_pref_f=self.limit_pref_f
        )

        # Keep statistics consistency between 2 implentations
        my_em = my_model.embedding_net
        my_em.mean = stat_dict['descriptor.mean'].reshape([self.ntypes, my_em.nnei, 4])
        my_em.stddev = stat_dict['descriptor.stddev'].reshape([self.ntypes, my_em.nnei, 4])
        my_em.deriv_stddev = np.tile(np.expand_dims(my_em.stddev, axis=-1), [1, 1, 1, 3])
        my_model.fitting_net.bias_atom_e = stat_dict['fitting_net.bias_atom_e']

        # Keep parameter value consistency between 2 implentations
        for name, param in my_model.named_parameters():
            var_name = torch2tf(name)
            var = vs_dict[var_name].value
            with torch.no_grad():
                param.data.copy_(torch.from_numpy(var))

        # Start forward computing
        pt_coord = torch.from_numpy(batch['coord'])
        pt_coord.requires_grad_(True)
        atype = batch['type']
        natoms = batch['natoms_vec']
        box = batch['box']
        l_energy = torch.from_numpy(batch['energy'])
        l_force = torch.from_numpy(batch['force'])
        p_energy, p_force = my_model(pt_coord, atype, natoms, box)
        cur_lr = my_lr.value(self.wanted_step)
        loss = my_loss(cur_lr, natoms, p_energy, p_force, l_energy, l_force)[0]
        self.assertTrue(np.allclose(head_dict['energy'], p_energy.detach().numpy()))
        self.assertTrue(np.allclose(head_dict['force'], p_force.detach().numpy()))
        self.assertTrue(np.allclose(head_dict['loss'], loss.detach().numpy()))

        # Compare gradient for consistency
        loss.backward()
        for name, param in my_model.named_parameters():
            var_name = torch2tf(name)
            var_grad = vs_dict[var_name].gradient
            param_grad = param.grad.numpy()
            self.assertTrue(np.allclose(var_grad, param_grad, rtol=1e-4, atol=1e-6))


if __name__ == '__main__':
    unittest.main()
