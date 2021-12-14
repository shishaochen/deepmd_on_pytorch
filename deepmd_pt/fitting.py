import logging
import numpy as np
import torch

from deepmd_pt import env


def Tensor(*shape):
    return torch.empty(shape, dtype=env.GLOBAL_PT_FLOAT_PRECISION)


class SimpleLinear(torch.nn.Module):

    def __init__(self, num_in, num_out, bavg=0., stddev=1., use_timestep=False, activate=True):
        '''Construct a linear layer.

        Args:
        - num_in: Width of input tensor.
        - num_out: Width of output tensor.
        - use_timestep: Apply time-step to weight.
        - activate: Whether apply TANH to hidden layer.
        '''
        super(SimpleLinear, self).__init__()
        self.num_in = num_in
        self.num_out = num_out
        self.resnet = use_timestep
        self.activate = activate

        self.matrix = torch.nn.Parameter(data=Tensor(num_in, num_out))
        torch.nn.init.normal_(self.matrix.data, std=stddev/np.sqrt(num_out+num_in))
        self.bias = torch.nn.Parameter(data=Tensor(1, num_out))
        torch.nn.init.normal_(self.bias.data, mean=bavg, std=stddev)
        if self.resnet:
            self.idt = torch.nn.Parameter(data=Tensor(1, num_out))
            torch.nn.init.normal_(self.idt.data, mean=0.1, std=0.001)

    def forward(self, inputs):
        '''Return X*W+b.'''
        hidden = torch.matmul(inputs, self.matrix) + self.bias
        if self.activate:
            hidden = torch.tanh(hidden)
        if self.resnet:
            hidden = hidden * self.idt
        return hidden


class ResidualDeep(torch.nn.Module):

    def __init__(self, type_id, embedding_width, neuron, bias_atom_e, resnet_dt=False):
        '''Construct a filter on the given element as neighbor.

        Args:
        - typei: Element ID.
        - embedding_width: Embedding width per atom.
        - neuron: Number of neurons in each hidden layers of the embedding net.
        - bias_atom_e: Average enery per atom for each element.
        - resnet_dt: Using time-step in the ResNet construction.
        '''
        super(ResidualDeep, self).__init__()
        self.type_id = type_id
        self.neuron = [embedding_width] + neuron

        deep_layers = []
        for ii in range(1, len(self.neuron)):
            one = SimpleLinear(
                num_in=self.neuron[ii-1],
                num_out=self.neuron[ii],
                use_timestep=(resnet_dt and ii > 1 and self.neuron[ii-1] == self.neuron[ii])
            )
            deep_layers.append(one)
        self.deep_layers = torch.nn.ModuleList(deep_layers)
        self.final_layer = SimpleLinear(self.neuron[-1], 1, bavg=bias_atom_e, activate=False)

    def forward(self, inputs):
        '''Calculate decoded embedding for each atom.

        Args:
        - inputs: Embedding net output per atom. Its shape is [nframes*nloc, self.embedding_width].

        Returns:
        - `torch.Tensor`: Output layer with shape [nframes*nloc, self.neuron[-1]].
        '''
        outputs = inputs
        for idx, linear in enumerate(self.deep_layers):
            if idx > 0 and linear.num_in == linear.num_out:
                outputs = outputs + linear(outputs)
            else:
                outputs = linear(outputs)
        return self.final_layer(outputs)


class EnergyFittingNet(torch.nn.Module):

    def __init__(self, ntypes, embedding_width, neuron, bias_atom_e, resnet_dt=True, **kwargs):
        '''Construct a fitting net for energy.

        Args:
        - ntypes: Element count.
        - embedding_width: Embedding width per atom.
        - neuron: Number of neurons in each hidden layers of the fitting net.
        - bias_atom_e: Average enery per atom for each element.
        - resnet_dt: Using time-step in the ResNet construction.
        '''
        super(EnergyFittingNet, self).__init__()
        self.ntypes = ntypes
        self.embedding_width = embedding_width
        assert self.ntypes == len(bias_atom_e), 'Element count mismatches!'

        filter_layers = []
        for type_i in range(self.ntypes):
            one = ResidualDeep(type_i, embedding_width, neuron, bias_atom_e[type_i], resnet_dt)
            filter_layers.append(one)
        self.filter_layers = torch.nn.ModuleList(filter_layers)

        if 'seed' in kwargs:
            logging.info('Set seed to %d in fitting net.', kwargs['seed'])
            torch.manual_seed(kwargs['seed'])

    def forward(self, inputs, natoms):
        '''Based on embedding net output, alculate total energy.

        Args:
        - inputs: Embedding matrix. Its shape is [nframes, natoms[0]*self.embedding_width].
        - natoms: Tell atom count and element count. Its shape is [2+self.ntypes].

        Returns:
        - `torch.Tensor`: Total energy with shape [nframes, natoms[0]].
        '''
        start_index = 0
        outs = []
        for type_i in range(self.ntypes):
            offset = start_index*self.embedding_width
            length = natoms[2+type_i]*self.embedding_width
            inputs_i = inputs[:, offset:offset+length]
            inputs_i = inputs_i.reshape(-1, self.embedding_width)  # Shape is [nframes*natoms[2+type_i], self.embedding_width]
            final_layer = self.filter_layers[type_i](inputs_i)
            final_layer = final_layer.view(-1, natoms[2+type_i])  # Shape is [nframes, natoms[2+type_i]]
            outs.append(final_layer)
            start_index += natoms[2+type_i]
        outs = torch.cat(outs, dim=1)  # Shape is [nframes, natoms[0]]
        return outs.to(env.GLOBAL_PT_FLOAT_PRECISION)
