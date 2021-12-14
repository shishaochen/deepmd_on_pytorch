import torch

from deepmd_pt.embedding_net import EmbeddingNet
from deepmd_pt.fitting import EnergyFittingNet
from deepmd_pt.stat import compute_output_stats, make_stat_input, merge_sys_stat


class EnergyModel(torch.nn.Module):

    def __init__(self, model_params, training_data):
        '''Based on components, construct a model for energy.

        Args:
        - model_params: The Dict-like configuration with model options.
        - training_data: The training dataset.
        '''
        super(EnergyModel, self).__init__()

        # Descriptor + Embedding Net
        descriptor_param = model_params.pop('descriptor')
        assert descriptor_param['type'] == 'se_e2_a', 'Only descriptor `se_e2_a` is supported!'
        self.embedding_net = EmbeddingNet(**descriptor_param)

        # Statistics
        data_stat_nbatch = model_params.get('data_stat_nbatch', 10)
        sampled = make_stat_input(training_data, data_stat_nbatch)
        merged = merge_sys_stat(sampled)
        coord = merged['coord']
        atype = merged['type']
        natoms = merged['natoms_vec']
        box = merged['box']
        self.embedding_net.compute_input_stats(coord, atype, natoms, box)

        # Fitting
        fitting_param = model_params.pop('fitting_net')
        assert fitting_param.pop('type', 'ener'), 'Only fitting net `ener` is supported!'
        fitting_param['ntypes'] = self.embedding_net.ntypes
        fitting_param['embedding_width'] = self.embedding_net.dim_out
        fitting_param['bias_atom_e'] = compute_output_stats(sampled['energy'], sampled['natoms_vec'])
        self.fitting_net = EnergyFittingNet(**fitting_param)

    def forward(self, coord, atype, natoms, box):
        '''Return total energy of the system.

        Args:
        - coord: Atom coordinates with shape [nframes, natoms[1]*3].
        - atype: Atom types with shape [nframes, natoms[1]].
        - natoms: Atom statisics with shape [self.ntypes+2].
        - box: Simulation box with shape [nframes, 9].

        Returns:
        - energy: Energy per atom.
        - force: XYZ force per atom.
        '''
        assert coord.requires_grad, 'Coordinate tensor must require gradient!'
        embedding = self.embedding_net(coord, atype, natoms, box)
        atom_energy = self.fitting_net(embedding, natoms)
        energy = atom_energy.sum(dim=-1)
        faked_grad = torch.ones_like(energy)
        force = torch.autograd.grad(energy, coord, grad_outputs=faked_grad, create_graph=True)[0]
        return energy, force
