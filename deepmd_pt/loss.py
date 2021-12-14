import torch

from deepmd_pt.env import GLOBAL_PT_FLOAT_PRECISION


class EnergyStdLoss(torch.nn.Module):

    def __init__(self, starter_learning_rate,
        start_pref_e=0.02, limit_pref_e=1., start_pref_f=1000., limit_pref_f=1., **kwargs):
        '''Construct a layer to compute loss on energy and force.'''
        super(EnergyStdLoss, self).__init__()
        self.starter_learning_rate = starter_learning_rate
        self.start_pref_e = start_pref_e
        self.limit_pref_e = limit_pref_e
        self.start_pref_f = start_pref_f
        self.limit_pref_f = limit_pref_f

    def forward(self, learning_rate, natoms, p_energy, p_force, l_energy, l_force):
        '''Return loss on loss and force.

        Args:
        - natoms: Tell atom count and element count. Its shape is [2+self.ntypes].
        - p_energy: Predicted energy of all atoms.
        - p_force: Predicted force per atom.
        - l_energy: Actual energy of all atoms.
        - l_force: Actual force per atom.

        Returns:
        - loss: Loss to minimize.
        '''
        coef = learning_rate / self.starter_learning_rate
        pref_e = self.limit_pref_e + (self.start_pref_e - self.limit_pref_e) * coef
        pref_f = self.limit_pref_f + (self.start_pref_f - self.limit_pref_f) * coef
        l2_ener_loss = torch.mean(torch.square(p_energy - l_energy))
        atom_norm_ener = torch.tensor(1./ natoms[0]).to(GLOBAL_PT_FLOAT_PRECISION)
        energy_loss = atom_norm_ener * (pref_e * l2_ener_loss)
        diff_f = l_force.view(-1) - p_force.view(-1)
        l2_force_loss = torch.mean(torch.square(diff_f))
        force_loss = (pref_f * l2_force_loss).to(GLOBAL_PT_FLOAT_PRECISION)
        rmse_e = l2_ener_loss.sqrt() * atom_norm_ener
        rmse_f = l2_force_loss.sqrt()
        return energy_loss + force_loss, rmse_e.detach(), rmse_f.detach()
