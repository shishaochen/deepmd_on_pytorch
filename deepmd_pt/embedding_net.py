import numpy as np
import torch

from deepmd_pt import env
from deepmd_pt.descriptor import SmoothDescriptor


def analyze_descrpt(matrix, ndescrpt, natoms):
    '''Collect avg, square avg and count of descriptors in a batch.'''
    ntypes = natoms.size - 2
    start_index = 0
    sysr = []  # 每类元素的径向均值
    sysa = []  # 每类元素的轴向均值
    sysn = []  # 每类元素的样本数量
    sysr2 = []  # 每类元素的径向平方均值
    sysa2 = []  # 每类元素的轴向平方均值
    for type_i in range(ntypes):
        end_index = start_index + ndescrpt * natoms[2+type_i]
        dd = matrix[:, start_index:end_index]  # 本元素所有原子的 descriptor
        start_index = end_index
        dd = np.reshape (dd, [-1, 4])  # Shape is [nframes*natoms[2+type_id]*self.nnei, 4]
        ddr = dd[:,:1]  # 径向值
        dda = dd[:,1:]  # XYZ 轴分量
        sumr = np.sum(ddr)
        suma = np.sum(dda) / 3.
        sumn = dd.shape[0]  # Value is nframes*natoms[2+type_id]*self.nnei
        sumr2 = np.sum(np.multiply(ddr, ddr))
        suma2 = np.sum(np.multiply(dda, dda)) / 3.
        sysr.append(sumr)
        sysa.append(suma)
        sysn.append(sumn)
        sysr2.append(sumr2)
        sysa2.append(suma2)
    return sysr, sysr2, sysa, sysa2, sysn


def compute_std(sumv2, sumv, sumn):
    '''Compute standard deviation.'''
    if sumn == 0:
        return 1e-2
    val = np.sqrt(sumv2/sumn - np.multiply(sumv/sumn, sumv/sumn))
    if np.abs(val) < 1e-2:
        val = 1e-2
    return val


def Tensor(*shape):
    return torch.empty(shape, dtype=env.GLOBAL_PT_FLOAT_PRECISION)


class ResidualLinear(torch.nn.Module):

    def __init__(self, num_in, num_out, bavg=0., stddev=1., resnet_dt=False):
        '''Construct a linear layer.

        Args:
        - num_in: Width of input tensor.
        - num_out: Width of output tensor.
        - resnet_dt: Using time-step in the ResNet construction.
        '''
        super(ResidualLinear, self).__init__()
        self.num_in = num_in
        self.num_out = num_out
        self.resnet = resnet_dt

        self.matrix = torch.nn.Parameter(data=Tensor(num_in, num_out))
        torch.nn.init.normal_(self.matrix.data, std=stddev/np.sqrt(num_out+num_in))
        self.bias = torch.nn.Parameter(data=Tensor(1, num_out))
        torch.nn.init.normal_(self.bias.data, mean=bavg, std=stddev)
        if self.resnet:
            self.idt = torch.nn.Parameter(data=Tensor(1, num_out))
            torch.nn.init.normal_(self.idt.data, mean=1., std=0.001)

    def forward(self, inputs):
        '''Return X*W+b.'''
        xw_plus_b = torch.matmul(inputs, self.matrix) + self.bias
        hidden = torch.tanh(xw_plus_b)
        if self.resnet:
            hidden = hidden * self.idt
        if self.num_in == self.num_out:
            return inputs + hidden
        elif self.num_in * 2 == self.num_out:
            return torch.cat([inputs, inputs], dim=1) + hidden
        else:
            return hidden


class TypeFilter(torch.nn.Module):

    def __init__(self, offset, length, neuron):
        '''Construct a filter on the given element as neighbor.

        Args:
        - offset: Element offset in the descriptor matrix.
        - length: Atom count of this element.
        - neuron: Number of neurons in each hidden layers of the embedding net.
        '''
        super(TypeFilter, self).__init__()
        self.offset = offset
        self.length = length
        self.neuron = [1] + neuron

        deep_layers = []
        for ii in range(1, len(self.neuron)):
            one = ResidualLinear(self.neuron[ii-1], self.neuron[ii])
            deep_layers.append(one)
        self.deep_layers = torch.nn.ModuleList(deep_layers)

    def forward(self, inputs):
        '''Calculate decoded embedding for each atom.

        Args:
        - inputs: Descriptor matrix. Its shape is [nframes*natoms[0], len_descriptor].

        Returns:
        - `torch.Tensor`: Embedding contributed by me. Its shape is [nframes*natoms[0], 4, self.neuron[-1]].
        '''
        inputs_i = inputs[:, self.offset*4:(self.offset+self.length)*4]
        inputs_reshape = inputs_i.reshape(-1, 4)  # shape is [nframes*natoms[0]*self.length, 4]
        xyz_scatter = inputs_reshape[:, 0:1]
        for linear in self.deep_layers:
            xyz_scatter = linear(xyz_scatter)
        xyz_scatter = xyz_scatter.view(-1, self.length, self.neuron[-1])  # shape is [nframes*natoms[0], self.length, self.neuron[-1]]
        inputs_reshape = inputs_i.view(-1, self.length, 4).permute(0, 2, 1)  # shape is [nframes*natoms[0], 4, self.length]
        return torch.matmul(inputs_reshape, xyz_scatter)


class EmbeddingNet(torch.nn.Module):

    def __init__(self, rcut, rcut_smth, sel, neuron=[24, 48, 96], axis_neuron=8, **kwargs):
        '''Construct an embedding net of type `se_a`.

        Args:
        - rcut: Cut-off radius.
        - rcut_smth: Smooth hyper-parameter for pair force & energy.
        - sel: For each element type, how many atoms is selected as neighbors.
        - filter_neuron: Number of neurons in each hidden layers of the embedding net.
        - axis_neuron: Number of columns of the sub-matrix of the embedding matrix.
        '''
        super(EmbeddingNet, self).__init__()
        self.rcut = rcut
        self.rcut_smth = rcut_smth
        self.filter_neuron = neuron
        self.axis_neuron = axis_neuron

        self.ntypes = len(sel)  # 元素数量
        self.sec = np.cumsum(sel)  # 每种元素在邻居中的位移
        self.nnei = self.sec[-1]  # 总的邻居数量
        self.ndescrpt = self.nnei * 4  # 描述符的元素数量

        wanted_shape = (self.ntypes, self.nnei, 4)
        self.mean = np.zeros(wanted_shape, dtype=env.GLOBAL_NP_FLOAT_PRECISION)
        self.stddev = np.ones(wanted_shape, dtype=env.GLOBAL_NP_FLOAT_PRECISION)
        self.deriv_stddev = np.tile(np.expand_dims(self.stddev, axis=-1), [1, 1, 1, 3])

        filter_layers = []
        start_index = 0
        for type_i in range(self.ntypes):
            one = TypeFilter(start_index, sel[type_i], self.filter_neuron)
            filter_layers.append(one)
            start_index += sel[type_i]
        self.filter_layers = torch.nn.ModuleList(filter_layers)

    @property
    def dim_out(self):
        """
        Returns the output dimension of this descriptor
        """
        return self.filter_neuron[-1] * self.axis_neuron

    def compute_input_stats(self, coord, atype, natoms, box):
        '''Update mean and stddev for descriptor elements.

        Args:
        - coord: Batched atom coordinates with shape [nframes, natoms[1]*3].
        - atype: Batched atom types with shape [nframes, natoms[1]].
        - natoms: Batched atom statisics with shape [self.ntypes+2].
        - box: Batched simulation box with shape [nframes, 9].
        '''
        sumr = []
        suma = []
        sumn = []
        sumr2 = []
        suma2 = []
        for cc, tt, nn, bb in zip(coord, atype, natoms, box):  # 逐个 Batch 的分析
            descriptor = SmoothDescriptor.apply(
                torch.from_numpy(cc), tt, nn, bb,
                self.mean, self.stddev, self.deriv_stddev,
                self.rcut, self.rcut_smth, self.sec
            )
            sysr, sysr2, sysa, sysa2, sysn = analyze_descrpt(descriptor.numpy(), self.ndescrpt, nn)
            sumr.append(sysr)
            suma.append(sysa)
            sumn.append(sysn)
            sumr2.append(sysr2)
            suma2.append(sysa2)
        sumr = np.sum(sumr, axis=0)
        suma = np.sum(suma, axis=0)
        sumn = np.sum(sumn, axis=0)
        sumr2 = np.sum(sumr2, axis=0)
        suma2 = np.sum(suma2, axis=0)
        all_davg = []
        all_dstd = []
        for type_i in range(self.ntypes) :
            davgunit = [[sumr[type_i]/(sumn[type_i]+1e-15), 0, 0, 0]]
            dstdunit = [[
                compute_std(sumr2[type_i], sumr[type_i], sumn[type_i]), 
                compute_std(suma2[type_i], suma[type_i], sumn[type_i]), 
                compute_std(suma2[type_i], suma[type_i], sumn[type_i]), 
                compute_std(suma2[type_i], suma[type_i], sumn[type_i])
            ]]
            davg = np.tile(davgunit, [self.nnei, 1])
            dstd = np.tile(dstdunit, [self.nnei, 1])
            all_davg.append(davg)
            all_dstd.append(dstd)
        self.mean = np.stack(all_davg)
        self.stddev = np.stack(all_dstd)
        self.deriv_stddev = np.tile(np.expand_dims(self.stddev, axis=-1), [1, 1, 1, 3])

    def forward(self, coord, atype, natoms, box):
        '''Calculate decoded embedding for each atom.

        Args:
        - coord: Tell atom coordinates with shape [nframes, natoms[1]*3].
        - atype: Tell atom types with shape [nframes, natoms[1]].
        - natoms: Tell atom count and element count. Its shape is [2+self.ntypes].
        - box: Tell simulation box with shape [nframes, 9].

        Returns:
        - `torch.Tensor`: descriptor matrix with shape [nframes, natoms[0]*self.filter_neuron[-1]*self.axis_neuron].
        '''
        nall = natoms[0]
        dmatrix = SmoothDescriptor.apply(
            coord, atype, natoms, box,
            self.mean, self.stddev, self.deriv_stddev,
            self.rcut, self.rcut_smth, self.sec
        )  # shape is [nframes, nall*self.ndescrpt]
        dmatrix = dmatrix.view(-1, self.ndescrpt)  # shape is [nframes*nall, self.ndescrpt]
        xyz_scatter = None
        for ii, transform in enumerate(self.filter_layers):
            ret = transform(dmatrix)  # shape is [nframes*nall, 4, self.filter_neuron[-1]]
            if ii == 0:
                xyz_scatter = ret
            else:
                xyz_scatter = xyz_scatter + ret
        xyz_scatter /= self.nnei
        xyz_scatter_1 = xyz_scatter.permute(0, 2, 1)
        xyz_scatter_2 = xyz_scatter[:,:,0:self.axis_neuron]
        result = torch.matmul(xyz_scatter_1, xyz_scatter_2)  # shape is [nframes*nall, self.filter_neuron[-1], self.axis_neuron]
        return result.view(-1, nall*self.filter_neuron[-1]*self.axis_neuron)
