import numpy as np

from collections import defaultdict


def make_stat_input(dataset, nbatches):
    '''Pack data for statistics.

    Args:
    - dataset: The dataset to analyze.
    - nbatches: Batch count for collecting stats.

    Returns:
    - all_stat: A dictionary of list of list storing data for stat.
        data can be accessed by all_stat[key][sys_idx][batch_idx][frame_idx]
    '''
    all_stat = defaultdict(list)
    for ii in range(dataset.nsystems):
        sys_stat = defaultdict(list)
        for _ in range(nbatches):
            stat_data = dataset.get_batch(sys_idx=ii)
            for dd in stat_data:
                if dd == 'natoms_vec':
                    stat_data[dd] = stat_data[dd].astype(np.int32) 
                sys_stat[dd].append(stat_data[dd])
        for dd in sys_stat:
            all_stat[dd].append(sys_stat[dd])
    return all_stat


def merge_sys_stat(all_stat):
    '''Merge fields cross data systems. Data can be accessed by:
        all_stat[key][merged_batch_idx][frame_idx]
    '''
    first_key = list(all_stat.keys())[0]
    nsys = len(all_stat[first_key])
    ret = defaultdict(list)
    for ii in range(nsys):
        for dd in all_stat:
            for bb in all_stat[dd][ii]:
                ret[dd].append(bb)
    return ret


def compute_output_stats(energy, natoms, rcond=1e-3):
    '''Update mean and stddev for descriptor elements.

    Args:
    - energy: Batched energy with shape [nframes, 1].
    - natoms: Batched atom statisics with shape [self.ntypes+2].

    Returns:
    - energy_coef: Average enery per atom for each element.
    '''
    sys_ener = np.array([])  # 每个 System 下各 Frame 平均能量
    for ss in range(len(energy)):  # 逐个 System
        sys_data = []
        for ii in range(len(energy[ss])):  # 逐个 Batch
            for jj in range(len(energy[ss][ii])):  # 逐个 Frame
                sys_data.append(energy[ss][ii][jj])
        sys_data = np.concatenate(sys_data)
        sys_ener = np.append(sys_ener, np.average(sys_data))

    sys_tynatom = np.array([])
    nsys = len(natoms)
    for ss in range(len(natoms)):
        sys_tynatom = np.append(sys_tynatom, natoms[ss][0].astype(np.float64))
    sys_tynatom = np.reshape(sys_tynatom, [nsys,-1])
    sys_tynatom = sys_tynatom[:,2:]  # 每个系统，每个元素的原子数量
    energy_coef, _, _, _ = np.linalg.lstsq(sys_tynatom, sys_ener, rcond)
    return energy_coef
