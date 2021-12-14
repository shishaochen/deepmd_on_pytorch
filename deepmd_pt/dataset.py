import glob
import logging
import numpy as np
import os

from typing import List

from deepmd_pt import env, my_random


def _shuffle_data(data):
    ret = {}
    nframes = data['coord'].shape[0]
    idx = np.arange (nframes)
    my_random.shuffle(idx)
    for kk in data :
        if type(data[kk]) == np.ndarray and \
            len(data[kk].shape) == 2 and \
            data[kk].shape[0] == nframes and \
            not ('find_' in kk) and \
            'type' != kk:
            ret[kk] = data[kk][idx]
        else :
            ret[kk] = data[kk]
    return ret, idx


def _get_subdata(data, idx=None):
    new_data = {}
    for ii in data:
        dd = data[ii]
        if 'find_' in ii:
            new_data[ii] = dd
        else:
            if idx is not None:
                new_data[ii] = dd[idx]
            else:
                new_data[ii] = dd
    return new_data


def _make_idx_map(atom_type):
    natoms = atom_type.shape[0]
    idx = np.arange(natoms)
    idx_map = np.lexsort((idx, atom_type))
    return idx_map


class DeepmdDataSystem(object):

    def __init__(self, sys_path: str, type_map: List[str] = None):
        '''Construct DeePMD-style frame collection of one system.

        Args:
        - sys_path: Paths to the system.
        - type_map: Atom types.
        '''
        self._atom_type = self._load_type(sys_path)
        self._natoms = len(self._atom_type)

        self._type_map = self._load_type_map(sys_path)
        if type_map is not None and self._type_map is not None:
            atom_type = [type_map.index(self._type_map[ii]) for ii in self._atom_type]
            self._atom_type = np.array(atom_type, dtype=np.int32)
            self._type_map = type_map
        self._idx_map = _make_idx_map(self._atom_type)

        self._data_dict = {}
        self.add('box', 9, must=True)
        self.add('coord', 3, atomic=True, must=True)
        self.add('energy', 1, atomic=False, must=False, high_prec=True)
        self.add('force',  3, atomic=True,  must=False, high_prec=False)

        self._sys_path = sys_path
        self._dirs = glob.glob(os.path.join(sys_path, 'set.*'))
        self._dirs.sort()

    def add(self, 
            key: str, 
            ndof: int, 
            atomic: bool = False, 
            must: bool = False, 
            high_prec: bool = False
    ):
        '''Add a data item that to be loaded.

        Args:
        - key: The key of the item. The corresponding data is stored in `sys_path/set.*/key.npy`
        - ndof: The number of dof
        - atomic: The item is an atomic property.
        - must: The data file `sys_path/set.*/key.npy` must exist. Otherwise, value is set to zero.
        - high_prec: Load the data and store in float64, otherwise in float32.
        '''
        self._data_dict[key] = {
            'ndof': ndof,
            'atomic': atomic,
            'must': must,
            'high_prec': high_prec
        }

    def get_batch(self, batch_size: int):
        '''Get a batch of data with at most `batch_size` frames. The frames are randomly picked from the data system.

        Args:
        - batch_size: Frame count.
        '''
        if not hasattr(self, '_frames'):
            set_size = 0
            self._set_count = 0
            self._iterator = 0
        else:
            set_size = self._frames['coord'].shape[0]
        if self._iterator + batch_size > set_size:
            set_idx = self._set_count % len(self._dirs)
            frames = self._load_set(self._dirs[set_idx])
            self._frames, _ = _shuffle_data(frames)
            set_size = self._frames['coord'].shape[0]
            self._iterator = 0
            self._set_count += 1
        iterator = min(self._iterator + batch_size, set_size)
        idx = np.arange(self._iterator, iterator)
        self._iterator += batch_size
        return _get_subdata(self._frames, idx)

    def get_ntypes(self):
        '''Number of atom types in the system.'''
        if self._type_map is not None:
            return len(self._type_map)
        else:
            return max(self._atom_type) + 1

    def get_natoms_vec(self, ntypes: int):
        '''Get number of atoms and number of atoms in different types.

        Args:
        - ntypes: Number of types (may be larger than the actual number of types in the system).
        '''
        natoms = len(self._atom_type)
        natoms_vec = np.zeros(ntypes).astype(int)
        for ii in range(ntypes) :
            natoms_vec[ii] = np.count_nonzero(self._atom_type == ii)
        tmp = [natoms, natoms]
        tmp = np.append(tmp, natoms_vec)
        return tmp.astype(np.int32)

    def _load_type(self, sys_path):
        return np.loadtxt(os.path.join(sys_path, 'type.raw'), dtype=np.int32, ndmin=1)

    def _load_type_map(self, sys_path):
        fname = os.path.join(sys_path, 'type_map.raw')
        if os.path.isfile(fname):
            with open(fname, 'r') as fin:
                content = fin.read()
            return content.split()                
        else:
            return None

    def _load_set(self, set_name):
        path = os.path.join(set_name, "coord.npy")
        if self._data_dict['coord']['high_prec'] :
            coord = np.load(path).astype(env.GLOBAL_ENER_FLOAT_PRECISION)
        else:
            coord = np.load(path).astype(env.GLOBAL_NP_FLOAT_PRECISION)
        if coord.ndim == 1:
            coord = coord.reshape([1, -1])
        assert(coord.shape[1] == self._data_dict['coord']['ndof'] * self._natoms)

        nframes = coord.shape[0]
        data = {'type': np.tile(self._atom_type[self._idx_map], (nframes, 1))}
        for kk in self._data_dict.keys():
            data['find_'+kk], data[kk] = self._load_data(
                set_name, 
                kk, 
                nframes, 
                self._data_dict[kk]['ndof'],
                atomic = self._data_dict[kk]['atomic'],
                high_prec = self._data_dict[kk]['high_prec'],
                must = self._data_dict[kk]['must']
            )
        return data

    def _load_data(self, set_name, key, nframes, ndof, atomic=False, must=True, high_prec=False):
        if atomic:
            ndof *= self._natoms
        path = os.path.join(set_name, key + '.npy')
        logging.info('Loading data from: %s', path)
        if os.path.isfile(path):
            if high_prec:
                data = np.load(path).astype(env.GLOBAL_ENER_FLOAT_PRECISION)
            else:
                data = np.load(path).astype(env.GLOBAL_NP_FLOAT_PRECISION)
            if atomic:
                data = data.reshape([nframes, self._natoms, -1])
                data = data[:,self._idx_map,:]
                data = data.reshape([nframes, -1])
            data = np.reshape(data, [nframes, ndof])
            return np.float32(1.0), data
        elif must:
            raise RuntimeError("%s not found!" % path)
        else:
            if high_prec:
                data = np.zeros([nframes,ndof]).astype(env.GLOBAL_ENER_FLOAT_PRECISION)                
            else :
                data = np.zeros([nframes,ndof]).astype(env.GLOBAL_NP_FLOAT_PRECISION)
            return np.float32(0.0), data


class DeepmdDataSet(object):

    def __init__(self, systems: List[str], batch_size: int, type_map: List[str]):
        '''Construct DeePMD-style dataset containing frames cross different systems.

        Args:
        - systems: Paths to systems.
        - batch_size: Max frame count in a batch.
        - type_map: Atom types.
        '''
        self._batch_size = batch_size
        self._type_map = type_map
        self._data_systems = [DeepmdDataSystem(ii, type_map=self._type_map) for ii in systems]
        self._ntypes = max([ii.get_ntypes() for ii in self._data_systems])
        self._natoms_vec = [ii.get_natoms_vec(self._ntypes) for ii in self._data_systems]

    @property
    def nsystems(self):
        return len(self._data_systems)

    def get_batch(self, sys_idx=None):
        '''Get a batch of frames from the selected system.'''
        if sys_idx is None:
            sys_idx = my_random.choice(np.arange(self.nsystems))
        b_data = self._data_systems[sys_idx].get_batch(self._batch_size)
        b_data['natoms_vec'] = self._natoms_vec[sys_idx]
        return b_data
