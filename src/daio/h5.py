import json 
#TODO: test replacing json with orjson
import warnings
from os.path import expanduser

import numpy as np
import h5py
#import hdf5plugin
#TODO: test defaulting to hdf5plugin.Zstd compression for json-serialized strings


def save_to_h5(filename, data, serialize=True, compression=None, json_compression='gzip', verbosity=1, file_mode='w', convert_numpy_to_native=False):
    ''' Save a nested dictionary data structure to an HDF5 file. 

    Args:
        filename (string): file name of the HDF5 file
        data (dict): Nested dictionary whose contents may be dict, ndarray, str, bytes, DataFrame and JSON-serializable objects
        serialize (boolean): enable JSON serialization
        compression (string): h5py compression type (e.g. 'gzip', 'lzf' or None)
        json_compression (string): h5py compression type for serialized JSON (default: 'gzip')
        file_mode (string): h5py.File access mode. 'w' (default) for create/detete and 'a' for create/append

    based on https://github.com/danionella/lib2p/blob/master/lib2putils.py
    '''

    def recursively_save_contents_to_group(h5file, path, data_item):
        assert isinstance(data_item, (dict))
        for key, item in data_item.items():
            if verbosity > 1:
                print('saving entry: {} -- {}'.format(path + key, type(item)))
            if isinstance(item, (np.ndarray, np.int64, np.float64, str, bytes, int, float)):
                comp = None if np.isscalar(item) else compression
                try:
                    h5file[path].create_dataset(key, data=item, compression=comp)
                except TypeError:
                    warnings.warn(f'\n\nkey: {key} -- Saving data with compression failed. Saving without compression.\n')
                    h5file[path].create_dataset(key, data=item, compression=None)
            elif type(item).__name__ == 'DataFrame':
                json_bytes = np.frombuffer(item.to_json().encode('utf-8'), dtype='byte')
                h5file[path].create_dataset(key, data=json_bytes, compression=json_compression)
                h5file[path + key].attrs['pandas_json_type'] = f'This {type(item)} was JSON serialized and UTF-8 encoded.'
            elif isinstance(item, dict):
                h5file[path].create_group(key)
                recursively_save_contents_to_group(h5file, path + key + '/', item)
            elif serialize:
                if verbosity > 0:
                    print(f'serializing {type(item)} at {path+key}', flush=True)
                #TODO: test replacing json with orjson
                json_bytes = json.dumps(item).encode('utf-8')
                h5file[path].create_dataset(key, data=np.frombuffer(json_bytes, dtype='byte'), compression=json_compression)
                h5file[path + key].attrs['json_type'] = f'This {type(item)} was JSON serialized and UTF-8 encoded.'
            else:
                raise ValueError(f'Cannot save {type(item)} to {path+key}. Consider enabling serialisation.')

    if convert_numpy_to_native:
        data = convert_numpy_to_native(data)

    filename = expanduser(filename)
    with h5py.File(filename, file_mode) as h5file:
        recursively_save_contents_to_group(h5file, '/', data)


def load_from_h5(filename):
    ''' Load an HDF5 file to a dictionary
    
    Args:
        filename (string): file name of the HDF5 file
        
    Returns:
        dict: file contents
    '''

    def recursively_load_contents_from_group(h5file, path):
        ans = dict()
        for key, item in h5file[path].items():
            if 'pandas_type' in item.attrs.keys():
                import pandas as pd
                ans[key] = pd.read_hdf(filename, path + key)
            elif 'pandas_json_type' in item.attrs.keys():
                import pandas as pd
                json_str = item[()].tobytes().decode('utf-8')
                ans[key] = pd.read_json(json_str)
            elif 'json_type' in item.attrs.keys():
                #TODO: test replacing json with orjson
                ans[key] = json.loads(item[()].tobytes())
            elif isinstance(item, h5py._hl.dataset.Dataset):
                if h5py.check_string_dtype(item.dtype) is not None:
                    item = item.asstr()
                ans[key] = item[()]
            elif isinstance(item, h5py._hl.group.Group):
                ans[key] = recursively_load_contents_from_group(h5file, path + key + '/')
            else:
                raise ValueError(f"I don't know what to do about {path+key}.")
        return ans

    filename = expanduser(filename)
    with h5py.File(filename, 'r') as h5file:
        return recursively_load_contents_from_group(h5file, '/')
    

class lazyh5:
    """ A lazy-loading interface for HDF5 files. 
    
    This class provides an easy way to access HDF5 file content without fully
    loading it into memory. It supports dynamic access to datasets and subgroups.

    Args:
            filepath (str): Path to the HDF5 file.
            h5path (str, optional): HDF5 group path. Defaults to '/'.
            overwrite (bool, optional): Whether to overwrite existing items. Defaults to False.
    """

    def __init__(self, filepath, h5path='/', overwrite=False):
        self._filepath = filepath
        self._h5path = h5path
        self._overwrite = overwrite

    def keys(self):
        """Lists the keys of the current HDF5 group.

        Returns:
            list: List of keys in the current HDF5 group.
        """
        with h5py.File(self._filepath, 'r') as f:
            return list(f[self._h5path].keys())

    def __getitem__(self, key):
        """Gets an item by key."""
        with h5py.File(self._filepath, 'r') as f:
            item = f[self._h5path][key]
            if isinstance(item, h5py.Group):
                return lazyh5(self._filepath, h5path=f"{self._h5path}/{key}")
            elif isinstance(item, h5py.Dataset):
                return item[()]
            else:
                raise KeyError(f"Unknown item type for key: {key}")

    def __getattr__(self, key):
        """Provides dynamic attribute-style access."""
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setattr__(self, key, value):
        """Sets an attribute or creates a new dataset."""
        if key.startswith('_'):
            super().__setattr__(key, value)
            return

        with h5py.File(self._filepath, 'a') as f:
            full_path = f"{self._h5path}/{key}".lstrip('/')
            if (full_path in f) and (not self._overwrite):
                raise AttributeError(f"Dataset or group '{key}' already exists.")
            else:
                f[self._h5path].create_dataset(key, data=value)

    def __len__(self):
        """Gets the number of items in the current HDF5 group."""
        return len(self.keys())

    def __repr__(self):
        """Provides a string representation of the object."""
        return f"<lazyh5 for file '{self._filepath}', HDF5 path '{self._h5path}' with {len(self)} items>"

    def _ipython_key_completions_(self):
        """Enables key completions in IPython."""
        return self.keys()

    def __dir__(self):
        """Lists all accessible attributes and keys."""
        return self.keys() + dir(super())
