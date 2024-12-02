import json, os
#TODO: test replacing json with orjson
import warnings
from os.path import expanduser, isfile

import numpy as np
import h5py
#import hdf5plugin
#TODO: test defaulting to hdf5plugin.Zstd compression for json-serialized strings


def save_to_h5(filename, data, serialize=True, compression=None, json_compression='gzip', verbosity=1, file_mode='w', convert_numpy_to_native=False, h5path='/'):
    ''' Save a nested dictionary data structure to an HDF5 file. 

    Args:
        filename (string): file name of the HDF5 file
        data (dict): Nested dictionary whose contents may be dict, ndarray, str, bytes, DataFrame and JSON-serializable objects
        serialize (boolean): enable JSON serialization
        compression (string): h5py compression type (e.g. 'gzip', 'lzf' or None)
        json_compression (string): h5py compression type for serialized JSON (default: 'gzip')
        file_mode (string): h5py.File access mode. 'w' (default) for create/detete and 'a' for create/append
        convert_numpy_to_native (boolean): convert numpy types to native python types
        h5path (string): path within the HDF5 file to save the data

    based on https://github.com/danionella/lib2p/blob/master/lib2putils.py
    '''

    def recursively_save_contents_to_group(h5file, path, data_item):
        assert isinstance(data_item, (dict))
        for key, item in data_item.items():
            if verbosity > 1:
                print('saving entry: {} -- {}'.format(path + key, type(item)))
            if isinstance(item, (np.ndarray, np.int64, np.float64, str, bytes, int, float)) or is_h5py_compatible_array(item):
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
        recursively_save_contents_to_group(h5file, h5path, data)


def load_from_h5(filename, h5path='/'):
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
        return recursively_load_contents_from_group(h5file, h5path)
    

class lazyh5:
    """ A lazy-loading interface for HDF5 files. 
    
    This class provides an easy way to access HDF5 file content without fully
    loading it into memory. It supports dynamic access to datasets and subgroups.

    Args:
            filepath (str): Path to the HDF5 file.
            h5path (str, optional): HDF5 group path. Defaults to '/'.
            readonly (str, optional): Whether to open the file in read-only mode. Defaults to None (read-only iff file exists).
    """

    def __init__(self, filepath, h5path='/', readonly=None):
        self._filepath = filepath
        self._h5path = h5path
        if readonly is None:
            readonly = isfile(filepath)
        self._readonly = readonly

    def keys(self):
        """Lists the keys of the current HDF5 group.

        Returns:
            list: List of keys in the current HDF5 group.
        """
        with h5py.File(self._filepath, 'r') as f:
            out = list(f[self._h5path].keys())
        return out
    
    def to_dict(self):
        """Reads the HDF5 file or group into a dictionary.

        Returns:
            dict: A dictionary representation of the HDF5 file.
        """
        return load_from_h5(self._filepath, h5path=self._h5path)
    
    def from_dict(self, data):
        """Writes a dictionary to the HDF5 file or group.

        Args:
            data (dict): A dictionary to write to the HDF5 file (if readonly is False).
        """
        if self._readonly:
            raise ValueError("Cannot add data to a read-only lazyh5 object.")
        if os.path.exists(self._filepath):
            with h5py.File(self._filepath, 'r') as f:
                for k, v in data.items():
                    if (self._h5path+'/'+k in f):
                        raise AttributeError(f"Dataset or group '{k}' already exists!")
        save_to_h5(self._filepath, data, h5path=self._h5path, file_mode='a')

    def remove_key(self, key):
        """Removes the HDF5 dataset or group. Note this does not free up space in the file.
        
        Args:
            key (str): The key to remove.
        """
        if self._readonly:
            raise ValueError("Cannot remove data from a read-only lazyh5 object.")
        else:
            if os.path.exists(self._filepath):
                with h5py.File(self._filepath, 'a') as f:
                    if key in f[self._h5path]:
                        del f[self._h5path][key]
                    else:
                        raise KeyError(f"Key '{key}' not found in HDF5 file.")

    def __getitem__(self, key):
        """Gets an item by key."""
        with h5py.File(self._filepath, 'r') as f:
            item = f[self._h5path][key]
            if isinstance(item, h5py.Group):
                return lazyh5(self._filepath, h5path=f"{self._h5path}/{key}".lstrip('/'), readonly=self._readonly)
            elif isinstance(item, h5py.Dataset):
                if h5py.check_string_dtype(item.dtype) is not None:
                    item = item.asstr()
                return item[()]
            else:
                raise KeyError(f"Unknown item type for key: {key}")
            
    def __setitem__(self, key, value):
        """Sets an item by key."""
        self.from_dict({key: value})

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
        
        self.from_dict({key: value})

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
    
    def inspect_structure(self, group=None, prefix=""):
        """Returns a dictionary with the hierarchical structure of the HDF5 file.

        Args:
            group (h5py.Group, optional): The HDF5 group to start from. Defaults to the root group.
            prefix (str, optional): The prefix for the keys representing the hierarchy. Defaults to "".

        Returns:
            dict: A dictionary where keys are paths to datasets or groups, and values are shape/dtype for datasets.
        """
        with h5py.File(self._filepath, 'r') as f:
            # Default to the group at the current h5path
            group = f[self._h5path] if group is None else group

            def _recursive_structure(g):
                structure = {}
                for key in g:
                    item = g[key]
                    if isinstance(item, h5py.Group):
                        structure[key] = _recursive_structure(item)
                    elif isinstance(item, h5py.Dataset):
                        if h5py.check_string_dtype(item.dtype) is not None:
                            if item.dtype.itemsize <= 50:
                                info = item.asstr()[()]
                            else:
                                info = f"<len {item.dtype.itemsize} str>"
                        else:
                            if item.size == 1:
                                info = item[()]
                            else:
                                info = f"<{item.shape} {item.dtype}>"
                        structure[key] = info
                return structure

            return _recursive_structure(group)
        
    def _ipython_display_(self):
        """Displays a summary of the object in IPython."""
        from IPython.display import JSON
        display(JSON(self.inspect_structure(), root='/'))


def is_h5py_compatible_array(obj):
    if isinstance(obj, (list, tuple)):
        try:
            # Ensure it can be converted to a numerical numpy array
            arr = np.asarray(obj)
            # Check if it is a numerical or string array (h5py supports these)
            return np.issubdtype(arr.dtype, np.number) or arr.dtype.char in {'S', 'U'}
        except (ValueError, TypeError):
            return False
    return False
