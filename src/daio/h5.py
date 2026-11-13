import warnings, os, io, posixpath, uuid
from os.path import expanduser, isfile, exists

import json

import numpy as np
import h5py
import hdf5plugin
#TODO: test defaulting to hdf5plugin.Zstd compression for json-serialized strings


class KeyExistsError(ValueError, AttributeError):
    """ Raised when writing a key that already exists in the HDF5 file (and overwriting is not enabled).

    Subclasses ValueError and, for backward compatibility with earlier versions of `lazyh5`, AttributeError. """


def save_to_h5(filename, data, serialize=True, compression=None, json_compression='gzip', verbosity=1, file_mode='w', convert_numpy_to_native=False, h5path='/'):
    """ Save a nested dictionary data structure to an HDF5 file. 

    All items are checked and encoded before the file is opened, so unsupported items raise an error without
    truncating or modifying the file.

    Args:
        filename (string): file name of the HDF5 file
        data (dict): Nested dictionary whose contents may be dict, ndarray, str, bytes, DataFrame and JSON-serializable objects
        serialize (boolean): enable JSON serialization
        compression (string): h5py compression type (e.g. 'gzip', 'lzf', 'zstd' or None)
        json_compression (string): h5py compression type for serialized JSON (default: 'gzip')
        file_mode (string): h5py.File access mode. 'w' (default) for create/detete and 'a' for create/append
        convert_numpy_to_native (boolean): convert numpy types to native python types
        h5path (string): path within the HDF5 file to save the data (created if it does not exist)

    based on https://github.com/danionella/lib2p/blob/master/lib2putils.py
    """
    _save(filename, data, serialize=serialize, compression=compression, json_compression=json_compression,
          verbosity=verbosity, file_mode=file_mode, convert_numpy_to_native=convert_numpy_to_native, h5path=h5path)


def _save(filename, data, serialize=True, compression=None, json_compression='gzip', verbosity=1, file_mode='w',
          convert_numpy_to_native=False, h5path='/', overwrite=False):
    opts = dict(serialize=serialize, verbosity=verbosity, compression=_compression_filter(compression),
                json_compression=_compression_filter(json_compression))
    for comp in (opts['compression'], opts['json_compression']):
        _check_compression(comp)
    if convert_numpy_to_native:
        data = _numpy_to_native(data)
    h5path = _normalize_h5path(h5path)
    # encode everything first: errors are raised before the file is opened (with mode 'w': truncated)
    prepared = _prepare_group(data, h5path, opts)
    with h5py.File(expanduser(filename), file_mode) as h5file:
        _write_items(h5file.require_group(h5path), prepared, overwrite=overwrite)


def load_from_h5(filename, h5path='/'):
    """ Load an HDF5 file to a dictionary
    
    Args:
        filename (string): file name of the HDF5 file
        h5path (string): path of the HDF5 group to load (default: '/')
        
    Returns:
        dict: file contents
    """
    filename = expanduser(filename)
    with h5py.File(filename, 'r') as h5file:
        return _read_group(h5file[_normalize_h5path(h5path)], filename)


_available_compression = set() # named filters that were found to work


def _check_compression(compression):
    """ Raise (before any file is opened) if a compression filter is not available. """
    if compression is None or (isinstance(compression, str) and compression in _available_compression):
        return
    with h5py.File(io.BytesIO(), 'w') as f:
        f.create_dataset('test', data=np.zeros(4), compression=compression)
    if isinstance(compression, str):
        _available_compression.add(compression)


def _normalize_h5path(h5path):
    """ Return an absolute HDF5 path without trailing (or repeated) slashes, e.g. 'a//b/' -> '/a/b'. """
    return '/' + '/'.join(part for part in str(h5path).split('/') if part)


def _compression_filter(compression):
    if isinstance(compression, str) and compression.lower() == 'zstd':
        return hdf5plugin.Zstd()
    return compression


def _h5_supports_dtype(dtype):
    try:
        h5py.h5t.py_create(dtype, logical=True)
        return True
    except TypeError:
        return False


def _all_str(arr):
    """ Whether all elements of an object array are str without NUL characters (which HDF5 strings cannot hold). """
    return all(isinstance(x, str) and '\x00' not in x for x in arr.flat)


def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f'Object of type {type(o).__name__} is not JSON serializable')


def _list_to_array(obj):
    """ Convert a list or tuple to an array that h5py can store, or return None (the list is then JSON serialized).
    Numeric lists become numeric arrays and lists of str become string arrays. Mixed lists (e.g. [1, 'a'], which
    numpy would convert to strings), bool lists, ragged lists and lists with None are left to JSON serialization. """
    try:
        arr = np.asarray(obj)
    except (ValueError, TypeError):
        return None
    if arr.dtype.kind in 'iu':
        return arr
    if arr.dtype.kind in 'fc':
        # integers that the float array cannot represent exactly (e.g. [-1, 2**63 + 1]): keep them exact as JSON
        if any(isinstance(x, int) and not isinstance(x, bool) and abs(x) > 2**53 for x in np.asarray(obj, dtype=object).flat):
            return None
        return arr
    if arr.dtype.kind == 'U': # (check the original elements: numpy converts [1, 'a'] to ['1', 'a'])
        return arr.astype(object) if _all_str(np.asarray(obj, dtype=object)) else None
    if arr.dtype.kind == 'S':
        # (numpy bytes arrays drop trailing NUL bytes, and HDF5 strings hold text: only plain ASCII bytes are stored)
        if all(isinstance(x, bytes) and b'\x00' not in x and x.isascii() for x in np.asarray(obj, dtype=object).flat):
            return arr
    return None


def _native(item):
    """ Return (data, dtype) for items stored as native HDF5 datasets, or None for items to be JSON serialized.
    Items that HDF5 cannot store raise TypeError (before the file is opened). """
    if isinstance(item, str):
        if '\x00' in item:
            raise TypeError('HDF5 strings cannot contain NUL characters')
        return item, None
    if isinstance(item, bytes):
        if b'\x00' in item or not item.isascii():
            raise TypeError('only ASCII bytes without NUL characters can be saved')
        return item, None
    if isinstance(item, int) and not -2**63 <= item < 2**64:
        return None # out of HDF5 integer range: JSON
    if isinstance(item, (int, float, np.int64, np.float64)):
        return item, None
    if isinstance(item, np.ndarray):
        if not _h5_supports_dtype(item.dtype):
            raise TypeError(f'arrays of dtype {item.dtype} cannot be saved')
        return item, None
    if isinstance(item, (list, tuple)):
        arr = _list_to_array(item)
        if arr is None:
            return None
        return arr, (h5py.string_dtype() if arr.dtype.kind == 'O' else None)
    return None


def _prepare_group(data, path, opts):
    """ Encode a (nested) dict into ('group', {key: node}) nodes, raising errors for items that cannot be saved. """
    if not isinstance(data, dict):
        raise TypeError(f'Expected a dict to save at {path}, got {type(data)}.')
    out = {}
    for key, item in data.items():
        if not isinstance(key, str):
            raise TypeError(f'Cannot save key {key!r} in {path}: keys must be strings.')
        parts = key.split('/')
        if not key or any(part in ('', '.', '..') for part in parts):
            raise ValueError(f'Cannot save key {key!r} in {path}: not a valid HDF5 name.')
        for k in range(1, len(parts)): # 'a/b' creates group 'a': it must not collide with another key 'a'
            if '/'.join(parts[:k]) in data:
                raise ValueError(f"Cannot save key {key!r} in {path}: collides with key {'/'.join(parts[:k])!r}.")
        out[key] = _prepare_item(item, posixpath.join(path, key), opts)
    return ('group', out)


def _prepare_item(item, path, opts):
    if opts['verbosity'] > 1:
        print('saving entry: {} -- {}'.format(path, type(item)))
    if isinstance(item, dict):
        return _prepare_group(item, path, opts)
    if type(item).__name__ == 'DataFrame':
        json_bytes = np.frombuffer(item.to_json().encode('utf-8'), dtype='byte')
        attrs = {'pandas_json_type': f'This {type(item)} was JSON serialized and UTF-8 encoded.'}
        return ('dataset', json_bytes, None, opts['json_compression'], attrs)
    try:
        native = _native(item)
    except TypeError as e:
        raise TypeError(f'Cannot save {type(item)} at {path}: {e}') from None
    if native is not None:
        data, dtype = native
        compression = None if np.ndim(data) == 0 else opts['compression']
        return ('dataset', data, dtype, compression, {})
    if not opts['serialize']:
        raise ValueError(f'Cannot save {type(item)} to {path}. Consider enabling serialisation.')
    if opts['verbosity'] > 0:
        print(f'serializing {type(item)} at {path}', flush=True)
    #TODO: test replacing json with orjson
    try:
        json_bytes = json.dumps(item, default=_json_default).encode('utf-8')
    except (TypeError, ValueError) as e:
        raise TypeError(f'Cannot save {type(item)} at {path}: {e}') from e
    attrs = {'json_type': f'This {type(item)} was JSON serialized and UTF-8 encoded.'}
    return ('dataset', np.frombuffer(json_bytes, dtype='byte'), None, opts['json_compression'], attrs)


def _write_node(group, key, node):
    if node[0] == 'group':
        subgroup = group.create_group(key)
        for k, n in node[1].items():
            _write_node(subgroup, k, n)
        return
    _, data, dtype, compression, attrs = node
    try:
        dataset = group.create_dataset(key, data=data, dtype=dtype, compression=compression)
    except TypeError:
        if compression is None:
            raise
        if key in group:
            del group[key]
        warnings.warn(f'{posixpath.join(group.name, key)}: compression is not supported for this item. Saving without compression.', stacklevel=4)
        dataset = group.create_dataset(key, data=data, dtype=dtype)
    for k, v in attrs.items():
        dataset.attrs[k] = v


def _write_items(group, prepared, overwrite=False):
    """ Write prepared items into group. Existing keys raise KeyExistsError (before anything is written), or are
    replaced if overwrite is True: the new item is written under a temporary name first, so that the old item is
    only removed once the new one was written successfully. A new item that fails to write is removed again. """
    items = prepared[1]
    existing = [key for key in items if key in group]
    if existing and not overwrite:
        raise KeyExistsError(f"Dataset or group {', '.join(repr(k) for k in existing)} already exists in '{group.name}'!")
    for key, node in items.items():
        if key not in group:
            root = key.split('/')[0] # (keys like 'a/b' create the group 'a')
            created_root = root not in group
            try:
                _write_node(group, key, node)
            except BaseException: # do not leave a partially written item behind
                if created_root and root in group:
                    del group[root]
                elif key in group:
                    del group[key]
                raise
            continue
        tmp, backup = f'__daio_tmp_{uuid.uuid4().hex}', f'__daio_old_{uuid.uuid4().hex}'
        try:
            _write_node(group, tmp, node)
        except BaseException:
            if tmp in group:
                del group[tmp]
            raise
        group.move(key, backup) # swap via a backup name, so that the original is never lost
        try:
            group.move(tmp, key)
        except BaseException:
            group.move(backup, key)
            if tmp in group:
                del group[tmp]
            raise
        del group[backup]


def _read_dataset(item):
    """ Read a dataset, decoding items that save_to_h5 serialized. """
    if 'pandas_json_type' in item.attrs:
        import pandas as pd
        return pd.read_json(io.StringIO(item[()].tobytes().decode('utf-8')))
    if 'json_type' in item.attrs:
        #TODO: test replacing json with orjson
        return json.loads(item[()].tobytes())
    if h5py.check_string_dtype(item.dtype) is not None:
        try:
            return item.asstr()[()]
        except UnicodeDecodeError: # not text (e.g. binary bytes saved by earlier versions): return raw bytes
            return item[()]
    return item[()]


def _read_item(item, filename):
    if 'pandas_type' in item.attrs:
        import pandas as pd
        return pd.read_hdf(filename, item.name)
    if isinstance(item, h5py.Dataset):
        return _read_dataset(item)
    if isinstance(item, h5py.Group):
        return _read_group(item, filename)
    raise ValueError(f"I don't know what to do about {item.name}.")


def _read_group(group, filename):
    return {key: _read_item(item, filename) for key, item in group.items()}
    

class lazyh5:
    """ A lazy-loading interface for HDF5 files. 
    
    This class provides an easy way to access HDF5 file content without fully loading it into 
    memory or keeping the file open. It supports dynamic access to datasets and subgroups.

    Args:
            filepath (str): Path to the HDF5 file.
            h5path (str, optional): HDF5 group path. Defaults to '/'.
            readonly (str, optional): Whether to open the file in read-only mode. Defaults to None (read-only iff file exists).
            erase_existing (bool, optional): Whether to erase existing file. Defaults to False.
    """

    def __init__(self, filepath, h5path='/', readonly=None, erase_existing=False):
        if erase_existing and readonly:
            raise ValueError('erase_existing=True cannot be combined with readonly=True.')
        if isinstance(filepath, (str, os.PathLike)):
            filepath = expanduser(filepath) # (like save_to_h5 and load_from_h5)
        self._filepath = filepath
        self._h5path = _normalize_h5path(h5path)
        if erase_existing and isfile(filepath):
            os.remove(filepath)
        if readonly is None:
            readonly = isfile(filepath)
        self._readonly = readonly

    def keys(self):
        """Lists the keys of the current HDF5 group.

        Returns:
            list: List of keys in the current HDF5 group (empty if a writable file has not been created yet).
        """
        if not self._readonly and not exists(self._filepath):
            return []
        with h5py.File(self._filepath, 'r') as f:
            out = list(f[self._h5path].keys())
        return out
    
    def to_dict(self):
        """Reads the HDF5 file or group into a dictionary.

        Returns:
            dict: A dictionary representation of the HDF5 file.
        """
        return load_from_h5(self._filepath, h5path=self._h5path)
    
    def from_dict(self, data, compression=None, json_compression='gzip', overwrite=False, verbosity=1):
        """Writes a dictionary to the HDF5 file or group.

        Args:
            data (dict): A dictionary to write to the HDF5 file (if readonly is False).
            compression (string): h5py compression type (e.g. 'gzip', 'lzf', 'zstd' or None)
            json_compression (string): h5py compression type for serialized JSON (default: 'gzip')
            overwrite (bool): replace existing keys. Otherwise, existing keys raise a KeyExistsError (a subclass of
                ValueError and AttributeError) and nothing is written. Existing items are only removed once their
                replacement has been written.
            verbosity (int): 0: silent, 1: print a note for serialized items, 2: print every item
        """
        if self._readonly:
            raise ValueError("Cannot add data to a read-only lazyh5 object.")
        _save(self._filepath, data, h5path=self._h5path, file_mode='a', compression=compression,
              json_compression=json_compression, overwrite=overwrite, verbosity=verbosity)
        return self

    def remove_key(self, key):
        """Removes the HDF5 dataset or group. Note this does not free up space in the file.
        
        Args:
            key (str): The key to remove.
        """
        if self._readonly:
            raise ValueError("Cannot remove data from a read-only lazyh5 object.")
        else:
            if exists(self._filepath):
                with h5py.File(self._filepath, 'a') as f:
                    if key in f[self._h5path]:
                        del f[self._h5path][key]
                    else:
                        raise KeyError(f"Key '{key}' not found in HDF5 file.")

    def __getitem__(self, key):
        """Gets an item by key. Groups are returned as lazyh5 objects, datasets are read (and decoded, like in
        load_from_h5)."""
        with h5py.File(self._filepath, 'r') as f:
            item = f[self._h5path][key]
            if isinstance(item, h5py.Group) and 'pandas_type' not in item.attrs:
                return lazyh5(self._filepath, h5path=posixpath.join(self._h5path, key), readonly=self._readonly)
            elif isinstance(item, (h5py.Group, h5py.Dataset)):
                return _read_item(item, self._filepath)
            else:
                raise KeyError(f"Unknown item type for key: {key}")
            
    def __setitem__(self, key, value):
        """Sets an item by key."""
        self.from_dict({key: value})

    def __getattr__(self, key):
        """Provides dynamic attribute-style access."""
        if key.startswith('__') or key in ('_filepath', '_h5path', '_readonly'):
            raise AttributeError(key) # special and internal attributes (e.g. looked up by copy, pickle or IPython)
        try:
            return self[key]
        except (KeyError, FileNotFoundError):
            raise AttributeError(key) from None

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
        try:
            content = f'with {len(self)} items'
        except (OSError, KeyError) as e:
            content = f'({type(e).__name__})'
        return f"<lazyh5 for file '{self._filepath}', HDF5 path '{self._h5path}' {content}>"

    def _ipython_key_completions_(self):
        """Enables key completions in IPython."""
        return self.keys()

    def __dir__(self):
        """Lists all accessible attributes and keys."""
        try:
            keys = self.keys()
        except (OSError, KeyError):
            keys = []
        return sorted(set(super().__dir__()) | set(keys))
    
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
        from IPython.display import JSON, display
        safe = json.loads(json.dumps(self.inspect_structure(), default=lambda o: o.item() if isinstance(o, np.generic) else str(o)))
        display(JSON(safe, root=self._filepath))


def _numpy_to_native(obj):
    ''' Recursively convert numpy scalars (np.generic) to native Python types; arrays are left as they are. '''
    if isinstance(obj, dict):
        return {k: _numpy_to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_numpy_to_native(v) for v in obj)
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def is_h5py_compatible_array(obj):
    """ Whether a list or tuple is saved as an HDF5 array (numeric lists and lists of str) rather than JSON. """
    return isinstance(obj, (list, tuple)) and _list_to_array(obj) is not None
