# Run: pytest tests/test_all.py
# All functions to be tested should start with test_ prefix

import warnings

import av
import numpy as np
from daio.video import VideoReader, VideoWriter
from daio.h5 import lazyh5, save_to_h5, load_from_h5

def test_trivial():
    assert True == True

def test_trivial2():
    assert False == False

def test_video(tmp_path):
    filename = tmp_path / 'test_video.mp4'
    with VideoWriter(filename, fps=25) as writer:
        for i in range(20):
            frame = np.random.randint(0,255,size=(720,1280), dtype='uint8')
            writer.write(frame)
    
    with VideoReader(filename) as reader:
        assert reader.shape == (20, 720, 1280)
        assert reader.size == 20 * 720 * 1280
        for frame in reader:
            frame.mean()

def _decode_all(filename):
    """ Reference: all frames from a plain sequential decode with PyAV. """
    with av.open(str(filename)) as container:
        return [frame.to_ndarray(format='gray') for frame in container.decode(video=0)]

def _check_random_access(reader, reference, n_random=40, seed=0):
    rng = np.random.default_rng(seed)
    n = len(reference)
    indices = list(rng.integers(0, n, n_random)) + [n - 1, 3, 4, 5, 13, 12, 11, 0, -1, -2]
    for i in indices:
        assert np.array_equal(reader[int(i)], reference[i]), f'frame {i}'

def _write_video_with_gaps(filename, **kwargs):
    """ Write a video with missing frames (irregular time stamps), as produced by cameras that drop frames. """
    n = 120
    pts = [i for i in range(n) if not (40 <= i < 45) and i % 17 != 0] # drop a block of frames and every 17th frame
    with VideoWriter(filename, fps=25, g='12', **kwargs) as writer:
        for p in pts:
            frame = np.full((64, 96), (p * 2) % 256, dtype='uint8') # brightness encodes the original frame number
            writer.write(frame, pts=p)
    return pts

def test_video_irregular_timestamps(tmp_path):
    """ The i-th frame obtained by seeking must equal the i-th frame obtained by sequential reading. """
    cases = [ # (extension, writer options, expected index source in 'auto' mode)
        ('mp4', {}, 'scan'), # B-frames: container index holds decode time stamps only -> scan
        ('mp4', {'bf': '0'}, 'container'), # no B-frames: container index suffices
        ('mkv', {}, 'scan'), # no complete container index
    ]
    for ext, opts, auto_source in cases:
        filename = tmp_path / f'test_video_gaps_{len(opts)}.{ext}'
        pts = _write_video_with_gaps(filename, **opts)
        reference = _decode_all(filename)
        assert len(reference) == len(pts)
        for index in ['auto', 'scan']:
            with VideoReader(filename, index=index) as reader:
                assert reader.index_source == (auto_source if index == 'auto' else 'scan')
                assert reader.seekable
                assert reader.nframes == len(pts)
                assert np.allclose(reader.frame_times, np.array(pts) / 25)
                _check_random_access(reader, reference)
            assert all(np.array_equal(a, b) for a, b in zip(VideoReader(filename, index=index), reference))

def test_video_cfr_uses_nominal_rate(tmp_path):
    filename = tmp_path / 'test_video_cfr.mp4'
    with VideoWriter(filename, fps=25) as writer:
        for i in range(50):
            writer.write(np.full((64, 96), i * 4, dtype='uint8'))
    reference = _decode_all(filename)
    with VideoReader(filename) as reader:
        assert reader.index_source is None # constant frame rate mp4 (with B-frames): nominal frame rate, no file scan
        for i in [30, 10, 49, 11, 12, 0, 25, -1, -50]:
            assert np.array_equal(reader[i], reference[i])
        assert reader.index_source is None # negative indices must not trigger a file scan
        for i in [50, -51, 1000]:
            try:
                reader[i]
                assert False, f'no IndexError for {i}'
            except IndexError:
                pass
        assert np.array_equal(reader[7], reference[7]) and np.array_equal(reader[8], reference[8]) # state ok after errors

def _remux(src, dst, skip=0, pts_map=None):
    """ Copy packets into a new file, optionally dropping the first `skip` packets or changing time stamps. """
    with av.open(str(src)) as ic, av.open(str(dst), 'w') as oc:
        ist = ic.streams.video[0]
        ost = oc.add_stream_from_template(ist)
        for k, packet in enumerate(ic.demux(ist)):
            if packet.dts is None or k < skip:
                continue
            if pts_map is not None:
                packet.pts = packet.dts = pts_map(packet.pts)
            packet.stream = ost
            oc.mux(packet)

def _ramp_frames(n=120):
    return [np.full((64, 96), (i * 2) % 256, dtype='uint8') for i in range(n)]

def test_video_starting_mid_gop(tmp_path):
    """ Streams that do not start with a key frame (e.g. cut recordings): frames the decoder drops are not counted. """
    for bf in ['0', '2']:
        src, cut = tmp_path / f'src{bf}.mkv', tmp_path / f'cut{bf}.mkv'
        with VideoWriter(src, fps=25, g='12', bf=bf, **{'x264-params': 'repeat-headers=1'}) as writer:
            writer.write_frames(_ramp_frames())
        _remux(src, cut, skip=5)
        reference = _decode_all(cut)
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            with VideoReader(cut) as reader:
                assert reader.nframes == len(reference)
                _check_random_access(reader, reference)
                assert reader.seekable

def _unreliable(filename):
    """ Open a file that is expected to trigger the sequential fallback; check frames are still correct. """
    reference = _decode_all(filename)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        with VideoReader(filename) as reader:
            _check_random_access(reader, reference)
            assert not reader.seekable
    assert any('seeking is not reliable' in str(w.message) for w in caught)

def test_video_avi_bframes_falls_back_to_sequential(tmp_path):
    """ avi with H.264 B-frames: decoded time stamps are not in display order, so seeking cannot be trusted. """
    filename = tmp_path / 'bframes.avi'
    with VideoWriter(filename, fps=25, g='12') as writer:
        writer.write_frames(_ramp_frames())
    _unreliable(filename)

def test_video_duplicate_timestamps_fall_back_to_sequential(tmp_path):
    src, dup = tmp_path / 'src.mkv', tmp_path / 'dup.mkv'
    with VideoWriter(src, fps=25, g='12', bf='0') as writer:
        writer.write_frames(_ramp_frames())
    step = 40 # mkv time base is 1 ms; 25 fps
    _remux(src, dup, pts_map=lambda t: t - step if (t // step) % 10 == 5 else t)
    _unreliable(dup)

def test_video_intra_refresh_falls_back_to_sequential(tmp_path):
    """ Intra refresh streams have no real key frames; frames decoded right after a seek are incomplete. """
    filename = tmp_path / 'intra_refresh.mp4'
    pattern = np.random.default_rng(0).integers(0, 255, (64, 192), dtype='uint8')
    with VideoWriter(filename, fps=25, g='24', bf='0', **{'x264-params': 'intra-refresh=1:scenecut=0'}) as writer:
        writer.write_frames([pattern[:, i:i + 96] for i in range(96)]) # smoothly moving texture
    _unreliable(filename)

def test_video_truncated_file_warns(tmp_path):
    """ A truncated recording (e.g. crashed camera) contains a damaged last packet: warn about it. """
    filename, truncated = tmp_path / 'full.avi', tmp_path / 'truncated.avi'
    with VideoWriter(filename, codec='mjpeg', fps=25, pix_fmt='yuvj420p') as writer:
        writer.write_frames(_ramp_frames(60))
    data = filename.read_bytes()
    truncated.write_bytes(data[:int(len(data) * 0.8)])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        with VideoReader(truncated) as reader:
            reference = []
            with av.open(str(truncated)) as container: # reference: decodable frames only
                for packet in container.demux(video=0):
                    try:
                        reference += [f.to_ndarray(format='gray') for f in packet.decode()]
                    except av.InvalidDataError:
                        pass
            for i in [10, 3, len(reference) - 1, 20]:
                assert np.array_equal(reader[i], reference[i])
            assert sum(1 for _ in reader) == len(reference)
    assert any('damaged' in str(w.message) for w in caught)

def test_video_slicing_and_iteration(tmp_path):
    filename = tmp_path / 'slicing.mp4'
    with VideoWriter(filename, fps=25, g='12') as writer:
        writer.write_frames(_ramp_frames(40))
    ref = np.array(_decode_all(filename))
    with VideoReader(filename) as r:
        assert len(r) == 40
        for sl in [slice(None), slice(-3, None), slice(None, None, 5), slice(18, 60), slice(30, 10, -4), slice(5, 5)]:
            out = r[sl]
            assert out.shape == ref[sl].shape and np.array_equal(out, ref[sl]), sl
        assert np.array_equal(r[[3, 1, 30, -1]], ref[[3, 1, 30, -1]])
        assert np.array_equal(r[np.arange(4)], ref[:4])
        assert np.array_equal(r[0:1, :10], ref[0:1, :10]) # frame axis is kept for slices
        assert np.array_equal(r[2, :10, 5:8], ref[2, :10, 5:8])
        assert np.array_equal(r[5:9, 3], ref[5:9, 3])
        # each loop keeps its own position
        assert np.array_equal(np.array(list(r)), ref)
        pairs = list(zip(r, r))
        assert len(pairs) == 40 and all(np.array_equal(a, b) for a, b in pairs)
        count = 0
        for a in r[:3]:
            for b in r:
                count += 1
        assert count == 3 * 40
        outer = [np.array_equal(a, ref[i]) for i, a in enumerate(r) if [b for b in r[:2]]]
        assert all(outer) and len(outer) == 40

def test_video_iteration_propagates_errors(tmp_path):
    filename = tmp_path / 'err.mp4'
    with VideoWriter(filename, fps=25) as writer:
        writer.write_frames(_ramp_frames(5))
    try:
        for _ in VideoReader(filename, format='notaformat'):
            pass
        assert False, 'error was swallowed'
    except ValueError:
        pass

def test_video_scan_notice(tmp_path, capsys, monkeypatch):
    """ A one-line notice is printed when a large file is scanned to build a frame index. """
    import daio.video
    filename = tmp_path / 'test_video_gaps.mkv'
    _write_video_with_gaps(filename)
    VideoReader(filename, index='scan')
    assert capsys.readouterr().out == '' # small file: no notice
    monkeypatch.setattr(daio.video, 'SCAN_NOTICE_BYTES', 0)
    VideoReader(filename, index='scan')
    out = capsys.readouterr().out
    assert out.count('\n') == 1 and 'test_video_gaps.mkv' in out and 'GB' in out

def test_video_failed_init_no_spurious_del_error(tmp_path):
    """ If opening fails, the original exception is raised and __del__ does not raise a second error. """
    import gc, sys
    unraisable = []
    old_hook = sys.unraisablehook
    sys.unraisablehook = unraisable.append
    try:
        try:
            VideoReader(tmp_path / 'nonexistent.mp4')
        except FileNotFoundError:
            pass
        try:
            VideoWriter(tmp_path / 'video.notaformat')
        except ValueError:
            pass
        gc.collect()
    finally:
        sys.unraisablehook = old_hook
    assert unraisable == [], [str(u.exc_value) for u in unraisable]

def test_hdf5(tmp_path):
    data = {
        'a': 1,
        'b': 'hello',
        'c': np.random.rand(3),
        'd': np.random.rand(3,3),
        'e': {
            'f': 2,
            'g': 'world',
            'h': np.random.rand(3),
            'i': np.random.rand(3,3),
        }
    }
    filename = tmp_path / 'test.h5'
    save_to_h5(filename, data)
    data2 = load_from_h5(filename)

    assert data['e']['g'] == data2['e']['g']

def test_video_writer_reader(tmp_path):
    """Test writing and reading a video file."""
    filename = tmp_path / 'test_video_writer_reader.mp4'
    frames = [np.random.randint(0, 255, size=(480, 640), dtype='uint8') for _ in range(10)]

    # Write video
    with VideoWriter(filename, fps=30) as writer:
        for frame in frames:
            writer.write(frame)

    # Read video
    with VideoReader(filename) as reader:
        assert reader.nframes == len(frames)
        for i, frame in enumerate(reader):
            pass

def test_video_static_shape(tmp_path):
    """Test static shape retrieval for a video file."""
    filename = tmp_path / 'test_video_static_shape.mp4'
    frames = [np.random.randint(0, 255, size=(480, 640), dtype='uint8') for _ in range(5)]

    # Write video
    with VideoWriter(filename, fps=25) as writer:
        for frame in frames:
            writer.write(frame)

    # Check static shape
    shape = VideoReader.static_shape(filename)
    assert shape[0] == len(frames)
    assert np.all(shape[1:] == (480, 640))

def test_lazyh5_basic_operations(tmp_path):
    """Test basic operations with lazyh5."""
    filename = tmp_path / 'test_lazyh5.h5'
    data = {
        'int': 42,
        'float': 3.14,
        'string': 'hello',
        'array': np.random.rand(3, 3),
        'nested': {'key': 'value'}
    }

    # Save data
    h5 = lazyh5(filename, readonly=False, erase_existing=True)
    h5.from_dict(data)

    # Load data
    h5_loaded = lazyh5(filename)
    assert h5_loaded['int'] == data['int']
    assert h5_loaded['float'] == data['float']
    assert h5_loaded['string'] == data['string']
    assert np.array_equal(h5_loaded['array'], data['array'])
    assert h5_loaded['nested']['key'] == data['nested']['key']

def test_lazyh5_remove_key(tmp_path):
    """Test removing a key from lazyh5."""
    filename = tmp_path /  'test_lazyh5_remove.h5'
    data = {'key1': 'value1', 'key2': 'value2'}

    # Save data
    h5 = lazyh5(filename, readonly=False)
    h5.from_dict(data)

    # Remove a key
    h5.remove_key('key1')
    assert 'key1' not in h5.keys()
    assert h5['key2'] == 'value2'

def test_hdf5_serialization(tmp_path):
    """Test saving and loading nested dictionaries with serialization."""
    filename = tmp_path /  'test_hdf5_serialization.h5'
    data = {
        'a': np.random.rand(5),
        'b': {'nested': np.random.rand(3, 3)},
        'c': 'test_string'
    }

    # Save data
    save_to_h5(filename, data, serialize=True)

    # Load data
    loaded_data = load_from_h5(filename)
    assert np.array_equal(loaded_data['a'], data['a'])
    assert np.array_equal(loaded_data['b']['nested'], data['b']['nested'])
    assert loaded_data['c'] == data['c']


def test_hdf5_numpy_to_native(tmp_path):
    import h5py
    data = {'a': np.float32(1.5), 'b': {'c': np.int32(3)}, 'd': [np.float32(1), np.float32(2)], 'e': np.arange(3)}
    filename = tmp_path / 'test_native.h5'
    save_to_h5(filename, data, convert_numpy_to_native=True, verbosity=0)
    with h5py.File(filename, 'r') as f:
        assert 'json_type' not in f['a'].attrs   # stored as an HDF5 scalar, not serialized
        assert 'json_type' not in f['b/c'].attrs
    data2 = load_from_h5(filename)
    assert data2['a'] == 1.5 and data2['b']['c'] == 3
    assert np.array_equal(data2['d'], [1., 2.]) and np.array_equal(data2['e'], np.arange(3))


def test_lazyh5_subgroup_writes(tmp_path):
    """ Nested dicts and serialized items can be written into subgroups (h5path handling). """
    filename = tmp_path / 'subgroups.h5'
    h5 = lazyh5(filename)
    h5['c'] = {}
    h5['c']['e'] = [2, 3, 4]
    h5['c']['x'] = {'y': 1}
    h5['c']['n'] = None
    h5['c']['j'] = [1, None]
    h5['c']['x']['z'] = {'deep': 'value'}
    loaded = lazyh5(filename).to_dict()
    assert loaded['c']['x']['y'] == 1 and loaded['c']['x']['z']['deep'] == 'value'
    assert loaded['c']['n'] is None and loaded['c']['j'] == [1, None]
    assert np.array_equal(loaded['c']['e'], [2, 3, 4])
    sub = lazyh5(filename)['c'].to_dict()
    assert sub['x'] == {'y': 1, 'z': {'deep': 'value'}} and sub['n'] is None and sub['j'] == [1, None]


def test_h5path_without_trailing_slash(tmp_path):
    filename = tmp_path / 'h5path.h5'
    save_to_h5(filename, {'g': {'sub': {'x': 1}, 'j': None}}, verbosity=0)
    for path in ['/g', 'g', '/g/', 'g/']:
        assert load_from_h5(filename, h5path=path) == {'j': None, 'sub': {'x': 1}}
    save_to_h5(filename, {'new': {'z': 1}, 'n': None}, h5path='/g', file_mode='a', verbosity=0)
    save_to_h5(filename, {'z': 2}, h5path='/newgrp/sub', file_mode='a', verbosity=0) # group is created
    loaded = load_from_h5(filename)
    assert loaded['g']['new']['z'] == 1 and loaded['g']['n'] is None and loaded['newgrp']['sub']['z'] == 2


def test_lazyh5_getitem_decodes_serialized_items(tmp_path):
    """ lazyh5[key] returns the same values as load_from_h5 (not raw JSON bytes). """
    filename = tmp_path / 'serialized.h5'
    data = {'none': None, 'lst': [1, None], 'bools': [True, False], 'd': {'k': [{'a': 1}]}}
    save_to_h5(filename, data, verbosity=0)
    h5 = lazyh5(filename)
    assert h5['none'] is None and h5['lst'] == [1, None] and h5['bools'] == [True, False]
    assert h5['d']['k'] == [{'a': 1}] and h5.d.k == [{'a': 1}]


def test_save_to_h5_failure_does_not_destroy_file(tmp_path):
    """ Unsupported items raise before the file is opened, so an existing file is not truncated. """
    import pytest
    filename = tmp_path / 'precious.h5'
    save_to_h5(filename, {'precious': np.arange(10)}, verbosity=0)
    with pytest.raises(TypeError, match='/b/c'):
        save_to_h5(filename, {'a': 1, 'b': {'c': {1, 2}}}, verbosity=0) # a set is not JSON serializable
    assert np.array_equal(load_from_h5(filename)['precious'], np.arange(10))
    with pytest.raises(ValueError, match='already exists'): # append mode: existing keys are checked before writing
        save_to_h5(filename, {'new': 1, 'precious': 2}, file_mode='a', verbosity=0)
    assert list(load_from_h5(filename)) == ['precious']


def test_lazyh5_overwrite_keeps_old_data_on_failure(tmp_path, monkeypatch):
    import pytest
    import daio.h5
    filename = tmp_path / 'overwrite.h5'
    h5 = lazyh5(filename)
    h5['x'] = np.arange(5)
    with pytest.raises(TypeError):
        h5.from_dict({'x': {1, 2}}, overwrite=True) # cannot be encoded: nothing is changed
    assert np.array_equal(h5['x'], np.arange(5))
    original = daio.h5._write_node
    def failing_write_node(group, key, node):
        original(group, key, node)
        raise OSError('simulated write error')
    monkeypatch.setattr(daio.h5, '_write_node', failing_write_node)
    with pytest.raises(OSError):
        h5.from_dict({'x': np.zeros(3)}, overwrite=True) # fails while writing: old data is kept
    monkeypatch.undo()
    assert np.array_equal(h5['x'], np.arange(5)) and h5.keys() == ['x']
    h5.from_dict({'x': 'new'}, overwrite=True)
    assert h5['x'] == 'new' and h5.keys() == ['x']
    with pytest.raises(ValueError, match='already exists'):
        h5['x'] = 1
    with pytest.raises(AttributeError): # backward compatible exception type
        h5['x'] = 1


def test_save_to_h5_write_time_errors_do_not_destroy_file(tmp_path):
    """ Errors that h5py would only raise after opening the file are caught before it is opened. """
    import pytest
    filename = tmp_path / 'precious.h5'
    save_to_h5(filename, {'precious': np.arange(3)}, verbosity=0)
    for bad, kwargs in [({'a': 1, '': 2}, {}), ({'.': 1}, {}), ({'a': 1, 'a/b': 2}, {}), ({'a': np.arange(4)}, {'compression': 'foo'})]:
        with pytest.raises(ValueError):
            save_to_h5(filename, bad, verbosity=0, **kwargs)
        assert np.array_equal(load_from_h5(filename)['precious'], np.arange(3)), bad
    # str arrays containing NUL characters (not supported by HDF5 strings) are stored as JSON
    save_to_h5(filename, {'s': np.array(['a\x00b', 'c'])}, verbosity=0)
    assert list(load_from_h5(filename)['s']) == ['a\x00b', 'c']


def test_lazyh5_failed_new_key_is_removed(tmp_path, monkeypatch):
    import pytest
    import daio.h5
    h5 = lazyh5(tmp_path / 'partial.h5')
    original = daio.h5._write_node
    def failing_write_node(group, key, node):
        if key == 'bad':
            raise OSError('simulated write error')
        original(group, key, node)
    monkeypatch.setattr(daio.h5, '_write_node', failing_write_node)
    with pytest.raises(OSError):
        h5.from_dict({'y': {'ok': 1, 'bad': 2}})
    monkeypatch.undo()
    assert h5.keys() == [] # no partially written group 'y' left behind
    h5.from_dict({'y': {'ok': 1}}) # retrying works
    assert h5['y']['ok'] == 1


def test_lazyh5_overwrite_swap_never_loses_original(tmp_path, monkeypatch):
    import pytest
    import h5py
    h5 = lazyh5(tmp_path / 'swap.h5')
    h5['x'] = np.arange(5)
    original_move = h5py.Group.move
    def interrupted_move(self, source, dest):
        if source.startswith('__daio_tmp_'):
            raise KeyboardInterrupt
        return original_move(self, source, dest)
    monkeypatch.setattr(h5py.Group, 'move', interrupted_move)
    with pytest.raises(KeyboardInterrupt):
        h5.from_dict({'x': np.zeros(3)}, overwrite=True)
    monkeypatch.undo()
    assert np.array_equal(h5['x'], np.arange(5))
    assert h5.keys() == ['x'] # no temporary items left behind


def test_lazyh5_attribute_access_to_underscore_keys(tmp_path):
    h5 = lazyh5(tmp_path / 'underscore.h5')
    h5['_meta'] = 5
    assert h5._meta == 5 # attribute access works for keys starting with '_', as in earlier versions


def test_hdf5_review_followups(tmp_path, monkeypatch):
    import pytest
    import daio.h5
    filename = tmp_path / 'followups.h5'
    # integer lists that float64 cannot represent exactly stay exact; mixed int/float lists stay float arrays
    save_to_h5(filename, {'big': [-1, 2**63 + 1], 'mixed': [1, 2.5]}, verbosity=0)
    loaded = load_from_h5(filename)
    assert loaded['big'] == [-1, 2**63 + 1] and loaded['mixed'].dtype == np.float64
    # bytes lists with NUL or non-ASCII bytes are refused (they used to be silently truncated), before writing
    save_to_h5(filename, {'precious': np.arange(3)}, verbosity=0)
    with pytest.raises(TypeError):
        save_to_h5(filename, {'l': [b'ab\x00', b'\xff\x00']}, verbosity=0)
    assert np.array_equal(load_from_h5(filename)['precious'], np.arange(3))
    save_to_h5(filename, {'l': [b'ab', b'cd']}, verbosity=0) # plain ASCII bytes lists are stored (and load as str)
    assert list(load_from_h5(filename)['l']) == ['ab', 'cd']
    # a failed nested key ('a/b') also removes the group it created
    h5 = lazyh5(tmp_path / 'nested.h5')
    original = daio.h5._write_node
    def failing_write_node(group, key, node):
        if key == 'a/b':
            group.create_group('a') # (h5py creates the intermediate group before failing)
            raise OSError('simulated write error')
        original(group, key, node)
    monkeypatch.setattr(daio.h5, '_write_node', failing_write_node)
    with pytest.raises(OSError):
        h5.from_dict({'a/b': 1})
    monkeypatch.undo()
    assert h5.keys() == []
    # '~' is expanded consistently
    monkeypatch.setenv('HOME', str(tmp_path))
    monkeypatch.setenv('USERPROFILE', str(tmp_path)) # (used by expanduser on Windows)
    home = lazyh5('~/home.h5')
    home['a'] = 1
    assert home.keys() == ['a'] and home['a'] == 1 and (tmp_path / 'home.h5').exists()


def test_hdf5_dataframe(tmp_path):
    import pytest
    pd = pytest.importorskip('pandas')
    filename = tmp_path / 'df.h5'
    df = pd.DataFrame({'a': [1.5, 2.5], 'b': [3, 4]}, index=[10, 20])
    save_to_h5(filename, {'df': df, 'g': {'df': df}}, verbosity=0)
    for loaded in [load_from_h5(filename)['df'], lazyh5(filename)['g']['df']]:
        assert loaded.index.tolist() == [10, 20]
        assert np.array_equal(loaded['a'], df['a']) and np.array_equal(loaded['b'], df['b'])


def test_hdf5_types_roundtrip(tmp_path):
    """ Round trip of str, bytes, numpy scalars, complex, big ints and string/mixed lists and arrays. """
    import warnings
    filename = tmp_path / 'types.h5'
    data = {
        'bytes_ascii': b'abc', # stored as an HDF5 string (loads as str, as before)
        'bytes_binary': b'\xff\x00\x01',
        'str_nul': 'a\x00b',
        'np_float32': np.float32(1.5),
        'np_int32': np.int32(3),
        'np_bool': np.bool_(True),
        'complex': 1 + 2j,
        'bigint': 2**70,
        'U_array': np.array(['a', 'bc']),
        'str_list': [np.str_('a'), 'bc'],
        'mixed_list': [1, 'a'], # must not be converted to ['1', 'a']
        'arrays_list': [np.zeros(2), np.ones(3)],
        'zero_d': np.array(5),
    }
    with warnings.catch_warnings():
        warnings.simplefilter('error') # no (misleading) compression warnings
        save_to_h5(filename, data, compression='gzip', verbosity=0)
    loaded = load_from_h5(filename)
    assert loaded['bytes_ascii'] == 'abc'
    assert loaded['bytes_binary'] == b'\xff\x00\x01'
    assert loaded['str_nul'] == 'a\x00b'
    for key in ['np_float32', 'np_int32', 'np_bool']:
        assert loaded[key] == data[key] and loaded[key].dtype == data[key].dtype
    assert loaded['complex'] == 1 + 2j and loaded['bigint'] == 2**70
    assert list(loaded['U_array']) == ['a', 'bc'] and list(loaded['str_list']) == ['a', 'bc']
    assert loaded['mixed_list'] == [1, 'a']
    assert loaded['arrays_list'] == [[0, 0], [1, 1, 1]]
    assert loaded['zero_d'] == 5
    import pytest
    with pytest.raises(TypeError, match='keys must be strings'):
        save_to_h5(filename, {1: 'a'}, verbosity=0)


def test_hdf5_reads_binary_bytes_from_old_files(tmp_path):
    """ Non-ASCII bytes stored as HDF5 strings (by earlier versions) load as bytes instead of raising. """
    import h5py
    filename = tmp_path / 'old.h5'
    with h5py.File(filename, 'w') as f:
        f.create_dataset('b', data=b'\xff\x01')
    assert load_from_h5(filename)['b'] == b'\xff\x01'


def test_lazyh5_robustness(tmp_path):
    import copy, pickle, pytest
    filename = tmp_path / 'robust.h5'
    h5 = lazyh5(filename) # file does not exist yet
    assert len(h5) == 0 and h5.keys() == [] and 'items' in repr(h5)
    assert not hasattr(h5, 'missing')
    h5['a'] = 1
    assert copy.copy(h5)['a'] == 1 and pickle.loads(pickle.dumps(h5))['a'] == 1
    assert {'a', 'to_dict', 'from_dict', 'keys'} <= set(dir(h5))
    with pytest.raises(ValueError):
        lazyh5(filename, readonly=True, erase_existing=True)
    assert filename.exists()
    assert 'FileNotFoundError' in repr(lazyh5(tmp_path / 'missing.h5', readonly=True)) # repr does not raise
