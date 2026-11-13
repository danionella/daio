# Run: pytest tests/test_all.py
# All functions to be tested should start with test_ prefix

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

def _write_video_with_gaps(filename, **kwargs):
    """ Write a video with missing frames (irregular time stamps), as produced by cameras that drop frames. """
    n = 120
    pts = [i for i in range(n) if not (40 <= i < 45) and i % 17 != 0] # drop a block of frames and every 17th frame
    with VideoWriter(filename, fps=25, gop_size='12', **kwargs) as writer:
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
        sequential = list(VideoReader(filename))
        assert len(sequential) == len(pts)
        for index in ['auto', 'scan']:
            with VideoReader(filename, index=index) as reader:
                assert reader.index_source == (auto_source if index == 'auto' else 'scan')
                assert reader.nframes == len(pts)
                assert len(reader.frame_times) == len(pts)
                rng = np.random.default_rng(0)
                for i in list(rng.integers(0, len(pts), 30)) + [len(pts) - 1, 3, 4, 5, -1]:
                    assert np.array_equal(reader[int(i)], sequential[i]), f'{ext}, index={index}, frame {i}'

def test_video_cfr_uses_nominal_rate(tmp_path):
    filename = tmp_path / 'test_video_cfr.mp4'
    with VideoWriter(filename, fps=25) as writer:
        for i in range(50):
            writer.write(np.full((64, 96), i * 4, dtype='uint8'))
    sequential = list(VideoReader(filename))
    with VideoReader(filename) as reader:
        assert reader.index_source is None # constant frame rate mp4 (with B-frames): nominal frame rate, no file scan
        for i in [30, 10, 49, 11, 12, 0, 25]:
            assert np.array_equal(reader[i], sequential[i])

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
