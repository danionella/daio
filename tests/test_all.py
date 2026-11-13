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

def test_video_without_timestamps_is_read_sequentially(tmp_path):
    """ Raw H.264 streams have no time stamps: no frame index is possible, but all frames are read correctly. """
    filename = tmp_path / 'raw.h264'
    with VideoWriter(filename, fps=25, g='12') as writer:
        writer.write_frames(_ramp_frames(60))
    _unreliable(filename)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        assert VideoReader(filename).nframes == 60

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
