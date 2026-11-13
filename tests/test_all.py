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

def _write_leading_frames_without_timestamps(filename, n=60, n_negative=20):
    """ mkv whose first frames have negative time stamps, which the muxer stores without time stamps. """
    from fractions import Fraction
    with av.open(str(filename), 'w', options={'avoid_negative_ts': 'disabled'}) as container:
        stream = container.add_stream('libx264', rate=25, options={'g': '12'})
        stream.width, stream.height, stream.pix_fmt = 96, 64, 'yuv420p'
        for k in range(n):
            frame = av.VideoFrame.from_ndarray(np.full((64, 96), (k * 4) % 256, np.uint8), format='gray')
            frame.pts, frame.time_base = k - n_negative, Fraction(1, 25)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)

def test_video_streams_without_timestamps(tmp_path):
    """ Files with (some) frames without time stamps are read sequentially, from paths and file objects:
    raw H.264, an mkv whose leading frames have no time stamps (seeking to 0 skips them), a raw H.264 stream cut
    mid-GOP (packets before the first key frame are not decoded) and raw MJPEG (packets without file positions). """
    import io
    raw, leading, cut, mjpeg = (tmp_path / name for name in ['raw.h264', 'leading.mkv', 'cut.h264', 'raw.mjpeg'])
    with VideoWriter(raw, fps=25, g='12', **{'x264-params': 'repeat-headers=1'}) as writer:
        writer.write_frames(_ramp_frames(60))
    _write_leading_frames_without_timestamps(leading)
    with av.open(str(raw)) as container: # a raw stream is just its packets: drop the first ones (mid-GOP cut)
        cut.write_bytes(b''.join(bytes(p) for k, p in enumerate(container.demux(video=0)) if k >= 5 and p.size))
    with VideoWriter(mjpeg, codec='mjpeg', pix_fmt='yuvj420p', fps=25) as writer:
        writer.write_frames(_ramp_frames(30))
    for filename, seekable in [(raw, False), (leading, False), (cut, False), (mjpeg, True)]:
        reference = _decode_all(filename)
        for source in [filename, open(filename, 'rb'), io.BytesIO(filename.read_bytes())]:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                with VideoReader(source) as reader:
                    assert reader.nframes == len(reference), (filename.name, source)
                    _check_random_access(reader, reference, n_random=10)
                    assert sum(1 for _ in reader) == len(reference) and reader.seekable == seekable
            assert seekable or any('seeking is not reliable' in str(w.message) for w in caught)
            if hasattr(source, 'close'):
                source.close()

def test_video_frame_count_corrected_when_video_ends_early(tmp_path):
    """ Without a frame index, frames are counted from packets; if fewer frames decode, the count is corrected. """
    import pytest
    filename = tmp_path / 'raw.h264'
    with VideoWriter(filename, fps=25, g='12') as writer:
        writer.write_frames(_ramp_frames(60))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        reader = VideoReader(filename)
        reader._nframes_counted += 5 # simulate packets that do not decode (e.g. damaged)
        assert sum(1 for _ in reader) == 60 and reader.nframes == 60
        reader._nframes_counted += 5
        with pytest.raises(IndexError):
            reader[62] # random access past the real end also corrects the count
        assert reader.nframes == 60
    assert any('ended after 60 frames, expected 65' in str(w.message) for w in caught)

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

def _run_isolated(code):
    """ Run code in a fresh interpreter (importing daio from the same location as this test session). """
    import os, subprocess, sys, daio
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([os.path.dirname(os.path.dirname(daio.__file__)), os.environ.get('PYTHONPATH', '')]))
    return subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, env=env, timeout=120)

def test_videowriter_frame_sizes(tmp_path):
    import pytest
    # odd sizes with libx264/yuv420p raise a clear error, and closing does not crash the interpreter (subprocess)
    result = _run_isolated(f"""
import numpy as np
from daio.video import VideoWriter
for frame_shape in [None, (65, 97)]:
    try:
        with VideoWriter({str(tmp_path / 'odd.mp4')!r}, frame_shape=frame_shape) as writer:
            writer.write(np.zeros((65, 97), np.uint8))
    except ValueError as e:
        assert 'even' in str(e), e
    else:
        raise AssertionError('no error')
print('ok')
""")
    assert result.returncode == 0 and 'ok' in result.stdout, result.stderr[-2000:]
    # codecs / pixel formats that support odd sizes keep working
    for codec, pix_fmt, ext in [('libx264', 'yuv444p', 'mp4'), ('ffv1', 'yuv420p', 'mkv'), ('mpeg4', 'yuv420p', 'avi'), ('mjpeg', 'yuvj420p', 'avi')]:
        filename = tmp_path / f'odd_{codec}.{ext}'
        with VideoWriter(filename, codec=codec, pix_fmt=pix_fmt) as writer:
            writer.write_frames([np.full((65, 97), i * 50, np.uint8) for i in range(3)])
        assert VideoReader(filename).shape == (3, 65, 97), codec
    # frames of a different shape raise (they used to be rescaled silently)
    for frame_shape in [None, (64, 96)]:
        with VideoWriter(tmp_path / 'shape.mp4', frame_shape=frame_shape) as writer:
            writer.write(np.zeros((64, 96), np.uint8))
            for bad in [np.zeros((128, 192), np.uint8), np.zeros((32, 96, 3), np.uint8)]:
                with pytest.raises(ValueError, match='shape'):
                    writer.write(bad)

def _audio_info(filename):
    with av.open(str(filename)) as container:
        return {s.type: (s.codec_context.layout.name if s.type == 'audio' else s.frames, round(float(s.duration * s.time_base), 2))
                for s in container.streams}

def test_videowriter_audio(tmp_path):
    import pytest
    sr = 44100
    t = np.arange(sr) / sr
    stereo = (0.5 * np.stack([np.sin(2 * np.pi * 440 * t), np.sin(2 * np.pi * 660 * t)])).astype(np.float32)
    frame = np.zeros((64, 96), np.uint8)
    # audio only: mono/stereo, int16/float, 1-d/2-d
    for name, chunk in {'mono_int16_1d': (stereo[0] * 32767).astype(np.int16), 'mono_float_2d': stereo[:1],
                        'stereo_int16': (stereo * 32767).astype(np.int16), 'stereo_float64': stereo.astype(np.float64)}.items():
        with VideoWriter(tmp_path / f'{name}.mp4') as writer:
            for i in range(3):
                writer.write_audio(chunk, sr)
        assert _audio_info(tmp_path / f'{name}.mp4') == {'audio': ('mono' if 'mono' in name else 'stereo', 3.0)}, name
    # audio and video: declare the stream that is written second (or both); with frame_shape, video is stream 0
    for kwargs, audio_first in [({'frame_shape': (64, 96)}, True), ({'audio_rate': sr, 'audio_channels': 2}, False),
                                ({'frame_shape': (64, 96), 'audio_rate': sr, 'audio_channels': 2}, True)]:
        filename = tmp_path / f'av_{len(kwargs)}_{audio_first}.mp4'
        with VideoWriter(filename, **kwargs) as writer:
            for second in range(3):
                if audio_first:
                    writer.write_audio(stereo, sr)
                writer.write_frames([frame] * 25)
                if not audio_first:
                    writer.write_audio(stereo)
        assert _audio_info(filename) == {'video': (75, 3.0), 'audio': ('stereo', 3.0)}, kwargs
        if 'frame_shape' in kwargs:
            with av.open(str(filename)) as container:
                assert container.streams[0].type == 'video'
    # undeclared second stream: clear errors (streams cannot be added once data was written)
    with VideoWriter(tmp_path / 'audio_first.mp4') as writer:
        writer.write_audio(stereo, sr)
        with pytest.raises(ValueError, match='frame_shape'):
            writer.write(frame)
    with VideoWriter(tmp_path / 'video_first.mp4') as writer:
        writer.write_frames([frame] * 50)
        with pytest.raises(ValueError, match='audio_rate'):
            writer.write_audio(stereo, sr)
    assert VideoReader(tmp_path / 'video_first.mp4').nframes == 50 # the video is still written
    # the audio stream uses the given sample rate; channel mismatches raise
    with VideoWriter(tmp_path / 'rate48k.mp4', audio_rate=48000) as writer:
        writer.write_audio(np.zeros(48000, np.float32))
    with av.open(str(tmp_path / 'rate48k.mp4')) as container:
        assert container.streams.audio[0].rate == 48000
    with pytest.raises(ValueError, match='channels'):
        with VideoWriter(tmp_path / 'channels.mp4', audio_rate=sr) as writer:
            writer.write_audio(stereo)

def test_videowriter_validation(tmp_path):
    import pytest
    from fractions import Fraction
    frame = np.zeros((64, 96), np.uint8)
    with VideoWriter(tmp_path / 'pts.mp4') as writer:
        writer.write_frames([np.full((64, 96), i * 8, np.uint8) for i in range(30)])
        for pts, match in [(5, 'pts must increase'), (31.5, 'integer')]: # decreasing pts used to drop frames silently
            with pytest.raises(ValueError, match=match):
                writer.write(frame, pts=pts)
    assert VideoReader(tmp_path / 'pts.mp4').nframes == 30
    with VideoWriter(tmp_path / 'pts_mixed.mp4') as writer: # explicit pts followed by automatic pts
        for pts in [10, None, None]:
            writer.write(frame, pts=pts)
    with VideoReader(tmp_path / 'pts_mixed.mp4', index='scan') as reader:
        assert reader.nframes == 3 and np.allclose(np.diff(reader.frame_times), 1 / 25)
    # frame rates: floats become fractions (low rates are not rounded to 0); invalid rates raise
    for fps, expected in [(29.97, Fraction(2997, 100)), (30000 / 1001, Fraction(30000, 1001)), (1e-4, Fraction(1, 10000))]:
        with VideoWriter(tmp_path / 'fps.mkv', fps=fps) as writer:
            writer.write(frame)
        assert writer.fps == expected, fps
    for fps in [0, -1, 0.0, float('nan')]:
        with pytest.raises(ValueError):
            VideoWriter(tmp_path / 'badfps.mp4', fps=fps)
    # codec options: non-string values are converted, None values ignored
    with VideoWriter(tmp_path / 'options.mp4', crf=18, tune=None) as writer:
        writer.write(frame)
    assert 'libx264' in VideoWriter.codecs_available # also works on the class
    with pytest.raises(ValueError, match='closed'):
        writer.write(frame)

def test_videowriter_close_and_warnings(tmp_path):
    import pytest
    frame = np.zeros((64, 96), np.uint8)
    # errors while flushing are raised by close() (not swallowed); closing again is a no-op
    writer = VideoWriter(tmp_path / 'flush.mp4')
    writer.write(frame)
    writer._mux_packets = lambda packets: (_ for _ in ()).throw(RuntimeError('flush failed'))
    with pytest.raises(RuntimeError, match='flush failed'):
        writer.close()
    writer.close()
    # ... but they do not replace an exception raised inside the with-block
    class MyError(Exception):
        pass
    with pytest.raises(MyError), pytest.warns(UserWarning, match='closing'):
        with VideoWriter(tmp_path / 'masked.mp4') as writer:
            writer.write(frame)
            writer._mux_packets = lambda packets: (_ for _ in ()).throw(RuntimeError('flush failed'))
            raise MyError()
    # empty writers warn, once per file (also with a declared audio stream); a failed __init__ does not
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('default')
        for name in ['empty1.mp4', 'empty2.mp4']:
            VideoWriter(tmp_path / name).close()
        VideoWriter(tmp_path / 'empty_audio.mp4', audio_rate=44100).close()
    assert sum('no file was written' in str(w.message) for w in caught) == 3
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        with pytest.raises(ValueError):
            VideoWriter(tmp_path / 'bad.mp4', fps=float('nan'))
        import gc; gc.collect()

def test_import_does_not_change_warning_filters():
    result = _run_isolated("""
import warnings
import numpy, av, h5py, hdf5plugin # dependencies may add their own filters
warnings.filterwarnings('error', category=RuntimeWarning)
before = list(warnings.filters)
import daio.video, daio.h5
assert warnings.filters == before, warnings.filters[:3]
print('ok')
""")
    assert result.returncode == 0 and 'ok' in result.stdout, result.stderr[-2000:]

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
