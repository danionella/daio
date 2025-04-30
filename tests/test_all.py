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
        for frame in reader:
            frame.mean()

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

