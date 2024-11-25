# Run: pytest tests/test_all.py
# All functions to be tested should start with test_ prefix

import numpy as np

def test_trivial():
    assert True == True

def test_trivial2():
    assert False == False

def test_video():
    from daio.video import VideoReader, VideoWriter
    with VideoWriter('test_video.mp4', fps=25) as writer:
        for i in range(20):
            frame = np.random.randint(0,255,size=(720,1280), dtype='uint8')
            writer.write(frame)
    
    with VideoReader('test_video.mp4') as reader:
        for frame in reader:
            frame.mean()

def test_hdf5():
    from daio.h5 import save_to_h5, load_from_h5
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
    save_to_h5('test.h5', data)
    data2 = load_from_h5('test.h5')

    assert data['e']['g'] == data2['e']['g']

