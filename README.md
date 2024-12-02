![Python Version](https://img.shields.io/badge/python-3.8+-blue)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![tests](https://github.com/danionella/daio/actions/workflows/test.yml/badge.svg)
[![PyPI - Version](https://img.shields.io/pypi/v/daio)](https://pypi.org/project/daio/)
[![Conda Version](https://img.shields.io/conda/v/danionella/daio)](https://anaconda.org/danionella/daio)
![GitHub last commit](https://img.shields.io/github/last-commit/danionella/daio)

# daio
Video and data IO tools for Python.

Links: [API documentation](http://danionella.github.io/daio), [GitHub repository](https://github.com/danionella/daio)

## Installation
- via conda or mamba: `conda install danionella::daio `
- if you prefer pip: `pip install daio[video,h5]`
- for development, clone this repository change to the directory containing `pyproject.toml` 
    - `conda env create -n env_name -f environment.yml`
    - `conda activate env_name`
    - `pip install -e .`

## Use 

### Video IO

Write video:
```python
from daio.video import VideoReader, VideoWriter
writer = VideoWriter('/path/to/video.mp4', fps=25)
for i in range(20):
    frame = np.random.randint(0,255,size=(720,1280), dtype='uint8')
    writer.write(frame)
writer.close()
```

Read video using speed-optimized array-like indexing or iteration:
```python
reader = VideoReader('/path/to/video.mp4')
frame_7 = reader[7]
first10_frames = reader[:10]
for frame in reader:
    process_frame(frame)
reader.close()
```

You can also use with statements to handle file closure:
```python
with VideoWriter('/path/to/video.mp4', fps=25) as writer:
    for i in range(20):
        frame = np.random.randint(0,255,size=(720,1280), dtype='uint8')
        writer.write(frame)
#or
with VideoReader('/path/to/video.mp4') as reader:
    frame_7 = reader[7]
```

### HDF5 file IO

Lazily load HDF5 with a dict-like interface (contents are only loaded when accessed):
```python
from daio.h5 import lazyh5
h5 = lazyh5('/path/to/datafile.h5')
b_loaded = h5['b']
e_loaded = h5['c']['e']
h5.keys()
```

Create a new HDF5 file (or add items to existing file by setting argument `readonly=False`):
```python
h5 = lazyh5('test.h5')
h5['a'] = 1
h5['b'] = 'hello'
h5['c'] = {} # create subgroup
h5['c']['e'] = [2,3,4]
```

Load entire HDF5-file to dict, or save dict to HDF5-file:
```python
# save dict to HDF5 file:
some_dict = dict(a = 1, b = np.random.randn(3,4,5), c = dict(g='nested'), d = 'some_string')
lazyh5('/path/to/datafile.h5').from_dict(some_dict)
# load dict from HDF5 file:
loaded = lazyh5('/path/to/datafile.h5').to_dict()
```

In Jupyter, you can interactively explore the file structure:

<img width="598" alt="image" src="https://github.com/user-attachments/assets/878d4af6-ce26-4ec3-9ba5-2701fba1c07e">


-----
<details><summary>Old interface (expand this)</summary>
    
```python
from daio.h5 import save_to_h5, load_from_h5
# save dict to HDF5 file:
some_dict = dict(a = 1, b = np.random.randn(3,4,5), c = dict(g='nested'), d = 'some_string')
save_to_h5('/path/to/datafile.h5', some_dict)
# load dict from HDF5 file:
dict_loaded = load_from_h5('/path/to/datafile.h5')
```

</details>
