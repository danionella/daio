import warnings

import numpy as np
import av

warnings.simplefilter("once")

class VideoReader:
    ''' A class to read video files quickly, using PyAV. It allows for fast seeking and reading of frames. 
    
    Args:
        filename (str): path to the video file
        format (str): format to read the video in. Default is 'gray'. Other options include 'rgb24', 'yuv420p', etc.
        threading (bool): enable threading in the decoder. Default is True.
        thread_count (int): number of threads to use for decoding. Default is 0 (auto).
        tolerance (float): tolerance for frame time stamp deviations (fraction of frame interval). Default is 0.1.
    '''
    def __init__(self, filename, format='gray', threading=True, thread_count=0, tolerance=0.1):
        self.container = av.open(filename)
        self.stream = self.container.streams.video[0]
        self.stream.codec_context.thread_count = thread_count
        self.stream.codec_context.thread_type = 'AUTO' if threading else 'SLICE' # FRAME/AUTO/SLICE
        self.framegenerator = self.container.decode(video=0)
        self.read_format = format
        self._pts_per_frame = 1 / (self.stream.guessed_rate * self.stream.time_base)
        self._frame_to_pts = lambda n: round(n * self._pts_per_frame) + self.stream.start_time
        self.wiggle_pts = tolerance * self._pts_per_frame
        self.rewind()
        if self.container.format.variable_fps:
            warn_transcode(f'Variable frame rate video detected. Seeking will likely be unrealiable. I will warn again if I detect seek gitches')
        if self.stream.average_rate != self.stream.guessed_rate:
            warn_transcode(f'Average frame rate ({self.stream.average_rate}) is different from nominal frame rate ({self.stream.guessed_rate}). Seeking may be unrealiable. I will warn again if I detect seek gitches')

    def read(self):
        ''' Read the next frame in the specified format. '''
        frame_obj = next(self.framegenerator)
        self.last_pts = frame_obj.pts
        im = frame_obj.to_ndarray(format=self.read_format)
        return im

    def rewind(self):
        ''' Rewind the video to the beginning. '''
        self.container.seek(0)
        self.framegenerator = self.container.decode(video=0)
        self.last_pts = None

    def read_frame(self, frame_idx):
        ''' Read the specified frame index. 
        
        Args:
            frame_idx (int): index of the frame to read
        '''
        if frame_idx == 0:
            self.rewind()
            return self.read()
        if self.last_pts is not None and (np.abs(self.last_pts - self._frame_to_pts(frame_idx-1)) <= self.wiggle_pts):
            return self.read()
        target_pts = self._frame_to_pts(frame_idx)
        self.container.seek(target_pts-self.stream.start_time, backward=True, stream=self.container.streams.video[0])
        self.framegenerator = self.container.decode(video=0)
        frame_obj = next(self.framegenerator)
        if frame_obj.pts > target_pts: #detecting overshoot (may happen due to variable frame rate)
            n_back = 100
            warn_transcode(f'Seek overshoot ({frame_obj.pts} > {target_pts}). Backtracking by {n_back} frames...')
            self.container.seek(self._frame_to_pts(frame_idx-n_back)-self.stream.start_time, backward=True, stream=self.container.streams.video[0])
            self.framegenerator = self.container.decode(video=0)
            frame_obj = next(self.framegenerator)
        while frame_obj.pts < (target_pts - self.wiggle_pts): 
            frame_obj = next(self.framegenerator)
        # frame_obj.pts should now be equal to target_pts
        if np.abs(frame_obj.pts - target_pts) > self.wiggle_pts:
            warn_transcode(f'Seek problem with frame {frame_idx}! pts: {frame_obj.pts}; target: {target_pts}; dts: {frame_obj.dts}; pict_type: {str(frame_obj.pict_type)}')
        frame = frame_obj.to_ndarray(format=self.read_format)
        self.last_pts = frame_obj.pts
        return frame
    
    def close(self):
        self.container.close()
        
    def __del__(self):
        self.close()
        
    def __getitem__(self, index):
        if isinstance(index, (int, np.integer)):  # single frame
            return self.read_frame(index)
        elif isinstance(index, slice):
            frames = [self.__getitem__(i) for i in np.r_[index]]
            return np.array(frames)
        elif isinstance(index, tuple):
            frames = [self.__getitem__(i)[index[1:]] for i in np.r_[index[0]]]
            if len(frames) == 1:
                return frames[0]
            else:
                return np.array(frames)
        else:
            raise NotImplementedError(f"slicing of {type(index)} : {index} not implemented yet")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False
    
    def __iter__(self):
        self.rewind()
        return self
    
    def __next__(self):
        try:
            return self.read()
        except:
            self.rewind()
            raise StopIteration
    
    @property
    def frame_shape(self):
        ''' Return the shape of the video frames. '''
        return self.container.streams.video[0].codec_context.height, self.container.streams.video[0].codec_context.width

    @property
    def nframes(self):
        ''' Return the number of frames in the video. '''
        return self.container.streams.video[0].frames

    @property
    def dtype(self):
        return np.uint8

    @property
    def shape(self):
        if self.read_format == 'gray':
            return (self.nframes, *self.frame_shape)
        elif self.read_format in ['rgb24', 'bgr24', 'yuv420p']:
            return (self.nframes, *self.frame_shape, 3)
        else:
            raise NotImplementedError(f"shape not implemented for format {self.read_format}")

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def size(self):
        return np.product(self.shape)
    
    @staticmethod
    def static_shape(filename):
        ''' Get the shape of a video (static method). 
        
        Args:
            filename (str): path to the video file'''
        with av.open(filename) as container:
            stream = container.streams.video[0]
            shape = np.array([stream.frames, stream.codec_context.height, stream.codec_context.width])
        return shape


def warn_transcode(msg):
    warnings.warn(msg, stacklevel=2)
    warnings.warn('Consider transcoding (ffmpeg -y -i "input.mp4" -c:v libx264 -pix_fmt yuv420p -preset superfast -crf 23 "output.mp4").', stacklevel=2)


class VideoWriter:
    ''' A class to write video files, using PyAV.
    
    Args:
        filename (str): path to the output video file
        codec (str): codec to use. Default is 'libx264'. Other options include 'h264', etc. Check VideoWriter.codecs_available for a list of available codecs.
        fps (int): frames per second. Default is 25.
        frame_shape (tuple): shape of the frames. Default is None (to be determined by the first frame written).
        **kwargs: additional keyword arguments to pass to the codec. Check e.g. `ffmpeg -h encoder=libx264`
    '''

    def __init__(self, filename, codec='libx264', fps=25, frame_shape=None, pix_fmt='yuv420p', **kwargs):
        self.container = av.open(filename, mode='w')
        #define codec defaults:
        if codec == 'h264':
            kwargs = dict(bit_rate=1000000, pix_fmt='yuv420p').update(kwargs)
        elif codec == 'libx264':
            kwargs = dict(crf=23, preset='superfast', pix_fmt='yuv420p').update(kwargs)
        #add video stream, if frame_shape is provided (otherwise create when the first frame is written)
        if frame_shape is not None:
            self.stream = self.container.add_stream(codec, rate=fps, height=frame_shape[0], width=frame_shape[1], pix_fmt=pix_fmt, **kwargs)
        else:
            self.stream = None
            self.codec = codec
            self.fps = fps
            self.pix_fmt = pix_fmt
            self.kwargs = kwargs if kwargs is not None else {}

    def write(self, im):
        """ Write a frame to the video file
        
        Args:
            im (np.ndarray): frame to write
        """
        if im.ndim == 2:
            format = 'gray'
        elif im.ndim == 3 and im.shape[2] == 3:
            format = 'rgb24'
        else:
            raise ValueError(f"Unsupported frame shape: {im.shape}")
        if self.stream is None:
            self.stream = self.container.add_stream(self.codec, rate=self.fps, height=im.shape[0], width=im.shape[1], pix_fmt=self.pix_fmt, **self.kwargs)
        out_frame = av.VideoFrame.from_ndarray(im, format=format)
        for packet in self.stream.encode(out_frame):
            self.container.mux(packet)

    def write_frames(self, frames):
        for frame in frames:
            self.write(frame)

    def write_audio(self, audio_chunk, audio_rate, pts=None, stream_args=None):
        """ Write an audio chunk to the video file

        Args:
            audio_chunk (np.ndarray): audio chunk to write
            audio_rate (int): audio sample rate
            pts (int): presentation timestamp of the audio chunk. Default is None (auto-increment)
            stream_args (dict): additional arguments to pass to the audio stream.
        """
        if stream_args is None:
            stream_args = dict(codec_name='aac', rate=44100)
        if self.audio_stream is None:
            self.audio_stream = self.container.add_stream(**stream_args)
            self.channel_layout = 'stereo' if (audio_chunk.ndim > 1 and audio_chunk.shape[0] == 2) else 'mono'
            self.audio_pts = 0
        audio_frame = av.AudioFrame.from_ndarray(audio_chunk, layout=self.channel_layout)
        audio_frame.sample_rate = audio_rate
        audio_frame.pts = self.audio_pts if pts is None else pts
        self.audio_pts = audio_frame.pts + audio_chunk.shape[1]
        for packet in self.audio_stream.encode(audio_frame):
            self.container.mux(packet)

    def close(self):
        """ Close the video file. """
        try:
            for packet in self.stream.encode():
                self.container.mux(packet)
        except:
            pass
        
        try: 
            for packet in self.audio_stream.encode():
                self.container.mux(packet)
        except:
            pass

        self.container.close()

    def __del__(self):
        self.close()
            
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False
    
    @property
    def codecs_available(self):
        ''' Return a list of available codecs. '''
        return sorted(av.codec.codecs_available)