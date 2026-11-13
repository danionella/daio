import os
import warnings

import numpy as np
import av

warnings.simplefilter("once")

SCAN_NOTICE_BYTES = 1e9  # print a notice when scanning files larger than this to build a frame index

class VideoReader:
    ''' A class to read video files quickly, using PyAV. It allows for fast seeking and reading of frames.

    Frame index `i` always refers to the i-th frame in presentation order, i.e. `reader[i]` returns the same frame as
    the i-th frame obtained by sequential reading. To map frame indices to time stamps, the reader uses (in order of
    preference, see `index` argument):

    1. the container index (e.g. mp4/mov sample table), if it is complete and the stream has no B-frames (frame
       reordering). The container index holds decode time stamps, which then equal presentation time stamps.
    2. the nominal frame rate, if the container index shows evenly spaced frames (constant frame rate).
    3. a frame index built by scanning all packet time stamps once (demuxing only, no decoding). This reads the
       whole file, and is needed e.g. for variable frame rate videos with B-frames, or mkv/ts containers.

    Args:
        filename (str): path to the video file
        format (str): format to read the video in. Default is 'gray'. Other options include 'rgb24', 'yuv420p', etc.
        threading (bool): enable threading in the decoder. Default is True.
        thread_count (int): number of threads to use for decoding. Default is 0 (auto).
        tolerance (float): tolerance for frame time stamp deviations (fraction of frame interval). Default is 0.1.
        index (str): how to map frame indices to time stamps. 'auto' (default): choose as described above (a file
            scan is also done lazily if a seek lands on an unexpected time stamp). 'scan': always scan the file.
            'nominal': always use the nominal frame rate (legacy behaviour; not reliable for variable frame rates).

    Seeks are verified: the frame found by a seek must be a key frame where decoding starts, decoded time stamps must
    increase, and (with a frame index) the frame must come from the expected position in the file. If a check fails
    (e.g. avi files with B-frames, or with time stamps that change after seeking), a warning is issued and the reader
    switches to sequential decoding (`seekable` becomes False): frames are still returned correctly, but jumping
    backwards decodes from the start of the file, which can be slow.
    '''
    def __init__(self, filename, format='gray', threading=True, thread_count=0, tolerance=0.1, index='auto'):
        if index not in ('auto', 'scan', 'nominal'):
            raise ValueError(f"index must be 'auto', 'scan' or 'nominal', not {index!r}")
        self.filename = filename
        self._threading, self._thread_count = threading, thread_count
        self._open()
        self.read_format = format
        self.index_mode = index
        self._start_time = self.stream.start_time or 0
        self._pts_per_frame = 1 / (self.stream.guessed_rate * self.stream.time_base)
        self.wiggle_pts = tolerance * self._pts_per_frame
        self._frame_pts = None  # sorted presentation time stamps of all frames (None: use nominal frame rate)
        self._keyframe_pts = None  # presentation time stamps of key frames
        self._keyframe_seek_ts = None  # time stamps to seek to for each key frame
        self._frame_pos = None  # file position of each frame's packet (aligned with _frame_pts; -1: unknown)
        self.seekable = True  # False: seeking is unreliable for this file, frames are decoded sequentially
        self.index_source = None  # where the frame index came from: None (nominal frame rate), 'container' or 'scan'
        if index == 'scan':
            self.build_index()
        elif index == 'auto':
            self._auto_index()
        else:
            self._calibrate()
        if self.container.format.name == 'avi' and self.stream.codec_context.has_b_frames:
            self._check_initial_order()
        self.rewind()
        if index == 'nominal' and self.stream.average_rate != self.stream.guessed_rate:
            warn_transcode(f'Average frame rate ({self.stream.average_rate}) is different from nominal frame rate ({self.stream.guessed_rate}). Seeking may be unrealiable. Consider using index="auto" or "scan".')

    def _open(self):
        ''' Open (or re-open) the file and configure the decoder. '''
        if hasattr(self, 'container'):
            self.container.close()
        self.container = av.open(self.filename)
        self.stream = self.container.streams.video[0]
        self.stream.codec_context.thread_count = self._thread_count
        self.stream.codec_context.thread_type = 'AUTO' if self._threading else 'SLICE' # FRAME/AUTO/SLICE
        self.stream.codec_context.flags |= av.codec.context.Flags.copy_opaque # decoded frames carry their packet's file position

    def _container_index(self):
        ''' Read the container index (no file scan). Returns (timestamps, is_keyframe, is_discard, pos) arrays, or None if the
        index is not complete (e.g. mkv/ts, fragmented mp4, avi with frame count mismatch). Timestamps are decode time
        stamps (dts) in units of stream.time_base. '''
        n = self.stream.frames
        entries = self.stream.index_entries
        if not n or len(entries) != n:
            return None
        rows = [(e.timestamp, e.pos, e.is_keyframe, e.is_discard) for e in entries]
        ts, pos, kf, discard = (np.array(col) for col in zip(*rows))
        return ts.astype(np.int64), kf.astype(bool), discard.astype(bool), pos.astype(np.int64)

    def _auto_index(self):
        ''' Choose the cheapest reliable way to map frame indices to time stamps (see class docstring). '''
        cidx = self._container_index()
        if cidx is None:
            return self.build_index()
        ts, kf, discard, pos = cidx
        # container index positions can differ from packet positions by a constant (e.g. avi chunk headers)
        self.rewind()
        first_packet = next((p for p in self.container.demux(self.stream) if p.size), None)
        if first_packet is not None and first_packet.pos >= 0 and pos[0] >= 0:
            pos = pos + (first_packet.pos - pos[0])
        if self.stream.codec_context.has_b_frames == 0:
            # no frame reordering: decode time stamps are presentation time stamps
            keep = ~discard
            self._set_index(ts[keep], ts[keep & kf], ts[keep & kf], source='container', frame_pos=pos[keep])
            return self._calibrate()
        uniform = np.all(np.abs(np.diff(ts) - self._pts_per_frame) <= self.wiggle_pts)
        is_avi = self.container.format.name == 'avi' # avi time stamps are synthesized; always use an index
        if uniform and not discard.any() and self.stream.average_rate == self.stream.guessed_rate and not is_avi:
            first = self._first_frame()
            # constant frame rate: use nominal frame rate, if decoding starts with the first frame in the file
            if first is not None and first.pts is not None and first.opaque is not None and first.opaque[0] == pos[0]:
                self._start_time = first.pts
                return
        self.build_index()

    def _first_frame(self):
        ''' Decode and return the first frame (None if there is none). '''
        self.rewind()
        first = next(self.framegenerator, None)
        self.rewind()
        return first

    def _calibrate(self):
        ''' Align the frame index (or nominal start time) with the first frame the decoder actually outputs: removes
        frames the decoder never outputs (e.g. when a stream starts mid-GOP) and, for the container index, a constant
        offset between container and decoded time stamps. Also flags duplicate time stamps. '''
        first = self._first_frame()
        if first is None or first.pts is None:
            return self._set_unseekable('decoded frames have no time stamps')
        if self._frame_pts is None:
            self._start_time = first.pts
            return
        if self.index_source == 'container':
            ref = None
            if first.opaque is not None and first.opaque[0] >= 0:
                match = np.flatnonzero(self._frame_pos == first.opaque[0])
                ref = self._frame_pts[match[0]] if len(match) else None
            if ref is None:
                return self._set_unseekable('cannot match the container index to decoded frames')
            offset = first.pts - int(ref)
            if offset:
                self._frame_pts += offset
                self._keyframe_pts += offset
        keep = self._frame_pts >= first.pts # frames before the first decoded frame are never output by the decoder
        self._frame_pts = self._frame_pts[keep]
        if self._frame_pos is not None:
            self._frame_pos = self._frame_pos[keep]
        if len(self._frame_pts) == 0 or self._frame_pts[0] != first.pts:
            return self._set_unseekable('frame index does not match decoded frames')
        if np.any(np.diff(self._frame_pts) == 0):
            return self._set_unseekable('duplicate time stamps')

    def _set_index(self, frame_pts, kf_pts, kf_seek_ts, source, frame_pos=None):
        frame_pts = np.asarray(frame_pts, dtype=np.int64)
        order = np.argsort(frame_pts, kind='stable')
        self._frame_pts = frame_pts[order]
        self._frame_pos = None if frame_pos is None else np.asarray(frame_pos, dtype=np.int64)[order]
        order = np.argsort(kf_pts, kind='stable')
        self._keyframe_pts = np.asarray(kf_pts, dtype=np.int64)[order]
        self._keyframe_seek_ts = np.asarray(kf_seek_ts, dtype=np.int64)[order]
        self.index_source = source

    def build_index(self):
        ''' Scan all packets (without decoding) to build an exact map from frame index to presentation time stamp. '''
        try:
            size = os.path.getsize(self.filename)
        except (TypeError, OSError): # e.g. file-like object or URL
            size = 0
        if size > SCAN_NOTICE_BYTES:
            print(f'daio: scanning {os.path.basename(self.filename)} ({size / 1e9:.1f} GB) to build a frame index...', flush=True)
        self.rewind()
        pts, pos, kf_pts, kf_seek = [], [], [], []
        for packet in self.container.demux(self.stream):
            if packet.size == 0 or packet.is_discard: # flush packets or samples discarded by edit lists
                continue
            ts = packet.pts if packet.pts is not None else packet.dts
            if ts is None:
                raise ValueError('Video packets have no time stamps. Cannot build a frame index. Consider transcoding (ffmpeg -y -i "input.mp4" -c:v libx264 -pix_fmt yuv420p -preset superfast -crf 23 "output.mp4").')
            pts.append(ts)
            pos.append(packet.pos)
            if packet.is_keyframe:
                kf_pts.append(ts)
                kf_seek.append(ts if packet.dts is None else min(ts, packet.dts))
        self._set_index(pts, kf_pts, kf_seek, source='scan', frame_pos=pos)
        self._calibrate()
        self.rewind()

    @property
    def has_index(self):
        ''' Whether an exact frame index (from the container index or a file scan) is used. '''
        return self._frame_pts is not None

    def _frame_to_pts(self, n):
        if self._frame_pts is not None:
            return int(self._frame_pts[n])
        return round(n * self._pts_per_frame) + self._start_time

    @property
    def frame_pts(self):
        ''' Presentation time stamps of all frames (in units of stream.time_base). '''
        if self._frame_pts is not None:
            return self._frame_pts.copy()
        return np.array([self._frame_to_pts(i) for i in range(self.nframes)], dtype=np.int64)

    @property
    def frame_times(self):
        ''' Presentation times of all frames (in seconds). '''
        return self.frame_pts * float(self.stream.time_base)

    def read(self):
        ''' Read the next frame in the specified format. '''
        frame_obj = next(self.framegenerator)
        if self.seekable and (frame_obj.pts is None or (self.last_pts is not None and frame_obj.pts <= self.last_pts)):
            self._set_unseekable('decoded time stamps are missing or not increasing')
        self.last_pts = frame_obj.pts
        self._next_idx += 1
        im = frame_obj.to_ndarray(format=self.read_format)
        return im

    def rewind(self):
        ''' Rewind the video to the beginning. '''
        try:
            self.container.seek(0)
        except av.FFmpegError: # e.g. files that do not start with a key frame
            self._open()
        self.framegenerator = self._decode()
        self.last_pts = None
        self._next_idx = 0

    def _decode(self):
        ''' Decode frames from the current position. Each frame carries its packet's file position and key frame flag
        (frame.opaque). '''
        for packet in self.container.demux(self.stream):
            packet.opaque = (packet.pos, packet.is_keyframe)
            yield from packet.decode()

    def _set_unseekable(self, reason):
        if self.seekable:
            self.seekable = False
            warnings.warn(f'{self.filename}: seeking is not reliable for this video ({reason}). Switching to sequential '
                          'decoding: frames are still read correctly, but jumping backwards decodes from the start of '
                          'the file, which can be slow. Consider transcoding (ffmpeg -y -i "input.avi" -c:v libx264 '
                          '-pix_fmt yuv420p -preset superfast -crf 23 "output.mp4").', stacklevel=3)

    def _check_initial_order(self, n=64):
        ''' Check that the first n decoded frames have increasing time stamps (not the case e.g. for some avi files
        with B-frames, where time stamps do not follow display order). '''
        self.rewind()
        last = None
        for _ in range(n):
            frame_obj = next(self.framegenerator, None)
            if frame_obj is None:
                break
            if frame_obj.pts is None or (last is not None and frame_obj.pts <= last):
                self._set_unseekable('decoded time stamps are missing or not in display order')
                break
            last = frame_obj.pts
        self.rewind()

    def _read_sequential(self, frame_idx):
        ''' Read frame frame_idx by decoding sequentially (from the start if needed). '''
        if frame_idx < self._next_idx:
            self.rewind()
        while self._next_idx < frame_idx:
            if next(self.framegenerator, None) is None:
                raise IndexError(f'frame index {frame_idx} out of range')
            self._next_idx += 1
        return self.read()

    def _seek_ts_candidates(self, target_pts, frame_idx):
        ''' Yield time stamps to seek to, going further back each time (to deal with overshoots). '''
        yield target_pts
        if self._keyframe_pts is not None:
            k = np.searchsorted(self._keyframe_pts, target_pts, side='right') - 1
            for j in range(k, max(k - 3, -1), -1):
                yield int(self._keyframe_seek_ts[j])
        elif frame_idx > 100:
            yield self._frame_to_pts(frame_idx - 100)
        yield None # decode from the start

    def _seek_and_decode(self, frame_idx, target_pts):
        ''' Seek to (and decode) the frame with presentation time stamp target_pts. Returns the frame object, or None
        if seeking turned out to be unreliable (the reader then switches to sequential decoding). '''
        exact = self._frame_pts is not None
        wiggle = 0 if exact else self.wiggle_pts
        frame_obj = None
        for seek_ts in self._seek_ts_candidates(target_pts, frame_idx):
            try:
                if seek_ts is None:
                    self.rewind()
                else:
                    self.container.seek(int(seek_ts), backward=True, stream=self.stream)
            except av.FFmpegError:
                continue
            self.framegenerator = self._decode()
            frame_obj = next(self.framegenerator, None)
            if frame_obj is None:
                continue
            problem = None
            if frame_obj.pts is None:
                problem = 'decoded frames have no time stamps'
            elif frame_obj.opaque is not None and frame_obj.opaque[1] and not frame_obj.key_frame:
                problem = 'key frames in the container are not decodable on their own (e.g. intra refresh)'
            elif frame_obj.pts > target_pts + wiggle: # overshoot (seek landed after the target)
                continue
            while problem is None and frame_obj.pts < target_pts - wiggle:
                last = frame_obj.pts
                frame_obj = next(self.framegenerator, None)
                if frame_obj is None:
                    break
                if frame_obj.pts is None or frame_obj.pts <= last:
                    problem = 'decoded time stamps are missing or not in display order'
            if problem is None and frame_obj is not None and exact and frame_obj.pts == target_pts and self._frame_pos is not None:
                expected_pos = self._frame_pos[frame_idx]
                pos = None if frame_obj.opaque is None else frame_obj.opaque[0]
                if expected_pos >= 0 and pos is not None and pos >= 0 and pos != expected_pos:
                    problem = 'time stamps change after seeking'
            if problem is not None:
                self._set_unseekable(problem)
                self.rewind()
                return None
            if frame_obj is None or (exact and frame_obj.pts != target_pts): # target frame was skipped; seek further back
                continue
            return frame_obj
        if frame_obj is None:
            self.rewind()
            raise IndexError(f'could not read frame {frame_idx} (pts {target_pts})')
        return frame_obj

    def read_frame(self, frame_idx):
        ''' Read the specified frame index.

        Args:
            frame_idx (int): index of the frame to read
        '''
        frame_idx = int(frame_idx)
        n = self.nframes
        if n:
            if frame_idx < 0:
                frame_idx += n
            if not 0 <= frame_idx < n:
                raise IndexError(f'frame index {frame_idx} out of range for video with {n} frames')
        elif frame_idx < 0:
            raise IndexError('negative frame indices are not supported for videos with unknown frame count')
        try:
            return self._read_frame(frame_idx)
        except StopIteration:
            self.rewind()
            raise IndexError(f'frame index {frame_idx} out of range') from None

    def _read_frame(self, frame_idx):
        if frame_idx == 0:
            self.rewind()
            return self.read()
        if frame_idx == self._next_idx:
            return self.read()
        if not self.seekable:
            return self._read_sequential(frame_idx)
        target_pts = self._frame_to_pts(frame_idx)
        frame_obj = self._seek_and_decode(frame_idx, target_pts)
        if frame_obj is None:
            return self._read_sequential(frame_idx)
        if np.abs(frame_obj.pts - target_pts) > (0 if self._frame_pts is not None else self.wiggle_pts):
            if self.index_mode == 'auto' and self.index_source != 'scan':
                warnings.warn(f'Unexpected time stamp when seeking to frame {frame_idx} (pts: {frame_obj.pts}; expected: {target_pts}). Irregular frame timing detected; scanning file to build a frame index.', stacklevel=2)
                self.build_index()
                return self._read_frame(frame_idx)
            warn_transcode(f'Seek problem with frame {frame_idx}! pts: {frame_obj.pts}; target: {target_pts}; dts: {frame_obj.dts}; pict_type: {str(frame_obj.pict_type)}')
        frame = frame_obj.to_ndarray(format=self.read_format)
        self.last_pts = frame_obj.pts
        self._next_idx = frame_idx + 1
        return frame

    def close(self):
        if not hasattr(self, 'container'): # __init__ failed before the file was opened
            return
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
        if self._frame_pts is not None:
            return len(self._frame_pts)
        return self.stream.frames

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
        return np.prod(self.shape)
    
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
        self.audio_stream = None
        #define codec defaults:
        if codec == 'h264':
            #kwargs = dict(bit_rate=1000000, pix_fmt='yuv420p').update(kwargs)
            kwargs = {**dict(bit_rate='1000000'), **kwargs}
        elif codec == 'libx264':
            #kwargs = dict(crf=23, preset='superfast', pix_fmt='yuv420p').update(kwargs)
            kwargs = {**dict(crf='23', preset='superfast'), **kwargs}
        #add video stream, if frame_shape is provided (otherwise create when the first frame is written)
        if frame_shape is not None:
            self.stream = self.container.add_stream(codec, rate=fps, height=frame_shape[0], width=frame_shape[1], pix_fmt=pix_fmt, options=kwargs)
        else:
            self.stream = None
            self.codec = codec
            self.fps = fps
            self.pix_fmt = pix_fmt
            self.kwargs = kwargs if kwargs is not None else {}

    def write(self, im, pts=None):
        """ Write a frame to the video file
        
        Args:
            im (np.ndarray): frame to write
            pts (int): presentation timestamp of the frame. Default is None (don't use unles you know what you are doing)
        """
        if im.ndim == 2:
            format = 'gray'
        elif im.ndim == 3 and im.shape[2] == 3:
            format = 'rgb24'
        else:
            raise ValueError(f"Unsupported frame shape: {im.shape}")
        if self.stream is None:
            self.stream = self.container.add_stream(self.codec, rate=self.fps, height=im.shape[0], width=im.shape[1], pix_fmt=self.pix_fmt, options=self.kwargs)
        out_frame = av.VideoFrame.from_ndarray(im, format=format)
        if pts is not None:
            out_frame.pts = pts
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
        if not hasattr(self, 'container'): # __init__ failed before the file was opened
            return
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