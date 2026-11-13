import os
import warnings

import numpy as np
import av
from fractions import Fraction


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
        n_corrupt = 0
        for packet in self.container.demux(self.stream):
            if packet.size == 0 or packet.is_discard: # flush packets or samples discarded by edit lists
                continue
            n_corrupt += packet.is_corrupt
            ts = packet.pts if packet.pts is not None else packet.dts
            if ts is None:
                raise ValueError('Video packets have no time stamps. Cannot build a frame index. Consider transcoding (ffmpeg -y -i "input.mp4" -c:v libx264 -pix_fmt yuv420p -preset superfast -crf 23 "output.mp4").')
            pts.append(ts)
            pos.append(packet.pos)
            if packet.is_keyframe:
                kf_pts.append(ts)
                kf_seek.append(ts if packet.dts is None else min(ts, packet.dts))
        if n_corrupt:
            self._warn_damaged(f'{n_corrupt} damaged packet(s) found, e.g. a truncated recording')
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
        if frame_obj.is_corrupt:
            self._warn_damaged(f'frame {self._next_idx} is corrupt')
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
        self._landing_packet = None # first packet after a seek: (pos, is_keyframe, is_discard)
        for packet in self.container.demux(self.stream):
            packet.opaque = (packet.pos, packet.is_keyframe)
            if self._landing_packet is None and packet.size:
                self._landing_packet = (packet.pos, packet.is_keyframe, packet.is_discard)
            try:
                frames = packet.decode()
            except av.InvalidDataError:
                if not packet.is_corrupt:
                    raise
                self._warn_damaged(f'undecodable damaged packet at byte {packet.pos} skipped')
                continue
            yield from frames

    def _warn_damaged(self, detail):
        if not getattr(self, '_warned_damaged', False):
            self._warned_damaged = True
            warnings.warn(f'{self.filename}: video contains damaged data ({detail}). Frames decoded from damaged data '
                          'may differ depending on how they are reached (sequentially or by seeking).', stacklevel=3)

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
            elif seek_ts is not None and not frame_obj.key_frame and not (self._landing_packet and self._landing_packet[2]):
                # (unless the seek landed on a key frame that an edit list discards)
                problem = 'decoding after a seek does not start at a key frame (e.g. intra refresh)'
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
        if frame_obj.is_corrupt:
            self._warn_damaged(f'frame {frame_idx} is corrupt')
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
        
    def _frame_indices(self, index):
        ''' Frame indices selected by a slice, or by a list/array of integers. '''
        if isinstance(index, slice):
            n = self.nframes
            if not n:
                if index.start is None or index.stop is None or index.start < 0 or index.stop < 0 or (index.step or 1) < 0:
                    raise IndexError('open-ended or negative slices need a known frame count')
                return range(index.start, index.stop, index.step or 1)
            return range(*index.indices(n))
        indices = np.asarray(index)
        if indices.dtype == bool or not np.issubdtype(indices.dtype, np.integer) or indices.ndim != 1:
            raise IndexError(f'frames can be indexed with integers, slices or 1-d integer arrays, not {index!r}')
        return [int(i) for i in indices]

    def _stack(self, frames):
        if frames:
            return np.array(frames)
        try:
            return np.empty((0, *self.shape[1:]), dtype=self.dtype)
        except NotImplementedError:
            return np.empty((0,), dtype=self.dtype)

    def __getitem__(self, index):
        if isinstance(index, (int, np.integer)):  # single frame
            return self.read_frame(index)
        if isinstance(index, tuple):
            first, rest = index[0], index[1:]
            if isinstance(first, (int, np.integer)):
                return self.read_frame(first)[rest]
            frames = [self.read_frame(i)[rest] for i in self._frame_indices(first)]
            if not frames:
                return self._stack([])[(slice(None), *rest)]
            return np.array(frames)
        return self._stack([self.read_frame(i) for i in self._frame_indices(index)])

    def __len__(self):
        return self.nframes

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False

    def __iter__(self):
        ''' Iterate over all frames. Each loop keeps its own position, so nested loops or zip(reader, reader) work. '''
        i = 0
        while not self.nframes or i < self.nframes:
            try:
                frame = self.read_frame(i)
            except IndexError:
                if self.nframes: # end of video reached early (e.g. damaged file)
                    warnings.warn(f'{self.filename}: video ended after {i} frames, expected {self.nframes}.', stacklevel=2)
                return
            yield frame
            i += 1

    def __next__(self):
        ''' Read the next frame (same as read()). '''
        return self.read()

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


class _classproperty:
    def __init__(self, fget):
        self.fget = fget

    def __get__(self, obj, cls):
        return self.fget(cls)


class VideoWriter:
    """ A class to write video files, using PyAV.
    
    Args:
        filename (str): path to the output video file
        codec (str): codec to use. Default is 'libx264'. Other options include 'h264', etc. Check VideoWriter.codecs_available for a list of available codecs.
        fps (int, float or Fraction): frames per second. Default is 25. Floats are converted to fractions (e.g. 29.97 -> 2997/100).
        frame_shape (tuple): shape of the frames. Default is None (to be determined by the first frame written).
        pix_fmt (str): pixel format of the encoded video. Default is 'yuv420p' (with libx264/libx265, frame height and
            width must then be even).
        audio_rate (int): sample rate of an audio stream to write with `write_audio`. Default is None (the audio stream
            is created by the first call to `write_audio`, which must then come before the first video frame is
            written, unless frame_shape is None). Audio written before the video stream exists is kept in memory
            until the first video frame is written (or until close()).
        audio_channels (int): number of audio channels (1 or 2) if audio_rate is given. Default is 1.
        **kwargs: additional keyword arguments to pass to the codec. Check e.g. `ffmpeg -h encoder=libx264`
    """

    def __init__(self, filename, codec='libx264', fps=25, frame_shape=None, pix_fmt='yuv420p', audio_rate=None, audio_channels=1, **kwargs):
        self.container = av.open(filename, mode='w')
        self._closed = False
        self._n_muxed = 0
        self._pending = []  # packets held back until the video stream exists (streams cannot be added once muxing has started)
        self._muxing = False
        self._next_pts = 0
        self.stream = None
        self.audio_stream = None
        #define codec defaults:
        if codec == 'h264':
            kwargs = {**dict(bit_rate='1000000'), **kwargs}
        elif codec == 'libx264':
            kwargs = {**dict(crf='23', preset='superfast'), **kwargs}
        try:
            self.codec = codec
            self.fps = Fraction(fps).limit_denominator(1001) if isinstance(fps, float) else fps
            self.pix_fmt = pix_fmt
            self.kwargs = {k: str(v) for k, v in kwargs.items() if v is not None} # codec options must be strings
            if audio_rate is not None:
                self._add_audio_stream(audio_rate, audio_channels)
            #add video stream, if frame_shape is provided (otherwise create when the first frame is written)
            if frame_shape is not None:
                self._add_video_stream(frame_shape[0], frame_shape[1])
        except BaseException:
            self._closed = True
            self.container.close()
            raise

    def _add_video_stream(self, height, width):
        stream = self.container.add_stream(self.codec, rate=self.fps, height=height, width=width, pix_fmt=self.pix_fmt, options=self.kwargs)
        try:
            stream.codec_context.open() # open the encoder now, so that unsupported settings raise a clear error here
        except av.FFmpegError as e:
            # the file cannot be completed with a stream whose encoder failed to open: close the writer
            self._closed = True
            self.container.close()
            comps = av.VideoFormat(self.pix_fmt, 2, 2).components
            subsampled = min(c.width for c in comps) < 2 or min(c.height for c in comps) < 2
            if (height % 2 or width % 2) and subsampled:
                raise ValueError(f'The {self.codec} encoder does not support frame size {height}x{width} (height x width) '
                                 f'with pix_fmt={self.pix_fmt!r}: height and width must be even. Crop or pad the frames, '
                                 f'or use e.g. pix_fmt="yuv444p".') from e
            raise
        self.stream = stream
        self.frame_shape = (height, width)
        self._mux_packets(self._pending)
        self._pending = []

    def _add_audio_stream(self, audio_rate, channels, stream_args=None):
        if channels not in (1, 2):
            raise ValueError(f'Only mono and stereo audio are supported, not {channels} channels.')
        if self._muxing:
            raise ValueError('Cannot add an audio stream after video frames were written. Pass audio_rate (and '
                             'audio_channels) to VideoWriter() to write audio and video in any order.')
        self.channel_layout = 'mono' if channels == 1 else 'stereo'
        if stream_args is None:
            stream_args = dict(codec_name='aac', rate=audio_rate, layout=self.channel_layout)
        self.audio_stream = self.container.add_stream(**stream_args)
        self.audio_rate = audio_rate
        self.audio_channels = channels
        self.audio_pts = 0

    def _mux_packets(self, packets):
        if self.stream is None: # the video stream may still be added: hold back packets
            self._pending.extend(packets)
            return
        for packet in packets:
            self._muxing = True
            self.container.mux(packet)
            self._n_muxed += 1

    def write(self, im, pts=None):
        """ Write a frame to the video file
        
        Args:
            im (np.ndarray): frame to write (uint8; shape (height, width) for grayscale or (height, width, 3) for RGB)
            pts (int): presentation timestamp of the frame, in units of 1/fps. Must increase from frame to frame.
                Default is None (the frame after the previous one; don't use unless you know what you are doing)
        """
        if self._closed:
            raise ValueError('Cannot write to a closed VideoWriter.')
        if im.ndim == 2:
            format = 'gray'
        elif im.ndim == 3 and im.shape[2] == 3:
            format = 'rgb24'
        else:
            raise ValueError(f"Unsupported frame shape: {im.shape}")
        if self.stream is None:
            self._add_video_stream(im.shape[0], im.shape[1])
        elif im.shape[:2] != self.frame_shape:
            raise ValueError(f'Frame shape {im.shape[:2]} differs from the video frame shape {self.frame_shape}.')
        if pts is None:
            pts = self._next_pts
        elif pts != int(pts):
            raise ValueError(f'pts must be an integer (in units of 1/fps), got {pts}.')
        elif int(pts) < self._next_pts:
            raise ValueError(f'pts must increase from frame to frame: got {pts} after {self._next_pts - 1}.')
        out_frame = av.VideoFrame.from_ndarray(im, format=format)
        out_frame.pts = int(pts)
        self._mux_packets(self.stream.encode(out_frame))
        self._next_pts = int(pts) + 1

    def write_frames(self, frames):
        for frame in frames:
            self.write(frame)

    def write_audio(self, audio_chunk, audio_rate=None, pts=None, stream_args=None):
        """ Write an audio chunk to the video file

        Args:
            audio_chunk (np.ndarray): audio chunk to write, with shape (channels, samples) or (samples,) for mono.
                dtype int16, or float32/float64 in the range [-1, 1].
            audio_rate (int): audio sample rate. Default is None (the audio_rate passed to VideoWriter())
            pts (int): presentation timestamp of the audio chunk, in samples. Default is None (auto-increment)
            stream_args (dict): additional arguments to pass to the audio stream (if it is created by this call).
        """
        if self._closed:
            raise ValueError('Cannot write to a closed VideoWriter.')
        audio_chunk = np.asarray(audio_chunk)
        if audio_chunk.ndim == 1:
            audio_chunk = audio_chunk[None]
        if audio_chunk.ndim != 2:
            raise ValueError(f'audio_chunk must have shape (channels, samples), got {audio_chunk.shape}.')
        if self.audio_stream is None:
            if audio_rate is None:
                raise ValueError('audio_rate is required.')
            self._add_audio_stream(audio_rate, audio_chunk.shape[0], stream_args)
        if audio_chunk.shape[0] != self.audio_channels:
            raise ValueError(f'audio_chunk has {audio_chunk.shape[0]} channels, but the audio stream has {self.audio_channels}. '
                             f'Use shape (channels, samples).')
        if audio_chunk.dtype == np.int16:
            sample_format = 's16p'
        elif audio_chunk.dtype.kind == 'f':
            audio_chunk, sample_format = audio_chunk.astype(np.float32), 'fltp'
        else:
            raise ValueError(f'audio_chunk must be int16 or float, not {audio_chunk.dtype}.')
        audio_frame = av.AudioFrame.from_ndarray(np.ascontiguousarray(audio_chunk), format=sample_format, layout=self.channel_layout)
        audio_frame.sample_rate = self.audio_rate if audio_rate is None else audio_rate
        audio_frame.pts = self.audio_pts if pts is None else pts
        self.audio_pts = audio_frame.pts + audio_chunk.shape[1]
        self._mux_packets(self.audio_stream.encode(audio_frame))

    def close(self):
        """ Close the video file. """
        if not hasattr(self, 'container'): # __init__ failed before the file was opened
            return
        if getattr(self, '_closed', True):
            return
        self._closed = True
        try:
            for stream in (self.stream, self.audio_stream):
                # flush the encoder (not if it could not be opened, e.g. after an unsupported frame size: PyAV crashes)
                if stream is not None and stream.codec_context.is_open:
                    self._mux_packets(stream.encode())
            for packet in self._pending: # audio only: packets held back for a video stream that was never added
                self.container.mux(packet)
                self._n_muxed += 1
            self._pending = []
            if self._n_muxed == 0 and not getattr(self, '_exiting_on_error', False):
                warnings.warn('VideoWriter closed without any frames: no file was written.', stacklevel=2)
        finally:
            self.container.close()

    def __del__(self):
        try:
            self.close()
        except Exception: # errors cannot be raised from __del__; call close() (or use a with statement) to see them
            pass
            
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self._exiting_on_error = exc_type is not None
        self.close()
        return False
    
    @_classproperty
    def codecs_available(cls):
        """ Return a list of available codecs. """
        return sorted(av.codec.codecs_available)
