import json
import os
import warnings

import numpy as np
import av
from fractions import Fraction


SCAN_NOTICE_BYTES = 1e9  # print a notice when scanning files larger than this to build a frame index

class _TagReader:
    ''' Metadata tag access for readers with an open PyAV container (self.container). '''

    def list_tags(self):
        ''' Metadata tags of the file, as exposed by the container format.

        Returns:
            dict: {'container': {name: value}, 'streams': {stream index: {name: value}}}. Values are strings. Which
            tags are available depends on the container: e.g. mp4 files keep custom container tags only if written
            with movflags=use_metadata_tags, mkv returns custom tag names in upper case, and avi drops them.
        '''
        return {'container': dict(self.container.metadata),
                'streams': {s.index: dict(s.metadata) for s in self.container.streams}}

    def read_tag(self, name, default=None, parse_json=True, stream=None):
        ''' Read one metadata tag. The name is matched exactly, or else ignoring case (mkv stores tag names in upper case).

        Args:
            name (str): tag name, e.g. 'settings'
            default: value returned if the tag does not exist. Default is None.
            parse_json (bool): if the value is valid JSON, return the parsed object (e.g. a settings dict). Note that
                plain numbers are valid JSON too ('512' -> 512). Default is True.
            stream (int or str): read a tag of a stream (by index, or by its handler_name tag) instead of the
                container. Default is None (container tags).

        Example:
            >>> settings = reader.read_tag('settings')  # dict, if stored as JSON
            >>> reader.read_tag('handler_name', stream=0)
        '''
        tags = self.container.metadata if stream is None else _find_stream(self.container, stream).metadata
        value = _get_tag(tags, name)
        if value is None:
            return default
        if parse_json:
            try:
                return json.loads(value)
            except ValueError:
                pass
        return value


class VideoReader(_TagReader):
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
        stream (int or str): video stream to read: its stream index (as in list_tags()), or its handler_name tag.
            Default is None (the first video stream).

    Seeks are verified: the frame found by a seek must be a key frame where decoding starts, decoded time stamps must
    increase, and (with a frame index) the frame must come from the expected position in the file. If a check fails
    (e.g. avi files with B-frames, or with time stamps that change after seeking), a warning is issued and the reader
    switches to sequential decoding (`seekable` becomes False): frames are still returned correctly, but jumping
    backwards decodes from the start of the file, which can be slow.
    '''
    def __init__(self, filename, format='gray', threading=True, thread_count=0, tolerance=0.1, index='auto', stream=None):
        if index not in ('auto', 'scan', 'nominal'):
            raise ValueError(f"index must be 'auto', 'scan' or 'nominal', not {index!r}")
        self.filename = filename
        self._stream_spec = stream
        self._threading, self._thread_count = threading, thread_count
        self._open()
        self._rewind_reopens = not self._seek_reaches_start()
        self.read_format = format
        self.index_mode = index
        self._start_time = self.stream.start_time or 0
        self._pts_per_frame = 1 / (self.stream.guessed_rate * self.stream.time_base)
        self.wiggle_pts = tolerance * self._pts_per_frame
        self._frame_pts = None  # sorted presentation time stamps of all frames (None: use nominal frame rate)
        self._times_cache = None  # (frame_pts, frame_times) as returned by these properties (read-only arrays)
        self._keyframe_pts = None  # presentation time stamps of key frames
        self._keyframe_seek_ts = None  # time stamps to seek to for each key frame
        self._frame_pos = None  # file position of each frame's packet (aligned with _frame_pts; -1: unknown)
        self._nframes_counted = None  # frame count of files without frame index (counted packets)
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
        if hasattr(self.filename, 'seek'): # file-like object: (re)open from its start
            self.filename.seek(0)
        self.container = av.open(self.filename)
        self.stream = _find_stream(self.container, self._stream_spec, 'video')
        self.stream.codec_context.thread_count = self._thread_count
        self.stream.codec_context.thread_type = 'AUTO' if self._threading else 'SLICE' # FRAME/AUTO/SLICE
        self.stream.codec_context.flags |= av.codec.context.Flags.copy_opaque # decoded frames carry their packet's file position

    def _first_packet_pos(self):
        packet = next((p for p in self.container.demux(self.stream) if p.size), None)
        return None if packet is None else packet.pos

    def _seek_reaches_start(self):
        ''' Whether seeking to 0 returns to the first packet in the file (not the case e.g. for files whose first
        frames have negative or no time stamps; the file is then re-opened to rewind). '''
        first = self._first_packet_pos()
        if first is None or first < 0: # packet positions unknown: cannot check, re-open to rewind (always safe)
            self._open()
            return False
        try:
            self.container.seek(0)
            ok = self._first_packet_pos() == first
        except av.FFmpegError:
            ok = False
        if ok:
            self.container.seek(0)
        else:
            self._open()
        return ok

    def _container_index(self):
        ''' Read the container index (no file scan). Returns (timestamps, is_keyframe, is_discard, pos, verified) or None
        if there is no usable index (e.g. mkv/ts: key frames only, avi with frame count mismatch). Timestamps are decode
        time stamps (dts) in units of stream.time_base. verified is True if the index length matches the frame count in
        the header. Without a frame count (e.g. fragmented mp4), the index needs to be checked (_index_tail_matches);
        entries beyond the end of the file (cut-off recordings) are dropped. '''
        n = self.stream.frames
        entries = self.stream.index_entries
        if not len(entries) or (n and len(entries) != n):
            return None
        rows = [(e.timestamp, e.pos, e.is_keyframe, e.is_discard, e.size) for e in entries]
        ts, pos, kf, discard, size = (np.array(col) for col in zip(*rows))
        if not n: # no frame count: only a complete per-frame index can be used (not a key-frames-only index)
            if kf.all():
                return None
            file_size = getattr(self.container, 'size', None)
            if file_size is not None and file_size > 0:
                inside = pos + size <= file_size
                ts, kf, discard, pos = ts[inside], kf[inside], discard[inside], pos[inside]
                if not kf.any():
                    return None
        return ts.astype(np.int64), kf.astype(bool), discard.astype(bool), pos.astype(np.int64), bool(n)

    def _index_tail_matches(self, ts, kf, pos):
        ''' Check an index without frame count: the packets from its last key frame to the end of the file must be
        exactly its last entries (by file position, in order), followed by nothing but damaged packets (cut-off
        recordings). Only packets are read, no frames decoded; stops at the first mismatch. '''
        k = int(np.flatnonzero(kf)[-1])
        index_of = {int(p): i for i, p in enumerate(pos)}
        expected = None # index entry of the next packet
        try:
            self.container.seek(int(ts[k]), backward=True, stream=self.stream)
            for packet in self.container.demux(self.stream):
                if packet.size == 0 or packet.is_discard:
                    continue
                if expected is None: # first packet after the seek: must be in the index, at or before entry k
                    expected = index_of.get(packet.pos)
                    if expected is None or expected > k:
                        return False
                if expected < len(pos):
                    if packet.pos != pos[expected]:
                        return False
                    expected += 1
                elif not packet.is_corrupt: # more packets than the index lists
                    return False
        except av.FFmpegError:
            return False
        finally:
            self.rewind()
        return expected == len(pos)

    def _auto_index(self):
        ''' Choose the cheapest reliable way to map frame indices to time stamps (see class docstring). '''
        cidx = self._container_index()
        if cidx is None:
            return self.build_index()
        ts, kf, discard, pos, verified = cidx
        # container index positions can differ from packet positions by a constant (e.g. avi chunk headers)
        self.rewind()
        first_pos = self._first_packet_pos()
        if first_pos is not None and first_pos >= 0 and pos[0] >= 0:
            pos = pos + (first_pos - pos[0])
        if self.stream.codec_context.has_b_frames == 0:
            # no frame reordering: decode time stamps are presentation time stamps
            keep = ~discard
            if not verified and not self._index_tail_matches(ts[keep], kf[keep], pos[keep]):
                return self.build_index()
            self._set_index(ts[keep], ts[keep & kf], ts[keep & kf], source='container', frame_pos=pos[keep])
            return self._calibrate()
        if not verified: # (the nominal frame rate path below needs the frame count)
            return self.build_index()
        uniform = np.all(np.abs(np.diff(ts) - float(self._pts_per_frame)) <= float(self.wiggle_pts))
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
        self._times_cache = None
        first = self._first_frame()
        if first is None or first.pts is None:
            return self._set_unseekable('decoded frames have no time stamps')
        if self._frame_pts is None:
            self._start_time = first.pts
            return
        if self.index_source == 'container':
            ref = None
            if first.opaque is not None and first.opaque[0] is not None and first.opaque[0] >= 0:
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
        self._times_cache = None
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
        first_key = None # index of the first key packet
        for packet in self.container.demux(self.stream):
            if packet.size == 0 or packet.is_discard: # flush packets or samples discarded by edit lists
                continue
            n_corrupt += packet.is_corrupt
            ts = packet.pts if packet.pts is not None else packet.dts
            if packet.is_keyframe and first_key is None:
                first_key = len(pts)
            pts.append(ts)
            pos.append(packet.pos if packet.pos is not None else -1)
            if packet.is_keyframe:
                kf_pts.append(ts)
                kf_seek.append(ts if packet.dts is None else min(ts, packet.dts))
        if n_corrupt:
            self._warn_damaged(f'{n_corrupt} damaged packet(s) found, e.g. a truncated recording')
        if any(ts is None for ts in pts): # no frame index possible: count frames, decode sequentially
            self._frame_pts = self._frame_pos = None
            self._nframes_counted = len(pts) - (first_key or 0) # (packets before the first key frame are not decoded)
            self.index_source = 'scan'
            self._set_unseekable('some frames have no time stamps')
            self.rewind()
            return
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
        ''' Presentation time stamps of all frames (in units of stream.time_base; a read-only array). Without a frame
        index (nominal frame rate, or files without time stamps), these are computed from the nominal frame rate. '''
        return self._frame_times_cached()[0]

    @property
    def frame_times(self):
        ''' Presentation times of all frames (in seconds; a read-only array). '''
        return self._frame_times_cached()[1]

    def _frame_times_cached(self):
        n = self.nframes # (without a frame index, the frame count can shrink while reading: then recompute)
        if self._times_cache is None or len(self._times_cache[0]) != n:
            if self._frame_pts is not None:
                pts = self._frame_pts.copy()
            else: # (as _frame_to_pts)
                step = Fraction(self._pts_per_frame) # (exact, rounded half to even like round(n * step))
                q, r = np.divmod(np.arange(n, dtype=np.int64) * step.numerator, step.denominator)
                pts = q + ((2 * r > step.denominator) | ((2 * r == step.denominator) & (q % 2 == 1))) + self._start_time
            times = pts * float(self.stream.time_base)
            pts.flags.writeable = times.flags.writeable = False
            self._times_cache = (pts, times)
        return self._times_cache

    def frame_time(self, index):
        ''' Presentation time (in seconds) of a frame index (an integer or an array of integers; negative indices count
        from the end), on the same clock as AudioReader.sample_time. '''
        k = np.asarray(index)
        if k.dtype.kind not in 'iu':
            raise TypeError(f'frame indices must be integers, not {k.dtype}')
        n = self.nframes
        if np.any((k < -n) | (k >= n)):
            raise IndexError(f'frame index out of range for video with {n} frames')
        return _scalar_or_array(self.frame_times[k])

    def frame_index(self, time, rounding='round'):
        ''' Frame index at a time (in seconds, on the clock of frame_times; a number or an array of times). E.g. the
        frame shown with an audio sample: reader.frame_index(audio.sample_time(i)).

        Args:
            time (float or array): time(s) in seconds
            rounding: 'round' (default): the frame with the nearest time (-1 or nframes for times more than half a
                frame interval before the first or after the last frame). 'floor': the frame shown at that time, i.e.
                the last frame at or before it (-1 for times before the first frame; the last frame for times after
                it). 'ceil': the first frame at or after that time (nframes for times after the last frame). None: a
                fractional index, interpolated between frame times (extrapolated beyond the first and last frame).
        Returns:
            int or array of int (float for rounding=None)
        '''
        times = self.frame_times
        if len(times) == 0:
            raise ValueError('the video has no frames with known times')
        t = np.asarray(time, dtype=np.float64)
        floor = np.searchsorted(times, t + _TIME_TOLERANCE, side='right') - 1
        if rounding == 'floor':
            index = floor
        elif rounding == 'ceil':
            index = np.searchsorted(times, t - _TIME_TOLERANCE, side='left')
        elif rounding == 'round':
            lo, hi = np.clip(floor, 0, len(times) - 1), np.clip(floor + 1, 0, len(times) - 1)
            index = np.where(np.abs(t - times[hi]) <= np.abs(t - times[lo]), hi, lo)
            if len(times) > 1: # (outside the frames by more than half a frame interval: no nearest frame)
                index = np.where(t < times[0] - (times[1] - times[0]) / 2 - _TIME_TOLERANCE, -1, index)
                index = np.where(t >= times[-1] + (times[-1] - times[-2]) / 2 - _TIME_TOLERANCE, len(times), index)
        elif rounding is None:
            if len(times) < 2:
                return _scalar_or_array(np.zeros_like(t) if len(times) else np.full_like(t, np.nan))
            k = np.clip(floor, 0, len(times) - 2)
            index = k + (t - times[k]) / (times[k + 1] - times[k])
        else:
            raise ValueError(f"rounding must be 'floor', 'ceil', 'round' or None, not {rounding!r}")
        return _scalar_or_array(index)

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
        if self._rewind_reopens:
            self._open()
        else:
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
            elif (seek_ts is not None and not frame_obj.key_frame
                  and not (frame_obj.pict_type == av.video.frame.PictureType.I and self._landing_packet and self._landing_packet[1])
                  and not (self._landing_packet and self._landing_packet[2])): # (edit lists may discard the key frame)
                if self._landing_packet and not self._landing_packet[1]:
                    continue # the seek landed on a non-key frame (e.g. mpeg-ts): seek further back
                problem = 'key frames in the file do not start a decodable picture (e.g. intra refresh)'
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
        except (StopIteration, IndexError) as e:
            if self._frame_pts is None and self._nframes_counted is not None and self._next_idx < self._nframes_counted:
                self._nframes_counted = self._next_idx # frame count without index: the video ended earlier
            if isinstance(e, IndexError):
                raise
            self.rewind()
            raise IndexError(f'frame index {frame_idx} out of range') from None

    def _read_frame(self, frame_idx):
        if frame_idx == 0:
            if self._next_idx != 0: # (at the start after rewind(): no need to rewind again)
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
            expected = self.nframes
            try:
                frame = self.read_frame(i)
            except IndexError:
                if expected: # end of video reached early (e.g. damaged file)
                    warnings.warn(f'{self.filename}: video ended after {i} frames, expected {expected}.', stacklevel=2)
                return
            yield frame
            i += 1

    def __next__(self):
        ''' Read the next frame (same as read()). '''
        return self.read()

    @property
    def frame_shape(self):
        ''' Return the shape of the video frames. '''
        return self.stream.codec_context.height, self.stream.codec_context.width

    @property
    def nframes(self):
        ''' Return the number of frames in the video. '''
        if self._frame_pts is not None:
            return len(self._frame_pts)
        if self._nframes_counted is not None:
            return self._nframes_counted
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
    def static_shape(filename, stream=None):
        ''' Get the shape of a video (static method). 
        
        Args:
            filename (str): path to the video file
            stream (int or str): video stream (stream index or handler_name). Default is None (the first video stream).'''
        with av.open(filename) as container:
            stream = _find_stream(container, stream, 'video')
            shape = np.array([stream.frames, stream.codec_context.height, stream.codec_context.width])
        return shape


class AudioReader(_TagReader):
    ''' A class to read audio from any file FFmpeg can open (e.g. mp4/m4a, mkv, wav, flac, mp3), using PyAV.

    Samples are returned as float32 arrays of shape (channels, samples), scaled to [-1, 1]. Sample index i is the i-th
    sample of a sequential decode from the start of the stream (no samples skipped or added), and `reader[a:b]`
    returns the same samples as reading sequentially. Fast random access uses the time stamps of the file; before the
    first random access, they are checked for gaps and overlaps (for codecs with a fixed frame size: using the container
    index if complete, else the packet time stamps; else by decoding the stream once). Files with gaps, overlaps or unusable time stamps (also flac and vorbis in ogg files,
    whose time stamps are unreliable after seeking) are read by counting decoded samples instead, with a warning:
    still correct, but jumping backwards decodes from the start of the file.

    Random access starts decoding a little before the requested position and discards that run-in: uncompressed and
    lossless audio (e.g. wav, flac) and mp3 are bit-exact. Some lossy decoders synthesize noise or dither, so samples
    after a seek can differ from a sequential decode: aac with perceptual noise substitution (used by FFmpeg's aac
    encoder by default, but not by VideoWriter) replaces noise-like bands with random noise of the same energy, so
    differences can be as large as that noise; opus differs by up to about 1e-4, ac3 by up to about 0.03.

    Args:
        filename (str or file-like): path to the file
        stream (int or str): audio stream to read: its stream index (as in list_tags()), or its handler_name tag.
            Default is None (the first audio stream).

    Example:
        >>> with AudioReader('/path/to/video.mp4') as audio:
        ...     everything = audio.read()              # shape (channels, samples)
        ...     snippet = audio[48000:96000]           # by sample index
        ...     during = audio.read_time(1.0, 1.04)    # by time (same clock as VideoReader.frame_times)
    '''
    RUN_IN = 0.25  # seconds decoded (and discarded) before a random access position

    def __init__(self, filename, stream=None):
        self.filename = filename
        self._stream_spec = stream
        self._open()
        self.sample_rate = self.stream.codec_context.sample_rate
        self.channels = self.stream.codec_context.layout.nb_channels
        self.seekable = True  # False: time stamps are missing or irregular, samples are decoded from the start
        self._ticks = float(self.stream.time_base * self.sample_rate)  # samples per time stamp tick
        self._offset = 0  # sample position = round(pts * ticks) + offset
        self._frame_size = None  # samples per frame (of the second frame), if determined
        self._grid = None  # coarse time stamps (e.g. 1 ms in mkv): (position, frame size, pts * ticks) of a reference frame
        self._start_time = 0.0
        self._nsamples = None
        self._cursor = None  # (sample position, frame generator, pending (position, samples)) after the last read
        self._checked = False  # whether the time stamps were checked for gaps (before the first random access)
        self._table = None  # (first sample, time, samples) of each decoded frame, for streams with gaps (see _frame_table)
        self._calibrate()

    def _open(self):
        if hasattr(self, 'container'):
            self.container.close()
        if hasattr(self.filename, 'seek'): # file-like object: (re)open from its start
            self.filename.seek(0)
        self.container = av.open(self.filename)
        self.stream = _find_stream(self.container, self._stream_spec, 'audio')

    def _rewind(self):
        self._open() # (seeking to 0 can skip the first frame, e.g. opus in mkv)

    def _decode(self):
        ''' Decode from the current position. Undecodable damaged packets (e.g. the last one of a cut-off recording)
        are skipped, with a warning, in every way of reading (so that sample counts agree). '''
        for packet in self.container.demux(self.stream):
            try:
                frames = packet.decode()
            except av.InvalidDataError:
                if not packet.is_corrupt:
                    raise
                if not getattr(self, '_warned_damaged', False):
                    self._warned_damaged = True
                    warnings.warn(f'{self._name}: undecodable damaged audio packet skipped (e.g. a cut-off recording)',
                                  stacklevel=3)
                continue
            for frame in frames:
                yield frame

    @property
    def _name(self):
        return self.filename if isinstance(self.filename, (str, os.PathLike)) else 'file-like object'

    def _frames(self, from_start=False):
        ''' Decode from the current position: yields (sample position from the time stamp, or None; samples). When
        decoding from the start, the first frame is at position 0 (its time stamp can precede a shortened first frame). '''
        for frame in self._decode():
            if from_start:
                pos, from_start = 0, False
            else:
                pos = None if frame.pts is None else self._pos(frame.pts)
            yield pos, _audio_frame_to_float(frame)

    def _pos(self, pts):
        ''' Sample position of a frame from its time stamp. Coarse time stamps are snapped to the frame grid. '''
        raw = pts * self._ticks
        if self._grid is None:
            return round(raw) + self._offset
        ref_pos, size, ref_raw = self._grid
        return ref_pos + round((raw - ref_raw) / size) * size

    def _on_time(self, pts, counted):
        ''' Whether a frame with this time stamp starts after exactly `counted` samples (for coarse time stamps: within
        their rounding, so that also gaps smaller than half a frame are found). '''
        if self._pos(pts) != counted:
            return False
        if self._grid is None:
            return True
        ref_pos, _, ref_raw = self._grid
        return abs(pts * self._ticks - (ref_raw + counted - ref_pos)) <= 1.5 * self._ticks

    def _calibrate(self, n_frames=8):
        ''' Align time stamp positions with sample counts of a sequential decode: the first frame can be shortened
        (e.g. encoder priming, opus pre-skip) without its time stamp being adjusted. Frames after the first one must be
        contiguous. Time stamps coarser than one sample (e.g. 1 ms in mkv) are snapped to the grid of frame starts,
        which needs a constant frame size. '''
        self._rewind()
        frames = [(frame.pts, frame.samples) for _, frame in zip(range(n_frames), self._decode())]
        self._rewind()
        if not frames:
            self._nsamples = 0
            return
        if any(pts is None for pts, _ in frames):
            return self._set_unseekable('missing time stamps')
        ref = min(1, len(frames) - 1) # the first frame may be shortened: calibrate on the second one
        self._frame_size = frames[ref][1] if ref else None
        counted = [sum(n for _, n in frames[:i]) for i in range(len(frames))]
        raw = [pts * self._ticks for pts, _ in frames]
        tb = float(self.stream.time_base)
        starts = [pts * tb - counted[i] / self.sample_rate for i, (pts, _) in enumerate(frames)][ref:]
        self._start_time = starts[0]
        if self._ticks <= 1: # time stamps resolve single samples
            self._offset = counted[ref] - round(raw[ref])
        else:
            size = frames[ref][1]
            if any(n != size for _, n in frames[ref:-1]):
                return self._set_unseekable('coarse time stamps and varying frame sizes')
            self._grid = (counted[ref], size, raw[ref])
            if ref and frames[0][1] == size: # (a first frame of full size is not shortened: its time stamp counts too)
                starts.append(frames[0][0] * tb)
            lo, hi = max(starts) - tb / 2, min(starts) + tb / 2 # (time stamps are rounded to the nearest tick)
            self._start_time = (lo + hi) / 2 if lo <= hi else float(np.mean(starts))
        if any(not self._on_time(frames[i][0], counted[i]) for i in range(ref, len(frames))):
            return self._set_unseekable('irregular time stamps')
        if abs(self._pos(frames[0][0])) >= frames[ref][1]: # more than a shortened first frame
            return self._set_unseekable('gap or overlap after the first frame')

    def _check_time_stamps(self):
        ''' Check (once, before the first random access) that the time stamps have no gaps or overlaps, so that
        positions from time stamps equal sample counts. If the codec guarantees a fixed frame size (aac, mp3, alac,
        flac with a fixed block size; confirmed by the decoded frames), every frame but the last must start exactly one
        frame size after the one before: checked in the container index if it is complete (e.g. mp4, also fragmented or
        cut off: see _complete_index; at most the end of the file is read), else in the time stamps of the packets
        (e.g. mkv: the file is read, but not decoded). A gap or overlap moves the time stamps of the following frames. Otherwise, the audio is decoded once
        and each frame's position is compared with the number of samples before it. '''
        if self._checked or not self.seekable:
            return self.seekable
        self._checked = True
        if self.container.format.name == 'ogg' and self.stream.codec_context.name != 'opus':
            # (after seeking, FFmpeg can assign wrong time stamps to the first packets of an ogg page, e.g. for flac and
            # vorbis, whose packet durations it cannot know from the packets alone)
            self._set_unseekable('time stamps of this codec in ogg files are unreliable after seeking')
            return False
        tolerance = 1.5 * max(self._ticks, 1) # (coarse time stamps are rounded)
        size = _fixed_frame_size(self.stream.codec_context)
        fixed = bool(size) and size == self._frame_size # (guaranteed by the codec, and confirmed by decoding)
        ok = None
        if fixed:
            starts = self._complete_index()
            if starts is not None: # (e.g. mp4, also fragmented)
                ok = _on_grid(starts, self._ticks, size, tolerance)
            else: # (e.g. mkv: reads the file, without decoding)
                ok = self._check_packet_time_stamps(size, tolerance)
        if ok is None: # (no fixed frame size, or packets without time stamps)
            ok, counted = True, 0
            self._rewind()
            for i, frame in enumerate(self._decode()):
                if i > 0 and (frame.pts is None or not self._on_time(frame.pts, counted)):
                    ok = False
                    break
                counted += frame.samples
            self._rewind()
            self._cursor = None
            if ok:
                self._nsamples = counted
        if not ok:
            self._set_unseekable('gaps or overlaps in the time stamps')
        return ok

    def _complete_index(self):
        ''' Time stamps of all packets from the container index, if it is known to be complete (else None). With a frame
        count in the header (e.g. mp4), the index must have one entry per frame. Without one (e.g. fragmented mp4, whose
        frame tables FFmpeg reads on opening, unless it reads only some of them, e.g. with a global sidx), entries
        beyond the end of the file are dropped (cut-off recordings), and the packets at the end of the file must be
        exactly the last entries (only the end of the file is read; see _index_tail_matches). '''
        entries = self.stream.index_entries
        n = self.stream.frames
        if n:
            return [e.timestamp for e in entries] if len(entries) == n else None
        file_size = getattr(self.container, 'size', None)
        rows = [(e.timestamp, e.pos) for e in entries
                if e.pos >= 0 and (not file_size or file_size <= 0 or e.pos + e.size <= file_size)]
        if not rows or not self._index_tail_matches(rows):
            return None
        return [ts for ts, _ in rows]

    def _index_tail_matches(self, rows, n_last=8):
        ''' Whether the packets from the n_last-th last index entry to the end of the file are exactly the last entries
        (by file position, in order), followed by nothing but damaged packets (cut-off recordings). An index that misses
        packets (partly loaded, or e.g. key frames only) does not match. '''
        k = max(len(rows) - n_last, 0)
        index_of = {pos: i for i, (_, pos) in enumerate(rows)}
        expected = None # index entry of the next packet
        try:
            self.container.seek(int(rows[k][0]), backward=True, stream=self.stream)
            for packet in self.container.demux(self.stream):
                if not packet.size:
                    continue
                if expected is None: # first packet after the seek: must be in the index, at or before entry k
                    expected = index_of.get(packet.pos)
                    if expected is None or expected > k:
                        return False
                if expected < len(rows):
                    if packet.pos != rows[expected][1]:
                        return False
                    expected += 1
                elif not packet.is_corrupt: # more packets than the index lists
                    return False
        except av.FFmpegError:
            return False
        finally:
            self._rewind()
            self._cursor = None
        return expected == len(rows)

    def _check_packet_time_stamps(self, size, tolerance):
        ''' Whether the packet time stamps lie on the grid of one frame size (see _on_grid), read without decoding.
        None if packets have no time stamps. '''
        self._rewind()
        try:
            pts = []
            for packet in self.container.demux(self.stream):
                if packet.size:
                    if packet.pts is None:
                        return None
                    pts.append(packet.pts)
            return _on_grid(pts, self._ticks, size, tolerance)
        finally:
            self._rewind()
            self._cursor = None

    def _set_unseekable(self, reason):
        if self.seekable:
            self.seekable = False
            warnings.warn(f'{self._name}: fast random access is not possible for this audio stream ({reason}). '
                          'Samples are counted while decoding instead: correct, but jumping backwards decodes from the '
                          'start of the file, which can be slow.', stacklevel=3)

    @property
    def nsamples(self):
        ''' Number of samples per channel (exact: determined by decoding the end of the stream, or, if needed, the whole
        stream once). '''
        if self._nsamples is None:
            self._nsamples = self._count_samples()
        return self._nsamples

    def _count_samples(self):
        if self._check_time_stamps() and self._nsamples is not None: # (counted while checking)
            return self._nsamples
        if self.stream.duration and self.seekable:
            end = (self.stream.start_time or 0) + self.stream.duration
            run_in = int(self.RUN_IN / self.stream.time_base)
            try:
                self.container.seek(int(end - run_in), backward=True, stream=self.stream)
                last = None
                for pos, data in self._frames():
                    if pos is None:
                        last = None
                        break
                    last = pos + data.shape[1]
                if last is not None:
                    return last
            except av.FFmpegError:
                pass
            finally:
                self._cursor = None
                self._rewind()
        self._rewind() # count by decoding everything
        total = sum(data.shape[1] for _, data in self._frames())
        self._rewind()
        return total

    @property
    def duration(self):
        ''' Duration in seconds. '''
        return self.nsamples / self.sample_rate

    @property
    def start_time(self):
        ''' Presentation time of sample 0 in seconds (same clock as VideoReader.frame_times). '''
        return self._start_time

    def read(self, start=0, stop=None):
        ''' Read samples [start, stop) (all samples by default). Returns a float32 array of shape (channels, samples). '''
        n = self.nsamples
        start, stop, _ = slice(start, stop).indices(n)
        out = np.zeros((self.channels, max(stop - start, 0)), dtype=np.float32)
        if stop <= start:
            return out
        expected = None
        cursor = self._cursor
        if cursor is not None and cursor[0] <= start and (cursor[0] == start or not self.seekable
                                                         or start - cursor[0] <= self.RUN_IN * self.sample_rate):
            frames, pending = cursor[1], cursor[2] # continue where the last read ended (also for small forward jumps)
            expected = cursor[0] if pending is None else None
        elif self._check_time_stamps():
            frames, pending = self._seek(start), None
            if frames is None: # time stamps turned out to be irregular
                return self.read(start, stop)
        else:
            self._rewind()
            frames, pending = self._counted(self._frames()), None
        filled = 0
        source = _chain_pending(pending, frames)
        for pos, data in source:
            if expected is not None and pos != expected:
                self._set_unseekable('irregular time stamps')
                self._cursor = None
                return self.read(start, stop)
            expected = pos + data.shape[1]
            a, b = max(start, pos), min(stop, expected)
            if b > a:
                out[:, a - start:b - start] = data[:, a - pos:b - pos]
                filled = b - start
            if expected >= stop:
                self._cursor = (stop, frames, (pos, data) if expected > stop else None)
                return out
            filled = max(filled, min(stop, expected) - start)
        self._cursor = None
        return out[:, :filled]

    def _counted(self, frames):
        ''' Positions by counting samples (for streams with unreliable time stamps; decoding from the start). '''
        pos = 0
        for _, data in frames:
            yield pos, data
            pos += data.shape[1]

    def _seek(self, start):
        ''' Position the decoder so that decoding continues with a frame containing sample start (after a run-in).
        Returns the frame generator, starting with that frame, or None if time stamps turned out to be irregular.
        After a seek, the first decoded frames can be shortened without adjusted time stamps (e.g. opus pre-roll), so
        positions are trusted from the first frame that follows its predecessor without a gap. '''
        self._cursor = None
        run_in = int(self.RUN_IN * self.sample_rate)
        for target in [start - run_in, start - 4 * run_in, None]:
            from_start = target is None or target <= 0
            if from_start:
                self._rewind()
            else:
                ts = int((self._start_time + target / self.sample_rate) / self.stream.time_base)
                try:
                    self.container.seek(ts, backward=True, stream=self.stream)
                except av.FFmpegError:
                    continue
            frames = self._frames(from_start=from_start)
            synced = from_start # (from the start, the first frame is at position 0)
            prev = current = None
            for item in frames:
                if item[0] is None:
                    break
                if not synced:
                    synced = prev is not None and prev[0] + prev[1].shape[1] == item[0]
                    prev = item
                    if not synced:
                        continue
                if current is None:
                    if item[0] > start: # landed after the requested sample: seek further back
                        break
                elif item[0] != current[0] + current[1].shape[1]:
                    self._set_unseekable('irregular time stamps')
                    return None
                current = item
                if current[0] + current[1].shape[1] > start:
                    return _chain_pending(current, frames)
            else:
                if current is not None: # end of the stream before start
                    return iter(())
        return None

    def read_time(self, start_time, stop_time):
        ''' Read the samples from start_time to stop_time, in seconds on the same clock as VideoReader.frame_times:
        samples sample_index(start_time) to sample_index(stop_time) (exclusive), each rounded to the nearest sample.
        Consecutive time windows (e.g. of consecutive video frames) thus split the audio without gaps or overlaps. '''
        return self.read(self.sample_index(start_time), self.sample_index(stop_time))

    def sample_time(self, index):
        ''' Presentation time (in seconds, on the same clock as VideoReader.frame_times) of a sample index (an integer
        or an array of integers, from 0 to nsamples; nsamples gives the time just after the last sample). Without gaps
        in the time stamps, sample i is at start_time + i / sample_rate. For streams with gaps or overlaps, times come
        from the time stamp of the decoded frame that contains the sample (the stream is decoded once to find them). '''
        i = np.asarray(index)
        if i.dtype.kind not in 'iu':
            raise TypeError(f'sample indices must be integers, not {i.dtype}')
        if np.any((i < 0) | (i > self.nsamples)):
            raise IndexError(f'sample indices must be from 0 to {self.nsamples}')
        if self._check_time_stamps():
            t = self._start_time + i / self.sample_rate
        else:
            starts, times, _ = self._frame_table()
            k = np.clip(np.searchsorted(starts, i, side='right') - 1, 0, None)
            t = times[k] + (i - starts[k]) / self.sample_rate
        return _scalar_or_array(t)

    def sample_index(self, time, rounding='round'):
        ''' Sample index at a time (in seconds, on the same clock as VideoReader.frame_times; a number or an array of
        times). The inverse of sample_time: sample_index(sample_time(i)) == i (except where time stamps overlap).

        Args:
            time (float or array): time(s) in seconds
            rounding: 'round' (default): the sample with the nearest time. 'floor': the last sample at or before
                that time (the sample playing at that time). 'ceil': the first sample at or after that time. Integer
                results are clipped to 0 ... nsamples, and times in a gap of the time stamps give the first sample
                after the gap. None: a fractional index (not clipped; extrapolated before the first and after the last
                sample).
        Returns:
            int or array of int (float for rounding=None)
        '''
        if rounding not in ('ceil', 'floor', 'round', None):
            raise ValueError(f"rounding must be 'ceil', 'floor', 'round' or None, not {rounding!r}")
        t = np.asarray(time, dtype=np.float64)
        rate = self.sample_rate
        if self._check_time_stamps():
            index = (t - self._start_time) * rate
        else:
            starts, times, sizes = self._frame_table()
            k = np.clip(np.searchsorted(np.maximum.accumulate(times), t + _TIME_TOLERANCE, side='right') - 1, 0, None)
            lo = np.where(k == 0, -np.inf, 0) # (in a gap: the end of the frame before it, i.e. the next sample)
            hi = np.where(k == len(starts) - 1, np.inf, sizes[k])
            index = starts[k] + np.clip((t - times[k]) * rate, lo, hi)
        if rounding is None:
            return _scalar_or_array(index)
        tolerance = _TIME_TOLERANCE * rate
        if rounding == 'ceil':
            index = np.ceil(index - tolerance)
        elif rounding == 'floor':
            index = np.floor(index + tolerance)
        else:
            index = np.floor(index + 0.5)
        return _scalar_or_array(np.clip(index, 0, self.nsamples).astype(np.int64))

    def _frame_table(self):
        ''' First sample (counted), presentation time and number of samples of each decoded frame (decodes the stream
        once). Frames without a time stamp follow the previous frame; a shortened first frame (e.g. encoder priming)
        starts at start_time. '''
        if self._table is None:
            starts, times, sizes = [], [], []
            pos, next_time = 0, self._start_time
            tb, rate = float(self.stream.time_base), self.sample_rate
            self._rewind()
            for frame in self._decode():
                t = next_time if frame.pts is None else frame.pts * tb
                if not starts and abs(t - self._start_time) * rate < (self._frame_size or frame.samples):
                    t = self._start_time # (time stamp of the first frame before shortening)
                starts.append(pos); times.append(t); sizes.append(frame.samples)
                pos += frame.samples
                next_time = t + frame.samples / rate
            self._rewind()
            self._cursor = None
            self._nsamples = pos
            self._table = (np.array(starts, dtype=np.int64), np.array(times, dtype=np.float64),
                           np.array(sizes, dtype=np.int64))
        return self._table

    def __getitem__(self, index):
        if isinstance(index, (int, np.integer)):
            i = int(index) + (self.nsamples if index < 0 else 0)
            if not 0 <= i < self.nsamples:
                raise IndexError(f'sample index {index} out of range for {self.nsamples} samples')
            return self.read(i, i + 1)[:, 0]
        if isinstance(index, slice):
            start, stop, step = index.indices(self.nsamples)
            if step == 1:
                return self.read(start, stop)
            lo, hi = (start, stop) if step > 0 else (stop + 1, start + 1)
            return self.read(lo, hi)[:, start - lo::step] if hi > lo else np.zeros((self.channels, 0), np.float32)
        raise TypeError(f'audio samples can be indexed with integers or slices, not {index!r}')

    def __len__(self):
        return self.nsamples

    def close(self):
        if hasattr(self, 'container'):
            self.container.close()

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False


_TIME_TOLERANCE = 1e-9 # seconds: times that differ by less (e.g. float rounding) are taken to be equal


def _scalar_or_array(values):
    ''' A Python scalar for 0-d results (int or float), else the array. '''
    values = np.asarray(values)
    return values.item() if values.ndim == 0 else values


def _on_grid(timestamps, ticks, size, tolerance):
    ''' Whether frames with these start time stamps (in ticks of `ticks` samples) follow each other without gaps or
    overlaps: every frame after the first starts a whole number of frame sizes after the second one (within tolerance,
    in samples). The first frame is skipped (it can be shortened, e.g. by encoder priming; see AudioReader._calibrate).
    Compared with the grid, not with the previous frame, so that rounding errors of coarse time stamps cannot add up. '''
    starts = np.asarray(timestamps[1:], dtype=np.float64) * ticks
    return bool(np.all(np.abs(starts - starts[0] - np.arange(len(starts)) * size) <= tolerance)) if len(starts) else True


def _fixed_frame_size(codec_context):
    ''' Samples per frame (all frames but the last), if the codec guarantees a fixed frame size, else None. '''
    if codec_context.frame_size: # (codecs with a fixed frame size, e.g. aac, mp3, opus)
        return codec_context.frame_size
    extradata = codec_context.extradata or b''
    if codec_context.name == 'alac' and len(extradata) >= 16 and extradata[4:8] == b'alac':
        return int.from_bytes(extradata[12:16], 'big') or None # (frameLength of the ALAC magic cookie)
    if codec_context.name == 'flac':
        if extradata[:4] == b'fLaC':
            extradata = extradata[8:]
        if len(extradata) >= 34: # STREAMINFO: minimum and maximum block size (equal: fixed block size)
            min_size, max_size = int.from_bytes(extradata[0:2], 'big'), int.from_bytes(extradata[2:4], 'big')
            return min_size if min_size == max_size else None
    return None


def _chain_pending(pending, frames):
    ''' Yield pending (if not None), then the items of frames. (No "yield from": closing this generator would close
    frames too, which is kept to continue reading.) '''
    if pending is not None:
        yield pending
    for item in frames:
        yield item


def _audio_frame_to_float(frame):
    ''' Samples of an audio frame as float32, shape (channels, samples), scaled to [-1, 1]. '''
    data = frame.to_ndarray()
    if not frame.format.is_planar: # interleaved: shape (1, samples * channels)
        data = data.reshape(-1, frame.layout.nb_channels).T
    if data.dtype == np.uint8:
        return (data.astype(np.float32) - 128) / 128
    if data.dtype.kind == 'i':
        return data.astype(np.float32) / float(2 ** (8 * data.dtype.itemsize - 1))
    return data.astype(np.float32)


def _find_stream(container, stream=None, kind=None):
    ''' A stream of the container: the first stream of the given kind ('video', 'audio'; stream=None), the stream with
    this index (int, as in list_tags()), or the stream whose handler_name tag is this string. '''
    candidates = [s for s in container.streams if kind is None or s.type == kind]
    if stream is None:
        if not candidates:
            raise ValueError(f'the file has no {kind} stream')
        return candidates[0]
    if isinstance(stream, (int, np.integer)) and not isinstance(stream, bool):
        if not 0 <= int(stream) < len(container.streams):
            raise IndexError(f'no stream {int(stream)}: the file has {len(container.streams)} streams')
        found = container.streams[int(stream)]
        if kind is not None and found.type != kind:
            raise ValueError(f'stream {int(stream)} is a {found.type} stream, not a {kind} stream')
        return found
    for s in candidates:
        if _get_tag(s.metadata, 'handler_name') == stream:
            return s
    raise KeyError(f"no {kind + ' ' if kind else ''}stream with handler_name {stream!r}")


def _get_tag(tags, name):
    ''' Tag value by name: exact match, else case-insensitive match. None if missing. '''
    if name in tags:
        return tags[name]
    matches = [value for key, value in tags.items() if key.lower() == name.lower()]
    return matches[0] if matches else None


def warn_transcode(msg):
    warnings.warn(msg, stacklevel=2)
    warnings.warn('Consider transcoding (ffmpeg -y -i "input.mp4" -c:v libx264 -pix_fmt yuv420p -preset superfast -crf 23 "output.mp4").', stacklevel=2)


def _fps_fraction(fps):
    """ Frame rate as int or Fraction. Floats are converted to the simplest fraction that matches them closely
    (e.g. 29.97 -> 2997/100, 30000/1001 -> 30000/1001). """
    if isinstance(fps, (float, np.floating)):
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError(f'fps must be a positive number, got {fps}.')
        frac = Fraction(float(fps)).limit_denominator(1001)
        if frac == 0 or abs(frac - fps) > 1e-6 * fps: # e.g. very low frame rates: use more precision
            frac = Fraction(float(fps)).limit_denominator(10**6)
        return frac
    if fps <= 0:
        raise ValueError(f'fps must be a positive number, got {fps}.')
    return fps


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
            is created by the first call to `write_audio`). Streams cannot be added once data was written: to write
            audio and video, pass audio_rate (if video comes first) or frame_shape (if audio comes first).
        audio_channels (int): number of audio channels (1 or 2) if audio_rate is given. Default is 1.
        **kwargs: additional keyword arguments to pass to the codec. Check e.g. `ffmpeg -h encoder=libx264`
    """

    def __init__(self, filename, codec='libx264', fps=25, frame_shape=None, pix_fmt='yuv420p', audio_rate=None, audio_channels=1, **kwargs):
        self.filename = filename
        self.container = av.open(filename, mode='w')
        self._closed = False
        self._n_muxed = 0
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
            self.fps = _fps_fraction(fps)
            self.pix_fmt = pix_fmt
            self.kwargs = {k: str(v) for k, v in kwargs.items() if v is not None} # codec options must be strings
            #add video stream, if frame_shape is provided (otherwise create when the first frame is written)
            if frame_shape is not None:
                self._add_video_stream(frame_shape[0], frame_shape[1])
            if audio_rate is not None:
                self._add_audio_stream(audio_rate, audio_channels)
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

    def _add_audio_stream(self, audio_rate, channels, stream_args=None):
        if channels not in (1, 2):
            raise ValueError(f'Only mono and stereo audio are supported, not {channels} channels.')
        if self._n_muxed:
            raise ValueError('Cannot add an audio stream after video was written: pass audio_rate (and audio_channels) '
                             'to VideoWriter().')
        self.channel_layout = 'mono' if channels == 1 else 'stereo'
        if stream_args is None:
            # (no perceptual noise substitution: the decoder would synthesize random noise, which differs between a
            # sequential decode and a decode after seeking)
            stream_args = dict(codec_name='aac', rate=audio_rate, layout=self.channel_layout, options={'aac_pns': '0'})
        self.audio_stream = self.container.add_stream(**stream_args)
        self.audio_rate = audio_rate
        self.audio_channels = channels
        self.audio_pts = 0

    def _mux_packets(self, packets):
        for packet in packets:
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
            if self._n_muxed:
                raise ValueError('Cannot add the video stream after audio was written: pass frame_shape to VideoWriter().')
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
            if self._n_muxed == 0 and not getattr(self, '_exiting_on_error', False):
                warnings.warn(f'{self.filename}: VideoWriter closed without any frames, no file was written.', stacklevel=2)
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
        if exc_type is None:
            self.close()
            return False
        try: # exiting with an exception: do not replace it with an error from closing the file
            self.close()
        except Exception as e:
            warnings.warn(f'{self.filename}: error while closing the video after an exception: {e!r}', stacklevel=2)
        return False
    
    @_classproperty
    def codecs_available(cls):
        """ Return a list of available codecs. """
        return sorted(av.codec.codecs_available)
