
__all__ = ('read_logfile', 'get_peaks', 'bin_peaks', 'get_filter')

import tables as tb
import numpy as np
import math
from scipy import signal


def read_logfile(filename, raw=False):
    fh = tb.open_file(filename)
    pressure = fh.root.raw_data.pressure
    freq = pressure._v_attrs.frequency
    scale = pressure._v_attrs.scale
    offset = pressure._v_attrs.offset
    bit_depth = pressure._v_attrs.bit_depth
    a = list(pressure.data)
    adc_data = np.array(a, dtype=np.uint32)
    if not raw:
        adc_data = adc_data / float(2 ** bit_depth) * scale - offset
    adc_idxs = np.array(pressure.ts_idx)
    adc_ts = np.array(pressure.ts)
    ts0 = adc_ts[0] - adc_idxs[0] / freq
    adc_ts = adc_ts - ts0
    odor_groups = fh.root.raw_data.odors._f_listNodes()
    odors = {}
    for odor in odor_groups:
        try:
            ts, state = odor.ts, odor.state
        except AttributeError:
            continue
        if not len(state):
            continue

        idxs = []
        i = 0
        while i < len(state):
            if state[i]:
                break
            i += 1
        if i == len(state):
            continue

        idxs.append(i)
        cond = state[i]
        while i < len(state):
            if state[i] != cond:
                idxs.append(i)
                cond = state[i]
            i += 1

        timestamps = [ts[i] - ts0 for i in idxs]
        if len(timestamps) % 2:
            timestamps.append(adc_ts[-1] - ts0)
        odors[odor._v_name] = zip(timestamps[::2], timestamps[1::2])

    fh.close()
    return (adc_data, freq, adc_idxs, adc_ts), odors


def get_peaks(data, order=1):
    maxima, = signal.argrelmax(data, order=order)
    minima, = signal.argrelmin(data, order=order)
    while minima[0] >= maxima[0]:
        maxima = maxima[1:]

    i = 1
    while i < min(len(maxima), len(minima)):
        if minima[i] <= maxima[i - 1]:
            minima = np.delete(minima, i)
        elif maxima[i] <= minima[i]:
            maxima = np.delete(maxima, i)
        else:
            i += 1

    if len(maxima) != len(minima):
        assert len(maxima) + 1 == len(minima)
        minima = minima[:-1]

    return minima, maxima


def bin_peaks(peaks, idxs, bin_size, overlap):
    assert len(peaks) == len(idxs)
    shift = max(bin_size - overlap, 1)
    s = -bin_size // 2
    e = bin_size + s
    max_len = math.ceil((idxs[-1] + 1) / float(shift))
    ends = np.zeros((max_len, ), dtype=np.int64)
    starts = np.zeros((max_len, ), dtype=np.int64)
    count = np.zeros((max_len, ), dtype=np.int)
    cum_amp = np.zeros((max_len, ), dtype=np.float64)
    mean_amp = np.zeros((max_len, ), dtype=np.float64)

    idx_e = idx_s = 0
    for i in range(len(ends)):
        while idx_s < len(idxs) and idxs[idx_s] < s:
            idx_s += 1
        starts[i] = idx_s
        while idx_e < len(idxs) and idxs[idx_e] < e:
            idx_e += 1
        ends[i] = idx_e
        s += shift
        e += shift

    for i in range(len(ends)):
        count[i] = ends[i] - starts[i]
        cum_amp[i] = np.sum(peaks[starts[i]:ends[i]])
        if count[i]:
            mean_amp[i] = cum_amp[i] / float(count[i])

    return (np.arange(0, idxs[-1] + 1, shift), count,
            cum_amp / np.max(cum_amp), mean_amp / np.max(mean_amp))


def get_filter(freq, animal_type='mouse'):
    if animal_type == 'mouse':
        return signal.iirfilter(
            3, [2 * 8 / freq, 2 * 13 / freq], btype='band', ftype='butter')
    return signal.iirfilter(
        3, [2 * 4 / freq, 2 * 12 / freq], btype='band', ftype='butter')
