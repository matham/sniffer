
import tables as tb
import numpy as np
import math
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib.mlab import specgram
from collections import defaultdict
import re
from os.path import join, basename, splitext

#__all__ = ('read_logfile', 'get_peaks', 'bin_peaks', 'get_filter')

odor_name_pat = re.compile('(.+)_(.+?)')

'''
G:\Sniffer_data\SP15\images>E:\FFmpeg\dev\ffmpeg-x64\bin\ffmpeg.exe -framerate 7
.5 -i 05-04-15_r22_%03d.png -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" -c:v libx264
 -r 7.5 -pix_fmt yuv420p out.mp4

G:\Sniffer_data\SP15\images>E:\FFmpeg\dev\ffmpeg-x64\bin\ffmpeg.exe -i out.mp4 -
ss 00:07:20 -i ..\05-04-15_r22.mp4 -filter_complex "nullsrc=size=3228x1214 [base
];[0:v] setpts=PTS-STARTPTS, scale=1614x1214 [upper];[1:v] setpts=PTS-STARTPTS,
scale=1614x1214 [lower];[base][upper] overlay=shortest=1 [tmp1];[tmp1][lower] ov
erlay=shortest=1:x=1614" -c:v libx264 merged.mp4

signal.lfilter(b, a, data)
'''


def read_logfile(filename, raw=False):
    fh = tb.open_file(filename)
    pressure = fh.root.raw_data.pressure
    freq = pressure._v_attrs.frequency
    scale = pressure._v_attrs.scale
    offset = pressure._v_attrs.offset
    bit_depth = pressure._v_attrs.bit_depth

    adc_data = np.array(list(pressure.data), dtype=np.uint32)
    if not raw:
        adc_data = adc_data / float(2 ** bit_depth) * scale - offset
    adc_idxs = np.array(pressure.ts_idx)
    adc_ts = np.array(pressure.ts)
    ts0 = adc_ts[0] - adc_idxs[0] / freq
    adc_ts = adc_ts - ts0

    odor_groups = fh.root.raw_data.odors._f_listNodes()
    odors = {}
    odor_ports = {}
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
        name, port = re.match(odor_name_pat, odor._v_name).groups()
        odors[name] = zip(timestamps[::2], timestamps[1::2])
        odor_ports[name] = port

    fh.close()
    return (adc_data, freq, adc_idxs, adc_ts), (odors, odor_ports)


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
    l, h = get_sniffing_freqs(animal_type)
    return signal.iirfilter(
        3, [2 * l / freq, 2 * h / freq], btype='band', ftype='butter')


def generate_stft(data, freq):
    Z, freqs, bins = specgram(data, NFFT=1024, Fs=freq, noverlap=512)
    return Z, freqs, bins


def display_stft(ax, Z, freqs, bins, post_zero=False):
    idxs = Z == 0
    Z = 10. * np.log10(Z)
    if post_zero:
        Z[idxs] = 0
    ax.imshow(Z, interpolation=None, aspect='auto',
              origin='lower', extent=(bins[0], bins[-1], freqs[0], freqs[-1]))


def plot_odors(ax, odors, low, high):
    for name, times in odors.items():
        for s, e in times:
            ax.plot([s, s], [low, high], '-k')
            ax.plot([e, e], [low, high], '-k')
            ax.text(s, high, name)


def extract_sniffing_data(data, freqs, animal_type):
    l, h = get_sniffing_idxs(freqs, animal_type=animal_type)
    return data[l:h, :], freqs[l:h]


def get_sniffing_freqs(animal_type):
    if animal_type == 'mouse':
        return 8, 13
    return 4, 12


def get_sniffing_idxs(freqs, animal_type):
    l, h = get_sniffing_freqs(animal_type)
    return sum(freqs < l), sum(freqs < h)


def compute_sniffing_power(
        data, freqs, animal_type, normalize=False, use_max=False):
    l, h = get_sniffing_freqs(animal_type)
    if use_max:
        sum_power = np.max(data[sum(freqs < l):sum(freqs < h), :], axis=0)
    else:
        sum_power = np.sum(data[sum(freqs < l):sum(freqs < h), :], axis=0)
    if normalize:
        sum_power /= np.percentile(sum_power, 95.)
    return sum_power


def get_normalizing_idxs(freqs):
    return sum(freqs < 25), sum(freqs < 35)


def limit_tails(
    data, freqs, upper_percentile=99.99, lower_percentile=None, max_val=None,
        min_val='auto', set_zero=True):
    if upper_percentile is not None:
        max_val = np.percentile(data, upper_percentile)
    if max_val is not None:
        data[data > max_val] = max_val

    if lower_percentile is not None:
        min_val = np.percentile(data, lower_percentile)
    if min_val is not None:
        if min_val == 'auto':
            l, h = get_normalizing_idxs(freqs)
            min_val = np.percentile(data[l:h, :], 99.99)
        if set_zero:
            data[data < min_val] = 0.
        else:
            data[data < min_val] = min_val
    return data, min_val, max_val


def reject_spectro_noise(data, threshold):
    flattened = data.reshape(data.shape[0] * data.shape[1])
    flattened = flattened[flattened != 0]
    t = np.percentile(flattened, threshold)
    data[data < t] = 0.


def generate_images(filename, output_dir, ts, te, win, framerate):
    jump = framerate / win
    plt.ion()
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = fig_size[0] * 2
    fig_size[1] = fig_size[1] * 2
    plt.rcParams["figure.figsize"] = fig_size
    (data, freq, _, _), _ = read_logfile(filename)
    b, a = get_filter(freq, animal_type='rat')
    data_filt = signal.lfilter(b, a, data)
    line,  = plt.plot([0, win], [-.75, .75])
    W = np.round(win / jump)
    i = 0
    while ts + i * jump < te:
        whole = int(i / W)
        rem = i % W
        s, e = int((ts + whole * win) * freq), int((ts + whole * win + (rem + 1) * jump) * freq)
        line.set_ydata(data_filt[s: e])
        line.set_xdata(np.arange(s, e) / freq)
        plt.xlim((s / freq, s / freq + win))
        if not i:
            plt.tight_layout()
        plt.savefig(join(output_dir, splitext(basename(filename))[0] + '_{:03d}.png'.format(i)))
        i += 1


def apply_trial_func(data, ts, odors, func, soffset=0, eoffset=0, **kwargs):
    res = defaultdict(list)
    for name, times in odors.items():
        for i, (s, e) in enumerate(times):
            s += soffset
            e += eoffset
            if len(times) == 1:
                trial = name
            else:
                trial = '{}_x{}'.format(name, i)

            if len(data.shape) == 1:
                d = data[np.sum(ts < s):np.sum(ts < e) + 1]
            elif data.shape[0] == 1:
                d = data[0, np.sum(ts < s):np.sum(ts < e) + 1]
            else:
                d = data[:, np.sum(ts < s):np.sum(ts < e) + 1]

            res[trial].append(func(d, **kwargs))

    return res
