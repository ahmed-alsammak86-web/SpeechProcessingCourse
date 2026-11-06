import torch


def find_words(waveform, sample_rate, threshold_ratio=0.1, min_gap=0.1, min_len=0.1):
    signal = waveform[0]
    frame = int(0.025 * sample_rate)
    hop = int(0.010 * sample_rate)

    frames = signal.unfold(0, frame, hop)
    energy = frames.pow(2).mean(dim=1).sqrt()
    active = (energy > threshold_ratio * energy.max()).tolist()

    segments = []
    start = None
    for i, is_speech in enumerate(active):
        if is_speech and start is None:
            start = i
        elif not is_speech and start is not None:
            segments.append([start * hop / sample_rate, (i * hop + frame) / sample_rate])
            start = None
    if start is not None:
        segments.append([start * hop / sample_rate, len(signal) / sample_rate])

    merged = []
    for s, e in segments:
        if merged and s - merged[-1][1] < min_gap:
            merged[-1][1] = e
        else:
            merged.append([s, e])

    return [seg for seg in merged if seg[1] - seg[0] >= min_len]


def find_8_words(waveform, sr):
    for gap in [0.1, 0.05, 0.02]:
        for ratio in [0.1, 0.07, 0.05, 0.15, 0.2, 0.03]:
            segs = find_words(waveform, sr, threshold_ratio=ratio, min_gap=gap)
            if len(segs) == 8:
                return segs
    return find_words(waveform, sr)