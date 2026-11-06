import os
import re
from collections import defaultdict
import numpy as np
import soundfile as sf
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from silero_vad import load_silero_vad, get_speech_timestamps

SR = 16000
FRAME = 400
HOP = 160
root = "voices_data"
pattern = re.compile(r"-(none|babb|musi|tele)-")
model = load_silero_vad()


def load(path):
    data, sr = sf.read(path, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    return torch.from_numpy(data)


def energy_vad(wav, margin_db=8.0):
    frames = wav.unfold(0, FRAME, HOP)
    rms = frames.pow(2).mean(dim=1).sqrt()
    db = 20 * torch.log10(rms + 1e-10)
    threshold = torch.quantile(db, 0.1) + margin_db
    return (db > threshold).numpy().astype(int)


def silero_frames(wav, n_frames):
    labels = np.zeros(n_frames, dtype=int)
    for seg in get_speech_timestamps(wav, model, sampling_rate=SR):
        labels[seg["start"] // HOP:seg["end"] // HOP] = 1
    return labels


scores = defaultdict(list)
example = None
for dirpath, _, files in os.walk(root):
    for f in files:
        if not f.endswith(".wav") or f.startswith("._"):
            continue
        match = pattern.search(f)
        if not match:
            continue
        wav = load(os.path.join(dirpath, f))
        energy = energy_vad(wav)
        ref = silero_frames(wav, len(energy))
        scores[match.group(1)].append(f1_score(ref, energy, zero_division=0))
        if example is None and match.group(1) == "musi":
            example = (f, wav, energy, ref)

types = ["none", "musi", "tele", "babb"]
print("Energy VAD vs Silero VAD (F1 on speech frames):")
for t in types:
    print(f"{t:6s} F1: {np.mean(scores[t]) * 100:.1f}%  ({len(scores[t])} files)")

plt.figure(figsize=(6, 4))
plt.bar(types, [np.mean(scores[t]) * 100 for t in types])
plt.ylabel("F1 (%)")
plt.title("Energy VAD agreement with Silero VAD")
plt.savefig("vad_f1_by_noise.png", dpi=150, bbox_inches="tight")

name, wav, energy, ref = example
t = np.arange(len(wav)) / SR
ft = np.arange(len(energy)) * HOP / SR
peak = wav.abs().max().item()

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(t, wav.numpy(), color="gray", linewidth=0.5)
ax.fill_between(ft, 0, ref * peak, step="post", color="green", alpha=0.3, label="Silero VAD")
ax.fill_between(ft, -energy * peak, 0, step="post", color="red", alpha=0.3, label="Energy VAD")
ax.set_title(name[:70])
ax.set_xlabel("Time (seconds)")
ax.legend()
plt.savefig("vad_example.png", dpi=150, bbox_inches="tight")
plt.show()