import os
import re
from collections import Counter
import soundfile as sf

root = "voices_data"
wavs = []
for dirpath, _, files in os.walk(root):
    for f in files:
        if f.endswith(".wav") and not f.startswith("._"):
            wavs.append(os.path.join(dirpath, f))

print("WAV files:", len(wavs))
print("\nExamples:")
for p in wavs[:5]:
    print(" ", os.path.basename(p))

pattern = re.compile(r"-(none|babb|musi|tele)-")
counts = Counter()
for p in wavs:
    m = pattern.search(os.path.basename(p))
    counts[m.group(1) if m else "other"] += 1
print("\nDistractor types:", dict(counts))

info = sf.info(wavs[0])
print(f"\nSample rate: {info.samplerate} | Duration: {info.duration:.2f} s")