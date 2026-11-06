import torchaudio
import matplotlib.pyplot as plt

dataset = torchaudio.datasets.YESNO(root=".", download=True)
waveform, sample_rate, labels = dataset[0]

time = [i / sample_rate for i in range(waveform.shape[1])]
plt.figure(figsize=(12, 4))
plt.plot(time, waveform[0])
plt.title(f"Labels: {labels}")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.show()