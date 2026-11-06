import numpy as np
import torch
import torchaudio
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

from vad_utils import find_8_words

torch.manual_seed(0)
SR = 8000

dataset = torchaudio.datasets.YESNO(root=".", download=True)
mfcc = torchaudio.transforms.MFCC(
    sample_rate=SR, n_mfcc=13,
    melkwargs={"n_fft": 200, "hop_length": 80, "n_mels": 23},
)


def word_features(word):
    m = mfcc(word)[0]
    return torch.cat([m.mean(dim=1), m.std(dim=1)]).numpy()


def add_noise(waveform, snr_db):
    signal_power = waveform.pow(2).mean()
    noise_power = signal_power / (10 ** (snr_db / 10))
    return waveform + torch.randn_like(waveform) * noise_power.sqrt()


recordings = []
for i in range(len(dataset)):
    waveform, sr, labels = dataset[i]
    segs = find_8_words(waveform, sr)
    if len(segs) == 8:
        recordings.append((waveform, segs, labels))


def extract(recs, snr_db=None):
    X, y = [], []
    for waveform, segs, labels in recs:
        if snr_db is not None:
            waveform = add_noise(waveform, snr_db)
        for (s, e), label in zip(segs, labels):
            word = waveform[:, int(s * SR):int(e * SR)]
            X.append(word_features(word))
            y.append(label)
    return np.array(X), np.array(y)


idx = np.arange(len(recordings))
train_idx, test_idx = train_test_split(idx, test_size=0.3, random_state=42)
train_recs = [recordings[k] for k in train_idx]
test_recs = [recordings[k] for k in test_idx]

X_train, y_train = extract(train_recs)

snr_levels = [None, 20, 10, 5, 0, -5]
test_sets = {snr: extract(test_recs, snr) for snr in snr_levels}

classifiers = {
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "SVM": SVC(),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
}

results = {}
for name, clf in classifiers.items():
    model = make_pipeline(StandardScaler(), clf)
    model.fit(X_train, y_train)
    results[name] = []
    for snr in snr_levels:
        X_test, y_test = test_sets[snr]
        results[name].append(accuracy_score(y_test, model.predict(X_test)) * 100)

names = ["clean" if s is None else f"{s} dB" for s in snr_levels]
print(f"{'Classifier':20s}" + "".join(f"{n:>9s}" for n in names))
for name, accs in results.items():
    print(f"{name:20s}" + "".join(f"{a:8.1f}%" for a in accs))

plt.figure(figsize=(9, 5))
for name, accs in results.items():
    plt.plot(names, accs, marker="o", label=name)
plt.title("YesNo: classifier accuracy vs noise level")
plt.xlabel("Noise level (SNR)")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig("noise_results.png", dpi=150, bbox_inches="tight")
plt.show()