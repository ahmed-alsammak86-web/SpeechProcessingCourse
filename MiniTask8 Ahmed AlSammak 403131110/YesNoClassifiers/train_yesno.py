import numpy as np
import torch
import torchaudio
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

dataset = torchaudio.datasets.YESNO(root=".", download=True)
mfcc = torchaudio.transforms.MFCC(
    sample_rate=8000, n_mfcc=13,
    melkwargs={"n_fft": 200, "hop_length": 80, "n_mels": 23},
)


def word_features(word):
    m = mfcc(word)[0]
    return torch.cat([m.mean(dim=1), m.std(dim=1)]).numpy()


X, y, groups = [], [], []
for i in range(len(dataset)):
    waveform, sr, labels = dataset[i]
    segs = find_8_words(waveform, sr)
    if len(segs) != 8:
        print(f"Skipping file {i}")
        continue
    for (s, e), label in zip(segs, labels):
        word = waveform[:, int(s * sr):int(e * sr)]
        X.append(word_features(word))
        y.append(label)
        groups.append(i)

X, y, groups = np.array(X), np.array(y), np.array(groups)

files = np.unique(groups)
train_files, test_files = train_test_split(files, test_size=0.3, random_state=42)
train_mask = np.isin(groups, train_files)
X_train, y_train = X[train_mask], y[train_mask]
X_test, y_test = X[~train_mask], y[~train_mask]
print(f"Words: {len(y)} | Train: {len(y_train)} | Test: {len(y_test)}")

classifiers = {
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "SVM": SVC(),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
}

print("\nResults:")
for name, clf in classifiers.items():
    model = make_pipeline(StandardScaler(), clf)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"{name:20s} accuracy: {acc * 100:.1f}%")0