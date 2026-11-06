import os
import re
import numpy as np
import soundfile as sf
import torch
import torchaudio
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold, cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import ConfusionMatrixDisplay

root = "voices_data"
pattern = re.compile(r"-(none|babb|musi|tele)-sp(\d+)-")
mfcc = torchaudio.transforms.MFCC(
    sample_rate=16000, n_mfcc=20,
    melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 40},
)


def features(path):
    data, sr = sf.read(path, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    x = torch.from_numpy(data).unsqueeze(0)
    m = mfcc(x)[0]
    return torch.cat([m.mean(dim=1), m.std(dim=1)]).numpy()


X, y, groups = [], [], []
for dirpath, _, files in os.walk(root):
    for f in files:
        if not f.endswith(".wav") or f.startswith("._"):
            continue
        match = pattern.search(f)
        if not match:
            continue
        X.append(features(os.path.join(dirpath, f)))
        y.append(match.group(1))
        groups.append(match.group(2))

X, y, groups = np.array(X), np.array(y), np.array(groups)
print(f"Files: {len(y)} | Speakers: {len(set(groups))}")

classifiers = {
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "SVM": SVC(),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
}

cv = GroupKFold(n_splits=5)
results = {}
print("\nResults (5-fold, split by speaker):")
for name, clf in classifiers.items():
    model = make_pipeline(StandardScaler(), clf)
    scores = cross_val_score(model, X, y, groups=groups, cv=cv)
    results[name] = scores.mean()
    print(f"{name:20s} accuracy: {scores.mean() * 100:.1f}% (+/- {scores.std() * 100:.1f})")

best = max(results, key=results.get)
model = make_pipeline(StandardScaler(), classifiers[best])
pred = cross_val_predict(model, X, y, groups=groups, cv=cv)

ConfusionMatrixDisplay.from_predictions(y, pred)
plt.title(f"Event detection - {best}")
plt.savefig("event_confusion.png", dpi=150, bbox_inches="tight")
plt.show()