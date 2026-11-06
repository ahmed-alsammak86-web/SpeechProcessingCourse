import torchaudio

dataset = torchaudio.datasets.YESNO(root=".", download=True)

print("عدد التسجيلات: - test_yesno.py:5", len(dataset))

waveform, sample_rate, labels = dataset[0]
print("شكل الصوت: - test_yesno.py:8", waveform.shape)
print("معدل العينات: - test_yesno.py:9", sample_rate)
print("التسميات: - test_yesno.py:10", labels)