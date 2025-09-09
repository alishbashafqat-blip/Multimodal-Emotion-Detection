import os, joblib, numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

BASE = r"E:\MultimodalPrototype"
train_npz = np.load(os.path.join(BASE, "fusion_train.npz"), allow_pickle=True)
test_npz  = np.load(os.path.join(BASE, "fusion_test.npz"),  allow_pickle=True)

X_train, y_train = train_npz["X"], train_npz["y"]
X_test,  y_test  = test_npz["X"],  test_npz["y"]

le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc  = le.transform(y_test)

# Simple & strong baseline for low-dim features
clf = LogisticRegression(max_iter=1000, multi_class="multinomial")
clf.fit(X_train, y_train_enc)

y_pred = clf.predict(X_test)
acc = accuracy_score(y_test_enc, y_pred)
print("Fusion Accuracy:", acc)
print("\nClassification Report:\n")
print(classification_report(y_test_enc, y_pred, target_names=le.classes_))

# save
joblib.dump(clf, os.path.join(BASE, "fusion_model.pkl"))
joblib.dump(le,  os.path.join(BASE, "fusion_label_encoder.pkl"))
print("✅ Saved fusion_model.pkl & fusion_label_encoder.pkl")
