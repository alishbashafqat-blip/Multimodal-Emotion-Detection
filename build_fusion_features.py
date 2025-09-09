import os, random, numpy as np
from glob import glob
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# ==== CONFIG (paths apne project ke) ====
BASE = r"E:\MultimodalPrototype"
FACE_TRAIN = os.path.join(BASE, "face_dataset", "train")
FACE_TEST  = os.path.join(BASE, "face_dataset", "test")
HAND_TRAIN = os.path.join(BASE, "hand_dataset", "train")
HAND_TEST  = os.path.join(BASE, "hand_dataset", "test")

FACE_MODEL_PATH = os.path.join(BASE, "face_emotion_model.h5")
HAND_MODEL_PATH = os.path.join(BASE, "hand_model.h5")  # <- jo naam tumne save kiya

# Per-class sample cap (prototype ke liye chhota rakho; badhane ke liye numbers badhao)
TRAIN_CAP_PER_CLASS = 300
TEST_CAP_PER_CLASS  = 100

random.seed(42)
np.random.seed(42)

# ==== load models ====
face_model = load_model(FACE_MODEL_PATH)
hand_model = load_model(HAND_MODEL_PATH)

# input sizes autodetect (H, W, C)
face_h, face_w = face_model.input_shape[1], face_model.input_shape[2]
hand_h, hand_w = hand_model.input_shape[1], hand_model.input_shape[2]

def preprocess(path, target_size):
    img = Image.open(path).convert("RGB").resize(target_size)
    arr = img_to_array(img) / 255.0
    return np.expand_dims(arr, axis=0)

def get_classes(path):
    # folders = subdirs only
    return sorted([d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))])

def collect_pairs(face_root, hand_root, cap_per_class):
    classes_face = set(get_classes(face_root))
    classes_hand = set(get_classes(hand_root))
    common = sorted(list(classes_face.intersection(classes_hand)))
    assert len(common) > 0, "No common class names between face and hand datasets!"

    X_list, y_list = [], []
    for cls in common:
        f_imgs = glob(os.path.join(face_root, cls, "*"))
        h_imgs = glob(os.path.join(hand_root, cls, "*"))
        if len(f_imgs) == 0 or len(h_imgs) == 0:
            continue

        # jitni pairs chahiye, utni random pick karo
        n = min(len(f_imgs), len(h_imgs), cap_per_class)
        f_pick = random.sample(f_imgs, n)
        h_pick = random.sample(h_imgs, n)

        for f_path, h_path in zip(f_pick, h_pick):
            # face probs
            f_in  = preprocess(f_path, (face_w, face_h))
            f_prob = face_model.predict(f_in, verbose=0)[0]  # shape (3,)
            # hand probs
            h_in  = preprocess(h_path, (hand_w, hand_h))
            h_prob = hand_model.predict(h_in, verbose=0)[0]  # shape (3,)

            feat = np.concatenate([f_prob, h_prob], axis=0)  # shape (6,)
            X_list.append(feat)
            y_list.append(cls)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=object)
    return X, y, common

print("Building TRAIN fusion features...")
X_train, y_train, classes_tr = collect_pairs(FACE_TRAIN, HAND_TRAIN, TRAIN_CAP_PER_CLASS)
print("Train:", X_train.shape, len(np.unique(y_train)))

print("Building TEST fusion features...")
X_test, y_test, classes_te = collect_pairs(FACE_TEST, HAND_TEST, TEST_CAP_PER_CLASS)
print("Test:", X_test.shape, len(np.unique(y_test)))

# sanity: classes order the same across splits (just for info; not strictly needed)
print("Classes (train):", classes_tr)
print("Classes (test): ", classes_te)

# save
np.savez(os.path.join(BASE, "fusion_train.npz"), X=X_train, y=y_train, classes=classes_tr)
np.savez(os.path.join(BASE, "fusion_test.npz"),  X=X_test,  y=y_test,  classes=classes_te)
print("✅ Saved fusion_train.npz & fusion_test.npz")
