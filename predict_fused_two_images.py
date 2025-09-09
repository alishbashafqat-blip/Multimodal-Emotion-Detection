import os, argparse, numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import joblib

# Add this dictionary after imports (upar wali section me)
emoji_map = {
    "angry": "😡",
    "happy": "😊",
    "sad": "😢"
}
BASE = r"E:\MultimodalPrototype"
FACE_MODEL_PATH = os.path.join(BASE, "face_emotion_model.h5")
HAND_MODEL_PATH = os.path.join(BASE, "hand_model.h5")
FUSION_MODEL   = os.path.join(BASE, "fusion_model.pkl")
LABEL_ENC_PATH = os.path.join(BASE, "fusion_label_encoder.pkl")

face_model = load_model(FACE_MODEL_PATH)
hand_model = load_model(HAND_MODEL_PATH)
fusion = joblib.load(FUSION_MODEL)
le = joblib.load(LABEL_ENC_PATH)

face_h, face_w = face_model.input_shape[1], face_model.input_shape[2]
hand_h, hand_w = hand_model.input_shape[1], hand_model.input_shape[2]

def preprocess(path, target_size):
    img = Image.open(path).convert("RGB").resize(target_size)
    arr = img_to_array(img) / 255.0
    return np.expand_dims(arr, axis=0)

def predict(face_img, hand_img):
    f_in = preprocess(face_img, (face_w, face_h))
    h_in = preprocess(hand_img, (hand_w, hand_h))

    f_prob = face_model.predict(f_in, verbose=0)[0]  # (3,)
    h_prob = hand_model.predict(h_in, verbose=0)[0]  # (3,)

    feat = np.concatenate([f_prob, h_prob])[None, :]  # (1,6)
    pred_id = fusion.predict(feat)[0]
    pred_label = le.inverse_transform([pred_id])[0]

    # optional: fused proba (logreg me available)
    if hasattr(fusion, "predict_proba"):
        proba = fusion.predict_proba(feat)[0]
        return pred_label, proba
    return pred_label, None

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--face", required=True, help="path to a face image")
    ap.add_argument("--hand", required=True, help="path to a hand image")
    args = ap.parse_args()

    label, proba = predict(args.face, args.hand)
    emoji = emoji_map.get(label, "")   # label ka corresponding emoji nikaal lo
    print("Predicted:", label, emoji)  # yahan emoji bhi print hoga
    if proba is not None:
        # class order is le.classes_
        print("Class order:", list(le.classes_))
        print("Probs:", proba)
