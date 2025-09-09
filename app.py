import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import collections
# --- Paths ---
BASE = r"E:\MultimodalPrototype"
FACE_MODEL_PATH = BASE + r"\face_emotion_model.h5"
HAND_MODEL_PATH = BASE + r"\hand_model.h5"
FUSION_MODEL   = BASE + r"\fusion_model.pkl"
LABEL_ENC_PATH = BASE + r"\fusion_label_encoder.pkl"

# --- Load Models ---
face_model = load_model(FACE_MODEL_PATH)
hand_model = load_model(HAND_MODEL_PATH)
fusion = joblib.load(FUSION_MODEL)
le = joblib.load(LABEL_ENC_PATH)

# --- Emoji Map ---
emoji_map = {"angry": "😡", "happy": "😊", "sad": "😢"}

# --- Input Sizes ---
face_h, face_w = face_model.input_shape[1:3]
hand_h, hand_w = hand_model.input_shape[1:3]

def preprocess(img, target_size):
    img = cv2.resize(img, target_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype("float32") / 255.0
    return np.expand_dims(img, axis=0)

def predict(face_roi, hand_roi):
    f_in = preprocess(face_roi, (face_w, face_h))
    h_in = preprocess(hand_roi, (hand_w, hand_h))

    f_prob = face_model.predict(f_in, verbose=0)[0]
    h_prob = hand_model.predict(h_in, verbose=0)[0]

    feat = np.concatenate([f_prob, h_prob])[None, :]
    pred_id = fusion.predict(feat)[0]
    label = le.inverse_transform([pred_id])[0]
    return label, emoji_map.get(label, "")

# --- Streamlit UI ---
st.set_page_config(page_title="Multimodal Emotion Detection", page_icon="🤖", layout="wide")
st.title("🤖 Multimodal Emotion Detection (Face + Hand)")

run = st.checkbox("🎥 Start Webcam")
FRAME_WINDOW = st.image([])

# Initialize capture only once
if "cap" not in st.session_state:
    st.session_state.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Windows DirectShow
   # last fallback
cap = st.session_state.cap

import mediapipe as mp

mp_face = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.6)
mp_hands = mp.solutions.hands.Hands(max_num_hands=1, min_detection_confidence=0.6)

if run:
    while True:
        ret, frame = cap.read()
        if not ret:
            st.warning("Camera not detected!")
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = frame.shape

        # --- Face Detection ---
        face_results = mp_face.process(rgb)
        face_crop = None
        if face_results.detections:
            det = face_results.detections[0]
            bbox = det.location_data.relative_bounding_box
            x1, y1 = int(bbox.xmin * w), int(bbox.ymin * h)
            x2, y2 = x1 + int(bbox.width * w), y1 + int(bbox.height * h)
            face_crop = frame[y1:y2, x1:x2]
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

        # --- Hand Detection ---
        hand_results = mp_hands.process(rgb)
        hand_crop = None
        if hand_results.multi_hand_landmarks:
            lm = hand_results.multi_hand_landmarks[0]
            x_coords = [p.x for p in lm.landmark]
            y_coords = [p.y for p in lm.landmark]
            x_min, x_max = int(min(x_coords)*w), int(max(x_coords)*w)
            y_min, y_max = int(min(y_coords)*h), int(max(y_coords)*h)
            hand_crop = frame[y_min:y_max, x_min:x_max]
            cv2.rectangle(frame, (x_min,y_min), (x_max,y_max), (255,0,0), 2)

        # --- Prediction ---
        if face_crop is not None and hand_crop is not None:
            label, emoji = predict(face_crop, hand_crop)
            cv2.putText(frame, f"{label.upper()} {emoji}", (50,50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3)

        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


        # Streamlit needs a break to update page
        if not run:
            break
else:
    st.write("☝️ Check the box above to start the live webcam feed.")
