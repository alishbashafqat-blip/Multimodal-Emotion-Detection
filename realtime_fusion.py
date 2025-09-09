import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import joblib

# Load models
face_model = load_model("face_emotion_model.h5")
hand_model = load_model("hand_model.h5")
fusion_clf = joblib.load("fusion_model.pkl")

# Class labels
class_names = ["angry", "happy", "sad"]

# MediaPipe setup
mp_face = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.6)
mp_hands = mp.solutions.hands.Hands(max_num_hands=1, min_detection_confidence=0.6)

# Webcam
cap = cv2.VideoCapture(0)

def preprocess(img, target_size):
    img = cv2.resize(img, target_size)
    img = img.astype("float32") / 255.0
    return np.expand_dims(img, axis=0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Face detection
    face_results = mp_face.process(rgb)
    face_input = None
    if face_results.detections:
        for det in face_results.detections:
            bbox = det.location_data.relative_bounding_box
            x1, y1 = int(bbox.xmin * w), int(bbox.ymin * h)
            x2, y2 = x1 + int(bbox.width * w), y1 + int(bbox.height * h)
            face_crop = frame[y1:y2, x1:x2]
            if face_crop.size != 0:
                face_input = preprocess(face_crop, (48,48))  # same as training
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            break  # only take first face

    # Hand detection
    hand_results = mp_hands.process(rgb)
    hand_input = None
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            x_coords = [lm.x for lm in hand_landmarks.landmark]
            y_coords = [lm.y for lm in hand_landmarks.landmark]
            x_min, x_max = int(min(x_coords) * w), int(max(x_coords) * w)
            y_min, y_max = int(min(y_coords) * h), int(max(y_coords) * h)
            hand_crop = frame[y_min:y_max, x_min:x_max]
            if hand_crop.size != 0:
                hand_input = preprocess(hand_crop, (48,48))  # same as training
                cv2.rectangle(frame, (x_min,y_min), (x_max,y_max), (255,0,0), 2)
            break  # only first hand

    # Prediction if both available
    if face_input is not None and hand_input is not None:
        face_probs = face_model.predict(face_input, verbose=0)[0]
        hand_probs = hand_model.predict(hand_input, verbose=0)[0]
        fused = np.concatenate([face_probs, hand_probs]).reshape(1, -1)
        '''final_label = fusion_clf.predict(fused)[0]
        final_proba = fusion_clf.predict_proba(fused)[0]

        text = f"Predicted: {final_label}"
        cv2.putText(frame, text, (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)'''
        final_idx = fusion_clf.predict(fused)[0]          # yahan number aayega (0/1/2)
        final_label = class_names[final_idx]              # number ko word me convert kar dega
        final_proba = fusion_clf.predict_proba(fused)[0]

        text = f"Predicted: {final_label}"
        cv2.putText(frame, text, (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)


    cv2.imshow("Multimodal Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
