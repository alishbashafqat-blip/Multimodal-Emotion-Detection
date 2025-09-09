# Multimodal Emotion Detection

This project is a **university AI model** that detects human emotions using **face and hand features**.  
It combines **facial expressions** (via MediaPipe face landmarks) and **hand gestures** (via MediaPipe hand landmarks) for better accuracy.

---

##  Features
- Detects 3 emotions: **Happy**, **Sad**, **Angry**
- Uses **Face model (CNN, Keras)** and **Hand model (CNN, Keras)**
- Combines both models using a **Fusion Classifier (Random Forest / Logistic Regression)**
- Works in **real-time with webcam**

---

##  Project Structure
- `train_face.py` → trains face emotion model  
- `train_hand.py` → trains hand gesture model  
- `train_fusion.py` → combines both models  
- `realtime_fusion.py` → runs real-time detection using webcam  
- `requirements.txt` → list of dependencies  
- Models (`.h5` and `.pkl`) → pre-trained models for demo  

---

##  How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
Run real-time demo:

python realtime_fusion.py
Applications
Human-computer interaction
Virtual classrooms
Gaming and entertainment
Mental health monitoring
Author:
      Alishba Shafqat
AI Project – 2025
