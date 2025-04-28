import cv2
import pytesseract
from keras.models import load_model
import numpy as np
import os

# Set Tesseract path (adjust if needed)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Load face detection and emotion detection models
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
emotion_model = load_model('emotion_model.h5')

# Emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def extract_text(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray)
    return text

def classify_question(text):
    if "?" in text and any(word in text.lower() for word in ["why", "explain", "describe"]):
        return "Descriptive"
    elif any(option in text for option in ["A)", "B)", "C)", "D)"]):
        return "MCQ"
    elif "fill in the blank" in text.lower():
        return "Fill-in-the-blank"
    else:
        return "Unknown"

def analyze_emotion(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    emotions_detected = []
    for (x, y, w, h) in faces:
        roi = gray[y:y+h, x:x+w]
        roi = cv2.resize(roi, (48, 48))
        roi = roi.astype("float") / 255.0
        roi = np.reshape(roi, (1, 48, 48, 1))
        prediction = emotion_model.predict(roi)
        emotion = emotion_labels[np.argmax(prediction)]
        emotions_detected.append(emotion)
    
    return emotions_detected if emotions_detected else ["No face detected"]

def run_learning_assistant(folder_path, webcam=False):
    images = os.listdir(folder_path)
    print("\n--- Analyzing Uploaded Images ---")

    for img_name in images:
        img_path = os.path.join(folder_path, img_name)
        print(f"\nProcessing image: {img_name}")

        question = extract_text(img_path)
        print("Extracted Question:\n", question)

        question_type = classify_question(question)
        print("Question Type Detected:", question_type)

    if webcam:
        cap = cv2.VideoCapture(0)
        print("\n--- Analyzing Facial Expressions (press 'q' to quit) ---")
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            emotions = analyze_emotion(frame)
            text_display = ', '.join(emotions)

            cv2.putText(frame, f'Emotion: {text_display}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow('Facial Emotion Detection', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

# Example run
if __name__ == "__main__":
    run_learning_assistant("test_images", webcam=True)
