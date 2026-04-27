import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load model
MODEL_PATH = "mask_detector_model.h5"
model = load_model(MODEL_PATH)

# Load Haar Cascade face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

CLASS_LABELS = ['Mask', 'No Mask']

# Start webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Cannot access webcam")
    exit()

print("Press 'q' to quit...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize for speed (optional)
    frame = cv2.resize(frame, (640, 480))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]

        # Preprocess
        face_resized = cv2.resize(face, (224, 224))
        face_array = face_resized / 255.0
        face_array = np.expand_dims(face_array, axis=0)

        # Predict
        prediction = model.predict(face_array)[0][0]

        label = CLASS_LABELS[1] if prediction > 0.5 else CLASS_LABELS[0]
        confidence = prediction if prediction > 0.5 else 1 - prediction

        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

        # Draw bounding box
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

        # Display label
        cv2.putText(
            frame,
            f"{label} ({confidence:.2f})",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )

    cv2.imshow("Mask Detection - Webcam", frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()