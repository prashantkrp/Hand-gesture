import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

# Load pre-trained gesture recognition model
model = tf.keras.models.load_model('hand_gesture_recognition_model.h5')

# Start real-time hand gesture recognition using webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Draw hand landmarks on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract the bounding box of the hand
            h, w, c = frame.shape
            x_min = int(min([landmark.x for landmark in hand_landmarks.landmark]) * w)
            x_max = int(max([landmark.x for landmark in hand_landmarks.landmark]) * w)
            y_min = int(min([landmark.y for landmark in hand_landmarks.landmark]) * h)
            y_max = int(max([landmark.y for landmark in hand_landmarks.landmark]) * h)

            # Extract the hand region
            hand_img = frame[y_min:y_max, x_min:x_max]
            if hand_img.size == 0:
                continue

            # Preprocess the hand image
            img = cv2.resize(hand_img, (64, 64))
            img = img / 255.0
            img = np.expand_dims(img, axis=0)

            # Predict the gesture
            predictions = model.predict(img)
            gesture = np.argmax(predictions)

            # Display the gesture on the frame
            cv2.putText(frame, f'Gesture: {gesture}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Hand Gesture Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
