
import mediapipe as mp
import cv2
import os
import uuid

# Initialize Mediapipe hands and drawing modules
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Open webcam
cap = cv2.VideoCapture(0)

# Initialize hands module with detection and tracking confidence thresholds
with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands: 
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert BGR image to RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Flip image horizontally
        image_rgb = cv2.flip(image_rgb, 1)
        
        # Process hand detection
        results = hands.process(image_rgb)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks on image
                mp_drawing.draw_landmarks(image_rgb, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                          mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2))
                
        # Convert RGB image back to BGR
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        
        # Display image
        cv2.imshow('Hand Tracking', image_bgr)
        
        # Save processed image with landmarks
        if results.multi_hand_landmarks:
            image_filename = os.path.join('Output Images', f'{uuid.uuid1()}.jpg')
            cv2.imwrite(image_filename, image_bgr)
        
        # Exit loop when 'q' key is pressed
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()
