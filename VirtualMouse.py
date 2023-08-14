import mediapipe as mp
import cv2
import os
import uuid
import pyautogui
import math

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
                
                # Get index finger and middle finger tip coordinates
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                
                # Get thumb tip and wrist coordinates
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                
                # Calculate distance between index and middle finger tips
                finger_distance = math.sqrt((index_tip.x - middle_tip.x)**2 + (index_tip.y - middle_tip.y)**2)
                
                # Calculate distance between index tip and thumb tip
                thumb_distance = math.sqrt((index_tip.x - thumb_tip.x)**2 + (index_tip.y - thumb_tip.y)**2)
                
                # Calculate distance between index tip and wrist
                wrist_distance = math.sqrt((index_tip.x - wrist.x)**2 + (index_tip.y - wrist.y)**2)
                
                # Check which fingers are up
                thumb_up = thumb_tip.y < wrist.y
                index_up = index_tip.y < wrist.y
                middle_up = middle_tip.y < wrist.y
                
                # Moving mode (only index finger)
                if index_up and not middle_up:
                    # Move the mouse pointer
                    pyautogui.moveTo(index_tip.x * frame.shape[1], index_tip.y * frame.shape[0])
                    
                # Clicking mode (both index finger and middle finger up)
                if index_up and middle_up:
                    # Click the mouse
                    pyautogui.click()
                
                # Zoom in/out based on thumb and index finger distance
                if thumb_up and index_up:
                    zoom_factor = thumb_distance / wrist_distance
                    pyautogui.scroll(zoom_factor * 10)
                elif not thumb_up and index_up:
                    zoom_factor = thumb_distance / wrist_distance
                    pyautogui.scroll(-zoom_factor * 10)
                
                # Scroll functionality (index and middle fingers)
                if index_up and middle_up:
                    if middle_tip.y < index_tip.y:
                        pyautogui.scroll(-1)  # Scroll up
                    elif middle_tip.y > index_tip.y:
                        pyautogui.scroll(1)   # Scroll down
                
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
