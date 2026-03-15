import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,  
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    success, frame = cap.read()
    if not success:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    total_fingers = 0
    left_fingers = 0
    right_fingers = 0

    if result.multi_hand_landmarks and result.multi_handedness:
        for hand_landmarks, handedness in zip(
            result.multi_hand_landmarks,
            result.multi_handedness
        ):
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            hand_label = handedness.classification[0].label
            lm = hand_landmarks.landmark
            fingers = 0

            
            if hand_label == "Right":
                if lm[4].x < lm[3].x:
                    fingers += 1
            else:  
                if lm[4].x > lm[3].x:
                    fingers += 1

            
            if lm[8].y < lm[6].y:
                fingers += 1
            if lm[12].y < lm[10].y:
                fingers += 1
            if lm[16].y < lm[14].y:
                fingers += 1
            if lm[20].y < lm[18].y:
                fingers += 1


            if hand_label == "Left":
                left_fingers = fingers
            else:
                right_fingers = fingers

            total_fingers += fingers

   
    cv2.rectangle(frame, (20, 20), (520, 180), (0, 0, 0), -1)

    cv2.putText(frame, f"Left Hand: {left_fingers}", (40, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    cv2.putText(frame, f"Right Hand: {right_fingers}", (40, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    cv2.putText(frame, f"Total: {total_fingers}", (40, 170),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)

    cv2.imshow("Two Hand Finger Counter", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
