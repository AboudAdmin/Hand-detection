import cv2
import mediapipe as mp

# إعداد Mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# إعداد الكاميرا
cap = cv2.VideoCapture(0)

# إعداد Mediapipe Hands
with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("فشل في قراءة الصورة من الكاميرا.")
            continue
        
        # عكس الصورة أفقياً لتكون مثل المرآة
        image = cv2.flip(image, 1)
        
        # تحويل الصورة إلى RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # معالجة الصورة باستخدام Mediapipe
        results = hands.process(image_rgb)
        
        # التحقق من وجود أي يد
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # الحصول على النقطة رقم 8 وهي رأس السبابة
                index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                
                # تحويل الإحداثيات من النطاق [0, 1] إلى إحداثيات الصورة
                h, w, c = image.shape
                cx, cy = int(index_finger_tip.x * w), int(index_finger_tip.y * h)
                
                # رسم دائرة في مكان الإصبع
                cv2.circle(image, (cx, cy), 10, (255, 0, 0), -1)
                
                # رسم المعالم على اليد
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        # عرض الصورة
        cv2.imshow('Finger Drawing', image)
        
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
