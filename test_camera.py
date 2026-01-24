import cv2
cap = cv2.VideoCapture(0)
if cap.isOpened():
    print('✅ الكاميرا تشتغل!')
    ret, frame = cap.read()
    if ret:
        cv2.imshow('Test', frame)
        cv2.waitKey(3000)
else:
    print('❌ الكاميرا ما تشتغل')
cap.release()
cv2.destroyAllWindows()
