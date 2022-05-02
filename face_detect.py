import cv2
# print(cv2.__version__)

face_cascade = cv2.CascadeClassifier('cascades\data\haarcascade_frontalface_alt2.xml')

cap = cv2.VideoCapture(0)

while (True):

    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    for (x, y, w, h) in faces:
        # print(x, y, w, h)
        roi_gray = gray[y:y + h, x:x + w]  # (ycord_start, ycord_end)
        # img_item = "trained_img.png"
        cv2.imwrite("trained_img.png", roi_gray)          #write image

    cv2.imshow('frame',frame)
    if cv2.waitKey(20) & 0XFF == ord('q'):
        break


#enter q keyword to exit....
