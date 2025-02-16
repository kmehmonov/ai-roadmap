import cv2

CASCASE_PATH = 'haarcascade_frontalface_default.xml'
cascade_clf = cv2.CascadeClassifier(CASCASE_PATH)

filename = 'images/maradona2.png'

frame = cv2.imread(filename=filename)

faces = cascade_clf.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=10, minSize=(30, 30))

for (x, y, w, h) in faces:
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow('Image', frame)
cv2.waitKey(0)
