import cv2


video = cv2.VideoCapture('http://192.168.0.169:8090/camera.mjpeg')

while(True):
    ret, frame = video.read()
    if ret:
        cv2.imshow('VIDEO', frame)
        cv2.waitKey(1)