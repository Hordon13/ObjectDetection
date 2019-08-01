import cv2

cam = cv2.VideoCapture(0)

while True:
    ret_val, img = cam.read()
    cv2.imshow("cam", img)

    if cv2.waitKey(3) == 27:
        break
