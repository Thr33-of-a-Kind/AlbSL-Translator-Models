import os
import cv2
import time

imageFolder = './images/I'
if not os.path.exists(imageFolder):
    os.makedirs(imageFolder)

imageCounter = 0

capture = cv2.VideoCapture(0)
while True:
    success, image = capture.read()
    cv2.imshow("Captured video", image)

    key = cv2.waitKey(200)

    if key == ord("s"):
        imageCounter += 1
        cv2.imwrite(f'{imageFolder}/Image_{time.time()}.jpg', image)
        print(imageCounter)
    elif key == ord("q"):
        break

capture.release()
cv2.destroyAllWindows()
