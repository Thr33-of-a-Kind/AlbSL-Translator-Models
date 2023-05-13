import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

offset = 20
imageSize = 300

imageFolder = "images/A"
imageCounter = 0

while True:
    success, image = cap.read()
    hands, image = detector.findHands(image)
    if hands:
        hand = hands[0]
        x, y, width, height = hand['bbox']

        imageWhite = np.ones((imageSize, imageSize, 3), np.uint8) * 255
        imageCrop = image[y - offset:y + height + offset, x - offset:x + width + offset]

        imageCropShape = imageCrop.shape

        aspectRatio = height / width

        if aspectRatio > 1:
            coefficient = imageSize / height
            widthCalculated = math.ceil(coefficient * width)
            imageResized = cv2.resize(imageCrop, (widthCalculated, imageSize))

            imageResizedShape = imageResized.shape
            widthGap = math.ceil((imageSize - widthCalculated) / 2)
            imageWhite[:, widthGap:widthCalculated + widthGap] = imageResized

        else:
            coefficient = imageSize / width
            heightCalculated = math.ceil(coefficient * height)
            imageResized = cv2.resize(imageCrop, (imageSize, heightCalculated))

            imageResizedShape = imageResized.shape
            heightGap = math.ceil((imageSize - heightCalculated) / 2)
            imageWhite[heightGap:heightCalculated + heightGap, :] = imageResized

        cv2.imshow("Croped Video Capture", imageCrop)
        cv2.imshow("Croped Video White Background", imageWhite)

    cv2.imshow("Captured Video", image)
    key = cv2.waitKey(1)

    if key == ord("s"):
        imageCounter += 1
        cv2.imwrite(f'{imageFolder}/Image_{time.time()}.jpg', imageWhite)
        print(imageCounter)
