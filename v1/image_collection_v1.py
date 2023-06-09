import math
import os
import time

import cv2
import mediapipe as mp
import numpy as np

# mpDraw = mp.solutions.drawing_utils
mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False,
                      max_num_hands=1,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5)

imageFolder = "./images/A"
if not os.path.exists(imageFolder):
    os.makedirs(imageFolder)

offset = 20
imageSize = 300
imageCounter = 0

capture = cv2.VideoCapture(0)
while True:
    success, image = capture.read()
    h, w, c = image.shape

    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(imageRGB)

    hand = None
    if results.multi_hand_landmarks:
        for handLandmarks in results.multi_hand_landmarks:
            xList = []
            yList = []
            for _, handLandmark in enumerate(handLandmarks.landmark):
                px, py, pz = int(handLandmark.x * w), int(handLandmark.y * h), int(handLandmark.z * w)
                xList.append(px)
                yList.append(py)

            xMin, xMax = min(xList), max(xList)
            yMin, yMax = min(yList), max(yList)
            boxWidth, boxHeight = xMax - xMin, yMax - yMin
            borderBox = xMin, yMin, boxWidth, boxHeight

            hand = {"borderBox": borderBox}

            # mpDraw.draw_landmarks(image, handLandmarks, mpHands.HAND_CONNECTIONS)

    imageBackground = None
    if hand:
        x, y, width, height = hand['borderBox']

        imageBackground = np.ones((imageSize, imageSize, 3), np.uint8) * 255
        imageCrop = image[y - offset:y + height + offset, x - offset:x + width + offset]
        imageCropShape = imageCrop.shape

        aspectRatio = height / width

        if aspectRatio > 1:
            coefficient = imageSize / height
            widthCalculated = math.ceil(coefficient * width)
            imageResized = cv2.resize(imageCrop, (widthCalculated, imageSize))

            imageResizedShape = imageResized.shape
            widthGap = math.ceil((imageSize - widthCalculated) / 2)
            imageBackground[:, widthGap:widthCalculated + widthGap] = imageResized
        else:
            coefficient = imageSize / width
            heightCalculated = math.ceil(coefficient * height)
            imageResized = cv2.resize(imageCrop, (imageSize, heightCalculated))

            imageResizedShape = imageResized.shape
            heightGap = math.ceil((imageSize - heightCalculated) / 2)
            imageBackground[heightGap:heightCalculated + heightGap, :] = imageResized

        cv2.imshow("Cropped captured video with background", imageBackground)

    cv2.imshow("Captured video", image)
    key = cv2.waitKey(1)

    if imageBackground:
        if key == ord("s"):
            imageCounter += 1
            cv2.imwrite(f'{imageFolder}/Image_{time.time()}.jpg', imageBackground)
            print(imageCounter)
    elif key == ord("q"):
        break

capture.release()
cv2.destroyAllWindows()
