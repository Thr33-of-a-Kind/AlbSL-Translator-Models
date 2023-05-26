import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("keras_model.h5", "labels.txt")

offset = 20
imageSize = 300

labels = [
    "A", "B", "C", "D", "Dh", "E", "F", "G",
    "Gj", "H", "I", "J", "K", "L", "Ll", "M",
    "N", "Nj", "O", "P", "Q", "R", "Rr", "T",
    "Th", "U", "V", "X", "Xh", "Y", "Z"
]

while True:
    success, image = cap.read()
    imageOutput = image.copy()
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
            imgResizedShape = imageResized.shape
            widthGap = math.ceil((imageSize - widthCalculated) / 2)
            imageWhite[:, widthGap:widthCalculated + widthGap] = imageResized
            prediction, index = classifier.getPrediction(imageWhite, draw=False)
        else:
            coefficient = imageSize / width
            heightCalculated = math.ceil(coefficient * height)
            imageResized = cv2.resize(imageCrop, (imageSize, heightCalculated))
            imgResizedShape = imageResized.shape
            heightGap = math.ceil((imageSize - heightCalculated) / 2)
            imageWhite[heightGap:heightCalculated + heightGap, :] = imageResized
            prediction, index = classifier.getPrediction(imageWhite, draw=False)

        cv2.rectangle(imageOutput, (x - offset, y - offset - 50), (x - offset + 90, y - offset - 50 + 50), (255, 0, 255), cv2.FILLED)
        cv2.putText(imageOutput, labels[index], (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
        cv2.rectangle(imageOutput, (x - offset, y - offset), (x + width + offset, y + height + offset), (255, 0, 255), 4)

        cv2.imshow("ImageCrop", imageCrop)
        cv2.imshow("ImageWhite", imageWhite)

    cv2.imshow("Image", imageOutput)
    cv2.waitKey(1)
