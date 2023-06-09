import os
import pickle

import cv2
import mediapipe as mp

labelsPath = "./labels.txt"

labelIndices = {}
with open(labelsPath, 'r') as file:
    for line in file:
        line = line.strip()
        if line:
            index, label = line.split(' ')
            labelIndices[label] = int(index)

print(labelIndices)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

imagesFolder = './images'

data = []
labels = []
for imageLetterDirectory in os.listdir(imagesFolder):
    for imageLetter in os.listdir(os.path.join(imagesFolder, imageLetterDirectory)):
        dataAux = []
        xList = []
        yList = []

        image = cv2.imread(os.path.join(imagesFolder, imageLetterDirectory, imageLetter))

        imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(imageRGB)

        if results.multi_hand_landmarks:
            for handLandmarks in results.multi_hand_landmarks:
                for _, handLandmark in enumerate(handLandmarks.landmark):
                    x = handLandmark.x
                    y = handLandmark.y
                    xList.append(x)
                    yList.append(y)

                for _, handLandmark in enumerate(handLandmarks.landmark):
                    x = handLandmark.x
                    y = handLandmark.y
                    dataAux.append(x - min(xList))
                    dataAux.append(y - min(yList))

            data.append(dataAux)
            labels.append(labelIndices[imageLetterDirectory])

pickleFile = open('data.pkl', 'wb')
pickle.dump({'data': data, 'labels': labels}, pickleFile)
pickleFile.close()
