import pickle
import warnings

import cv2
import mediapipe as mp
import numpy as np

warnings.filterwarnings("ignore")

labelsPath = "./labels.txt"

labels = {}
with open(labelsPath, 'r') as file:
    for line in file:
        line = line.strip()
        if line:
            index, predicted_character = line.split(' ')
            labels[int(index)] = predicted_character

pickles = pickle.load(open('drin/random_forest/knn.pkl', 'rb'))
model = pickles['model']

mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=True, min_detection_confidence=0.3)

capture = cv2.VideoCapture(0)
while True:
    dataAux = []
    xList = []
    yList = []

    success, image = capture.read()
    height, width, _ = image.shape

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

        x1 = int(min(xList) * width) - 10
        y1 = int(min(yList) * height) - 10

        x2 = int(max(xList) * width) - 10
        y2 = int(max(yList) * height) - 10

        dataAux = np.pad(dataAux, (0, 84 - len(dataAux)))

        prediction = model.predict([np.asarray(dataAux)])

        predicted_character = labels[int(prediction[0])]

        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 4)
        cv2.putText(image, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)

    cv2.imshow('AlbSL Translator', image)
    key = cv2.waitKey(1)

    if key == ord("q"):
        break

capture.release()
cv2.destroyAllWindows()
