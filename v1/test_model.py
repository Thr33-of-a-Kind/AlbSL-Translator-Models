import cv2
import numpy as np
import mediapipe as mp
import math
import tensorflow

modelPath = "./keras_model.h5"
labelsPath = "./labels.txt"

labels = {}
with open(labelsPath, 'r') as file:
    for line in file:
        line = line.strip()
        if line:
            index, label = line.split(' ')
            labels[int(index)] = label

data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
model = tensorflow.keras.models.load_model(modelPath)


def predict(img):
    image_size = cv2.resize(img, (224, 224))
    image_array = np.asarray(image_size)
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    data[0] = normalized_image_array

    predicted = model.predict(data)
    predicted_index = np.argmax(predicted)

    return labels[predicted_index]


capture = cv2.VideoCapture(0)
mpDraw = mp.solutions.drawing_utils
mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False,
                      max_num_hands=1,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5)

offset = 20
imageSize = 300

while True:
    success, image = capture.read()
    h, w, c = image.shape

    imageOutput = image.copy()

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

            mpDraw.draw_landmarks(image, handLandmarks, mpHands.HAND_CONNECTIONS)

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
            imgResizedShape = imageResized.shape
            widthGap = math.ceil((imageSize - widthCalculated) / 2)

            imageBackground[:, widthGap:widthCalculated + widthGap] = imageResized

            label = predict(imageBackground)
        else:
            coefficient = imageSize / width
            heightCalculated = math.ceil(coefficient * height)

            imageResized = cv2.resize(imageCrop, (imageSize, heightCalculated))
            imgResizedShape = imageResized.shape
            heightGap = math.ceil((imageSize - heightCalculated) / 2)

            imageBackground[heightGap:heightCalculated + heightGap, :] = imageResized

            label = predict(imageBackground)

        cv2.rectangle(imageOutput, (x - offset, y - offset - 50), (x - offset + 90, y - offset - 50 + 50), (255, 0, 0), cv2.FILLED)
        cv2.putText(imageOutput, label, (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
        cv2.rectangle(imageOutput, (x - offset, y - offset), (x + width + offset, y + height + offset), (255, 0, 0), 4)

    cv2.imshow("Image", imageOutput)
    key = cv2.waitKey(5)

    if key == ord("q"):
        break

capture.release()
cv2.destroyAllWindows()
