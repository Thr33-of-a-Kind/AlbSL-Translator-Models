import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
import pickle
import warnings

warnings.filterwarnings("ignore")
st.set_option('deprecation.showfileUploaderEncoding', False)
labelsPath = "./labels.txt"
modelPath = "./model.pkl"

labels = {}
with open(labelsPath, 'r') as file:
    for line in file:
        line = line.strip()
        if line:
            index, predicted_character = line.split(' ')
            labels[int(index)] = predicted_character

pickles = pickle.load(open(modelPath, 'rb'))
model = pickles['model']

mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=True, min_detection_confidence=0.3)

st.title("Albanian Sign Language Translation")
st.subheader("About")
st.markdown(
    """
    This Albanian Sign Language Translation project was developed by **Team Three of a Kind** consisting of:
    - Drini Karkini
    - Joana Jaupi
    - Eugen Selenica

    The aim of the project is to translate sign language gestures captured by a webcam into text.
    The hand gestures are detected using the MediaPipe library, and a pre-trained model is used to predict
    the corresponding characters or words based on the detected gestures.
    """
)

video_capture = cv2.VideoCapture(0)

# Create placeholders for displaying the video and translation
video_placeholder = st.empty()
translation_placeholder = st.empty()

while True:
    dataAux = []
    xList = []
    yList = []

    success, image = video_capture.read()
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

    # Convert the annotated image to RGB format for display in Streamlit
    annotated_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Display the video frame and translation in Streamlit
    video_placeholder.image(annotated_image, channels="RGB")
    translation_placeholder.write("Predicted Character: " + predicted_character)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close Streamlit
video_capture.release()
cv2.destroyAllWindows()
