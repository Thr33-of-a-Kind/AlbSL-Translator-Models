import pickle

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, \
    classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB


def evaluation_results(model, data, labels):
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)
    model.fit(x_train, y_train)

    y_predict = model.predict(x_test)

    print(classification_report(y_test, y_predict))
    conf_matrix = confusion_matrix(y_test, y_predict)


    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g', square=True)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.savefig('naive_bayes_confusion_matrix.png')
    plt.show()


# Load data from pickle file
pickles = pickle.load(open('../../data.pkl', 'rb'))

data = []
for d in pickles['data']:
    data.append(np.pad(d, (0, 84 - len(d))))

data = np.asarray(data)
labels = np.asarray(pickles['labels'])

model = GaussianNB()

evaluation_results(model, data, labels)
