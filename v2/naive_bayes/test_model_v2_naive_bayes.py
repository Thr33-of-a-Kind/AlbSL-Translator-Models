import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split

def generate_confusion_matrix(model, data, labels):
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)
    model.fit(x_train, y_train)

    y_predict = model.predict(x_test)

    accuracy = accuracy_score(y_test, y_predict)
    precision = precision_score(y_test, y_predict, average='macro', zero_division=1)
    recall = recall_score(y_test, y_predict, average='macro')
    f1 = f1_score(y_test, y_predict, average='macro')
    conf_matrix = confusion_matrix(y_test, y_predict)

    print('Accuracy: {:.2%}'.format(accuracy))
    print('Precision: {:.2%}'.format(precision))
    print('Recall: {:.2%}'.format(recall))
    print('F1 score: {:.2%}'.format(f1))

    # Generate and plot the confusion matrix as a heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g', square=True)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()

def main():
    # Load data from pickle file
    pickles = pickle.load(open('../data.pkl', 'rb'))

    data = []
    for d in pickles['data']:
        data.append(np.pad(d, (0, 84 - len(d))))

    data = np.asarray(data)
    labels = np.asarray(pickles['labels'])

    model = GaussianNB()  # Create a Naive Bayes classifier instance

    generate_confusion_matrix(model, data, labels)

if __name__ == '__main__':
    main()
