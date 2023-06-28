import os.path
import pickle

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

pickles = pickle.load(open(os.path.dirname(__file__) + '/../data.pkl', 'rb'))

data = []
for d in pickles['data']:
    data.append(np.pad(d, (0, 84 - len(d))))

data = np.asarray(data)
labels = np.asarray(pickles['labels'])

num_iterations = 10  # Number of iterations for averaging accuracy scores

accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []

model = DecisionTreeClassifier()
for _ in range(num_iterations):
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)
    model.fit(x_train, y_train)

    y_predict = model.predict(x_test)

    accuracy = accuracy_score(y_test, y_predict)
    precision = precision_score(y_test, y_predict, average='macro')
    recall = recall_score(y_test, y_predict, average='macro')
    f1 = f1_score(y_test, y_predict, average='macro')

    accuracy_scores.append(accuracy)
    precision_scores.append(precision)
    recall_scores.append(recall)
    f1_scores.append(f1)

average_accuracy = np.mean(accuracy_scores)
average_precision = np.mean(precision_scores)
average_recall = np.mean(recall_scores)
average_f1 = np.mean(f1_scores)

print('Average accuracy: {:.2%}'.format(average_accuracy))
print('Average precision: {:.2%}'.format(average_precision))
print('Average recall: {:.2%}'.format(average_recall))
print('Average F1 score: {:.2%}'.format(average_f1))

pickleFile = open('decision_tree.pkl', 'wb')
pickle.dump({'model': model}, pickleFile)
pickleFile.close()
