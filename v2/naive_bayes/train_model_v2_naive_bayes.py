import pickle
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

# Load data from pickle file
pickles = pickle.load(open('../data.pkl', 'rb'))

data = []
# Pad each data item and append to the data list
for d in pickles['data']:
    data.append(np.pad(d, (0, 84 - len(d))))

data = np.asarray(data)
labels = np.asarray(pickles['labels'])

num_iterations = 10  # Number of iterations for averaging accuracy scores

accuracy_scores = []naiv
precision_scores = []
recall_scores = []
f1_scores = []

model = GaussianNB()  # Create a Naive Bayes classifier instance
for _ in range(num_iterations):
    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)
    model.fit(x_train, y_train)  # Fit the classifier to the training data

    y_predict = model.predict(x_test)  # Make predictions on the test data

    # Compute evaluation metrics
    accuracy = accuracy_score(y_test, y_predict)
    precision = precision_score(y_test, y_predict, average='macro', zero_division=1)
    recall = recall_score(y_test, y_predict, average='macro')
    f1 = f1_score(y_test, y_predict, average='macro')

    # Append scores to the respective lists
    accuracy_scores.append(accuracy)
    precision_scores.append(precision)
    recall_scores.append(recall)
    f1_scores.append(f1)

# Compute average scores
average_accuracy = np.mean(accuracy_scores)
average_precision = np.mean(precision_scores)
average_recall = np.mean(recall_scores)
average_f1 = np.mean(f1_scores)

# Print the average scores
print('Average accuracy: {:.2%}'.format(average_accuracy))
print('Average precision: {:.2%}'.format(average_precision))
print('Average recall: {:.2%}'.format(average_recall))
print('Average F1 score: {:.2%}'.format(average_f1))

# Save the trained model using pickle
pickleFile = open('model.pkl', 'wb')
pickle.dump({'model': model}, pickleFile)
pickleFile.close()
