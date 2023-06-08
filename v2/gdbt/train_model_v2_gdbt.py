import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# Load the data from the pickle file
pickles = pickle.load(open('../data.pkl', 'rb'))

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
classification_reports = []
best_score = -1
model = XGBClassifier(n_estimators=2, max_depth=6, learning_rate=1, objective="binary:logistic")
for _ in range(num_iterations):
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)
    model.fit(x_train, y_train)

    y_predict = model.predict(x_test)

    accuracy = accuracy_score(y_test, y_predict)
    precision = precision_score(y_test, y_predict, average='macro')
    recall = recall_score(y_test, y_predict, average='macro')
    f1 = f1_score(y_test, y_predict, average='macro')
    classification = classification_report(y_test, y_predict)



    if best_score < accuracy:
        best_score = accuracy
        best_y_test = y_test
        best_y_predict = y_predict
        best_labels = model.classes_

    accuracy_scores.append(accuracy)
    precision_scores.append(precision)
    recall_scores.append(recall)
    f1_scores.append(f1)
    classification_reports.append(classification)


average_accuracy = np.mean(accuracy_scores)
average_precision = np.mean(precision_scores)
average_recall = np.mean(recall_scores)
average_f1 = np.mean(f1_scores)

# Save the evaluation metrics to a text file
with open('metrics.txt', 'w') as f:
    f.write('Average accuracy: {:.2%}\n'.format(average_accuracy))
    f.write('Average precision: {:.2%}\n'.format(average_precision))
    f.write('Average recall: {:.2%}\n'.format(average_recall))
    f.write('Average F1 score: {:.2%}\n'.format(average_f1))
    f.write('\n--- Classification Reports ---\n\n')
    for i, report in enumerate(classification_reports):
        f.write(f'Iteration {i + 1}:\n')
        f.write(report)
        f.write('\n\n')



# Plot the boxplot of all metrics
data = [accuracy_scores, precision_scores, recall_scores, f1_scores]
labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
plt.boxplot(data, labels=labels)
plt.title('Evaluation Metrics')
plt.ylabel('Score')
plt.savefig('metrics_boxplot.png')
plt.close()


confusion_matrix_display = ConfusionMatrixDisplay.from_predictions(best_y_test, best_y_predict)

# Plot the confusion matrix
fig, ax = plt.subplots(figsize=(8, 6))
confusion_matrix_display.plot(ax=ax)

# Add a title and axis labels
plt.title('Confusion matrix of the iteration with best accuracy')
plt.xlabel('Predicted')
plt.ylabel('True')

# Save the figure as a PNG image
plt.savefig('confusion_matrix.png', dpi=300)
plt.close()


# Save the trained model to a pickle file
pickleFile = open('model.pkl', 'wb')
pickle.dump({'model': model}, pickleFile)
pickleFile.close()