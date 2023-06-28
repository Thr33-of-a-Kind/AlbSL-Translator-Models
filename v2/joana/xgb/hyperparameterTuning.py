import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier

# Load the data from the pickle file
pickles = pickle.load(open('../../data.pkl', 'rb'))

data = []
for d in pickles['data']:
    data.append(np.pad(d, (0, 84 - len(d))))

data = np.asarray(data)
labels = np.asarray(pickles['labels'])

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Define the parameter grid for grid search
param_grid = {
    'n_estimators': [10, 50],
    'max_depth': [3, 6],
    'learning_rate': [0.1]
}

# Create an instance of the XGBoost classifier
model = XGBClassifier(objective="binary:logistic")

# Perform grid search to find the best parameters
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', cv=5)
grid_search.fit(x_train, y_train)

# Get the best parameters and best score
best_params = grid_search.best_params_
best_score = grid_search.best_score_

# Train the model with the best parameters
best_model = XGBClassifier(**best_params)
best_model.fit(x_train, y_train)

# Make predictions on the testing data using the best model
y_predict = best_model.predict(x_test)

# Calculate evaluation metrics using the best model
accuracy = accuracy_score(y_test, y_predict)
precision = precision_score(y_test, y_predict, average='macro')
recall = recall_score(y_test, y_predict, average='macro')
f1 = f1_score(y_test, y_predict, average='macro')
classification = classification_report(y_test, y_predict)

# Save the evaluation metrics and best parameters to a text file
with open('metricsGridSearchCV.txt', 'w') as f:
    f.write('Best parameters: {}\n'.format(best_params))
    f.write('Best score: {:.2%}\n'.format(best_score))
    f.write('Accuracy: {:.2%}\n'.format(accuracy))
    f.write('Precision: {:.2%}\n'.format(precision))
    f.write('Recall: {:.2%}\n'.format(recall))
    f.write('F1 score: {:.2%}\n'.format(f1))
    f.write('\n--- Classification Report ---\n\n')
    f.write(classification)

