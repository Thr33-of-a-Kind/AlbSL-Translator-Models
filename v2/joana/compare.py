import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeClassifier


pickles = pickle.load(open('../data.pkl', 'rb'))

data = []
for d in pickles['data']:
    data.append(np.pad(d, (0, 84 - len(d))))

data = np.asarray(data)
labels = np.asarray(pickles['labels'])

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

names = ["knn", "xgb", " ridge"]
classifiers = [KNeighborsClassifier(),  XGBClassifier(n_estimators=50, max_depth=6, learning_rate=0.1, objective="binary:logistic"), RidgeClassifier()]

scores = []
for name, classifier in zip(names, classifiers):
    classifier.fit(x_train, y_train)
    score = classifier.score(x_test, y_test)
    scores.append(score)

data_frame = pd.DataFrame()
data_frame['name'] = names
data_frame['score'] = scores

sns.set(style="whitegrid")
sns.set_color_codes("pastel")
plt.figure(figsize=(15, 5))
ax = sns.barplot(x="score", y="name", data=data_frame)
ax.set(xlim=(0, 1))
plt.title('Models Score Comparison')
plt.xlabel('Score')
plt.ylabel('Model')
plt.tight_layout()
plt.savefig('comparison.png')
plt.show()

for name, score in zip(names, scores):
    print(f'Name: {name}')
    print('Score: {:.3%}'.format(score))
    print()