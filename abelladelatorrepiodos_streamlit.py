import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

df = pd.read_csv('Obesity Classification.csv')
le = LabelEncoder()

df['Gender'] = le.fit_transform(df['Gender'])
df.drop(labels='ID', axis=1, inplace=True)

X = df.drop('Label', axis=1)
y = df['Label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

RClassifier = RandomForestClassifier()
RClassifier.fit(X_train, y_train)
print(X_test)
y_pred = RClassifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("ACCURACY: ", accuracy)

with open('RandomForest_model.pkl', 'wb') as f:
    pickle.dump(RClassifier, f)

