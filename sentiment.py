import pandas as pd

# Loading the dataset
data = pd.read_csv('sentiment_tweets3.csv')

# Inspecting the dataset
print(data.tail(10))

# Preprocessing the data
data.dropna(inplace=True)
data = data.rename(columns = {'label (depression result)' : 'label'})

# Features and target variable
X = data['message to examine']
y = data['label']

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

tfidf = TfidfVectorizer()

# Vectorize the text data
X_tfidf = tfidf.fit_transform(X)

# sampling
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_tfidf, y)
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)

from sklearn.metrics import classification_report

# Predict the test set results
y_pred = model.predict(X_test)

# Print classification report
print(classification_report(y_test, y_pred))

import pickle

# Save the model
with open('sentiment_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

with open('tfidf_vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(tfidf, vectorizer_file)