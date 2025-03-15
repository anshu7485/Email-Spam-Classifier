import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

# Load dataset with encoding to handle non-UTF-8 characters
data = pd.read_csv('spam_classifier/spam.csv', encoding='ISO-8859-1')

# Print the first few rows to inspect the dataset
print(data.head())

# Drop unnecessary columns (Unnamed columns)
data = data[['v1', 'v2']]  # Keep only the first two columns

# Rename the columns for easier access
data.columns = ['label', 'message']  # Rename 'v1' to 'label' and 'v2' to 'message'

# Convert 'ham' and 'spam' to 0 and 1
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Feature (X) and Target (y)
X = data['message']  # Features are the email/SMS text
y = data['label']    # Labels are 0 or 1 (ham or spam)

# Vectorization
vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

# Model training (Naive Bayes classifier)
model = MultinomialNB()
model.fit(X_train, y_train)

# Model evaluation
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save the trained model and the vectorizer
with open('spam_classifier/model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

with open('spam_classifier/vectorizer.pkl', 'wb') as vec_file:
    pickle.dump(vectorizer, vec_file)
