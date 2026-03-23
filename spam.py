# ================= IMPORT LIBRARIES =================
import pandas as pd
import numpy as np
import re
import pickle

# Visualization
import seaborn as sns
import matplotlib.pyplot as plt

# NLP Libraries
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Machine Learning
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score


# ================= DOWNLOAD NLTK DATA =================
nltk.download('punkt')
nltk.download('stopwords')


# ================= LOAD DATASET =================
# Dataset format: LABEL (spam/ham) and MESSAGES (text)
messages = pd.read_csv('EmailCollection', sep='\t', names=['LABEL', 'MESSAGES'])

# ================= DATA VISUALIZATION =================
# Show distribution of spam vs ham
sns.countplot(x='LABEL', data=messages)
plt.title("Spam vs Ham Distribution")
plt.show()


# ================= TEXT PREPROCESSING =================
stemmer = PorterStemmer()
corpus = []

for i in range(len(messages)):
    # Remove non-alphabet characters
    review = re.sub('[^a-zA-Z]', ' ', messages['MESSAGES'][i])

    # Convert to lowercase
    review = review.lower()

    # Tokenization
    review = review.split()

    # Remove stopwords and apply stemming
    review = [
        stemmer.stem(word)
        for word in review
        if word not in stopwords.words('english')
    ]

    # Join words back into sentence
    review = ' '.join(review)

    corpus.append(review)

print("Sample processed text:", corpus[:5])


# ================= FEATURE EXTRACTION =================
# Convert text into numerical vectors (Bag of Words)
cv = CountVectorizer(max_features=3500)
X = cv.fit_transform(corpus).toarray()

# Encode labels (spam = 1, ham = 0)
y = pd.get_dummies(messages['LABEL'])
y = y.iloc[:, 1].values


# ================= TRAIN TEST SPLIT =================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)


# ================= MODEL TRAINING =================
# Using Multinomial Naive Bayes
model = MultinomialNB(alpha=0.8)
model.fit(X_train, y_train)


# ================= MODEL EVALUATION =================
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)


# ================= TEST WITH SAMPLE MESSAGE =================
sample_message = "Congratulations! You have won a free prize. Call now!"
data = [sample_message]

vect = cv.transform(data).toarray()
prediction = model.predict(vect)[0]

if prediction == 0:
    print("Result: Ham (Not Spam)")
else:
    print("Result: Spam")


# ================= SAVE MODEL & VECTORIZER =================
# Save trained model
with open('model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

# Save CountVectorizer
with open('cv-transform.pkl', 'wb') as vectorizer_file:
    pickle.dump(cv, vectorizer_file)

print("Model and vectorizer saved successfully!")