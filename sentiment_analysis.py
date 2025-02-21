import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# Download the vader lexicon for sentiment analysis
nltk.download('vader_lexicon')

# Step 1: Load the dataset
df = pd.read_csv('restaurant_reviews.csv')

# Step 2: Preprocess the text data
# Initialize Sentiment Intensity Analyzer (VADER)
sia = SentimentIntensityAnalyzer()

# Step 3: Create sentiment score for each review
df['sentiment_score'] = df['review_text'].apply(lambda x: sia.polarity_scores(x)['compound'])

# Step 4: Assign sentiment labels (positive if score > 0, negative if score < 0)
df['sentiment_label'] = df['sentiment_score'].apply(lambda x: 'positive' if x > 0 else 'negative')

# Step 5: Convert text data into features using TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['review_text'])

# Step 6: Define the target variable (sentiment_label)
y = df['sentiment_label']

# Step 7: Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 8: Initialize and train a Random Forest Classifier
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Step 9: Evaluate the model
y_pred = clf.predict(X_test)
print("Classification Report for Test Data:")
print(classification_report(y_test, y_pred))

# Step 10: Save the model and vectorizer for future use
joblib.dump(clf, 'sentiment_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

# Step 11: Function to predict sentiment for new user input
def predict_sentiment(user_review):
    # Load the model and vectorizer
    model = joblib.load('sentiment_model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
    
    # Transform the user review using the saved vectorizer
    review_vectorized = vectorizer.transform([user_review])
    
    # Predict the sentiment of the review
    sentiment = model.predict(review_vectorized)[0]
    return sentiment

# Example of how to predict sentiment based on user input
user_review = input("Enter a restaurant review: ")
sentiment = predict_sentiment(user_review)
print(f"The sentiment of the review is: {sentiment}")
