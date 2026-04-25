# Step 1: Import Libraries
import nltk
from nltk.corpus import movie_reviews
import random
from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy

# Step 2: Download required NLTK datasets
nltk.download('movie_reviews')
nltk.download('punkt')

# Step 3: Load movie reviews dataset
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

# Shuffle the documents for random train/test split
random.shuffle(documents)

print("Sample document:", documents[0])  # Check sample

# Step 4: Prepare feature list
all_words = nltk.FreqDist(w.lower() for w in movie_reviews.words())
word_features = list(all_words)[:2000]  # Top 2000 words

# Function to extract features from a document
def extract_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features[f'contains({word})'] = (word in document_words)
    return features

# Test feature extraction on one positive review
features = extract_features(movie_reviews.words('pos/cv000_29590.txt'))
print("Sample features:", list(features.items())[:10])

# Step 5: Convert documents into feature sets
featuresets = [(extract_features(d), c) for (d, c) in documents]

# Step 6: Split into training and test sets (80% train, 20% test)
train_set = featuresets[:1600]
test_set = featuresets[1600:]

# Step 7: Train Naive Bayes Classifier
classifier = NaiveBayesClassifier.train(train_set)

# Step 8: Test accuracy
print("Accuracy on test set:", accuracy(classifier, test_set))

# Step 9: Function to predict sentiment of new review
def predict_sentiment(text):
    words = nltk.word_tokenize(text.lower())
    feats = extract_features(words)
    return classifier.classify(feats)

# Test the classifier on a sample review
sample_review = "This movie was absolutely fantastic, I loved it!"
print("Review:", sample_review)
print("Predicted Sentiment:", predict_sentiment(sample_review))

# Step 10: Show most informative features
print("\nTop 10 Most Informative Features:")
classifier.show_most_informative_features(10)
