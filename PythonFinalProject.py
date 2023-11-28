import plotly.express as px
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
from wordcloud import STOPWORDS
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_squared_error,
    mean_absolute_error,
)

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('vader_lexicon')

# Task 2: Data Cleansing, Classification, and Word Clouds
# 2.1 Load the data into Python and perform the initial sanity checks

# Load the books dataset
books_df = pd.read_csv('books_data.csv')

# Load the reviews dataset
reviews_df = pd.read_csv('Books_rating.csv')

# Merge the datasets on the 'Title' column
merged_df = pd.merge(reviews_df, books_df, on='Title', how='inner')

# Perform initial checks
# Set option to display all columns
pd.set_option('display.max_columns', None)

print(merged_df[["Title", "review/score","review/summary","review/text"]].head()) #Display the first few rows to understand the data structure
print(merged_df.info())  # Check data types
print(merged_df.describe(include='all'))

pd.reset_option('display.max_columns')

# understand the extent of the missing data
missing_data = merged_df.isna().sum()
print(missing_data)

# handling duplicates titles of the top 10
# Define the old title and the desired new title
old_title = '"A careless word-- a needless sinking": A history of the staggering losses suffered by the U.S. Merchant Marine, both in ships and personnel during World War II'
new_title = '"A careless word, a needless sinking": A history of the staggering losses suffered by the U.S. Merchant Marine, both in ships and personnel during World War II'

# Replace the old title with the new title in the DataFrame
merged_df['Title'] = merged_df['Title'].replace(old_title, new_title)

# treating missing values that are important for the analysis: title and review/summary
merged_df.dropna(subset=['Title'], inplace=True)  #removing entire rows of NA titles from the dataset (only 208)
merged_df.dropna(subset=['review/summary'], inplace=True)   #dropping rows so that only complete reviews are considered

# check if the missing data was treated
missing_data = merged_df.isna().sum()
print(missing_data)

#cleaning the text from review/summary column

# Define a function for text preprocessing
def preprocess_text(text):
    # 1. Text Cleaning
    # Remove special characters, numbers, and multiple spaces
    text = re.sub(r'[^a-zA-Z\s]', '', text) #special characters
    text = re.sub(r'\d+', '', text) #numbers
    text = re.sub(' +', ' ', text) #multiple spaces

    # 2. Tokenization
    words = word_tokenize(text)

    # 3. Lowercasing
    words = [word.lower() for word in words]

    # 4. Stopword Removal
    stopwords = set(STOPWORDS)
    words = [word for word in words if word not in stopwords]

    # Join the words back into a cleaned text
    cleaned_text = ' '.join(words)

    return cleaned_text

# Apply text preprocessing to the 'review/summary' column
merged_df['review/summary'] = merged_df['review/summary'].apply(preprocess_text)



# 2.2 Generate initial analytical visualizations to understand the reviews (i.e., histogram) including a word cloud (using the wordcloud package). Discuss your findings.

# Printing some basic information about the dataset

print("There are {} observations and {} features in this dataset. \n".format(merged_df.shape[0],merged_df.shape[1]))

print("There are {} different books in this dataset.\n".format(len(merged_df.Title.unique())))
top_titles = merged_df['Title'].value_counts().head(10)
print("Top 10 Titles:")
print(top_titles)

unique_categories = merged_df['categories'].unique().astype(str)
print("There are {} different categories in this dataset.\n".format(len(merged_df.categories.unique())))
top_categories = merged_df['categories'].value_counts().head(10)
print("Top 10 Categories:")
print(top_categories)

# Visualize the top 10 categories based on the number of books in each category
top_categories = merged_df['categories'].value_counts().head(10)
plt.figure(figsize=(8, 5))
top_categories.plot(kind='bar', edgecolor='black')
plt.xlabel('Category')
plt.ylabel('Number of Books')
plt.title('Top 10 Categories by Number of Books')
plt.xticks(rotation=45)
plt.show()

# Create a histogram of review ratings
plt.hist(merged_df['review/score'], bins=5, edgecolor='black')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.title('Distribution of Review Ratings')
plt.show()

# Print the top 10 titles based on highest ratings

# Group the data by 'Title' and calculate the mean of 'review/score'
title_ratings_mean = merged_df.groupby('Title')['review/score'].mean()

# Sort the titles by mean rating in descending order and select the top 10
top_titles = title_ratings_mean.nlargest(10)
print("Top 10 Titles based on Highest Average Ratings:")
print(top_titles)

# Creating a histogram of review text lengths (number of characters)
review_lengths = merged_df['review/text'].str.len()

# Define the bin edges
bin_edges = range(0, 1050, 50)

plt.hist(review_lengths, bins=bin_edges, edgecolor='black')
plt.xlabel('Review Text Length (Number of Characters)')
plt.ylabel('Frequency')
plt.title('Distribution of Review Text Length')
plt.xticks(range(0, 1050, 50))  # Set the x-axis ticks to match the bin edges
plt.show()

# Creating a histogram of review text lengths (number of characters)
review_lengths = merged_df['review/summary'].str.len()
plt.hist(review_lengths, bins=20, edgecolor='black')
plt.xlabel('Review Summary Length (Number of Characters)')
plt.ylabel('Frequency')
plt.title('Distribution of Review Summary Length')
plt.show()

# Creating a wordcloud for text reviews

# WordCloud

text = " ".join(str(review_summary) for review_summary in merged_df['review/summary'])
print("There are {} words in the combination of all summary reviews.".format(len(text.split())))

# Create stopword list:

stopwords = set(STOPWORDS)
stopwords.update(["read", "book", "novel", "chapter", "author", "story", "one", "review", "another", "thing"])

# Generate a word cloud image
wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)

# Display the wordcloud:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

# 2.3 Classify reviews into two ‘sentiment’ categories called positive and negative
# Classifying my reviews into positive and negative sentiments by defining a functSion to classify sentiment using VADER
def classify_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    sentiment = analyzer.polarity_scores(text)

    if sentiment['compound'] >= 0.0:
        return "positive"
    else:
        return "negative"

# Apply the sentiment classification function to your DataFrame and create a new 'sentiment' column
merged_df['sentiment'] = merged_df['review/summary'].apply(classify_sentiment)

# 2.4 Generate positive and negative word clouds. Discuss your findings while comparing the positive and negative summaries
# Generate positive and negative word clouds
positive_reviews_text = " ".join(merged_df[merged_df['sentiment'] == 'positive']['review/text'])
negative_reviews_text = " ".join(merged_df[merged_df['sentiment'] == 'negative']['review/text'])

# Generate word clouds for positive and negative reviews
positive_wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(positive_reviews_text)
negative_wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(negative_reviews_text)

# Display the positive word cloud
plt.figure(figsize=(8, 8))
plt.imshow(positive_wordcloud, interpolation='bilinear')
plt.title("Word Cloud for Positive Reviews")
plt.axis("off")
plt.show()

# Display the negative word cloud
plt.figure(figsize=(8, 8))
plt.imshow(negative_wordcloud, interpolation='bilinear')
plt.title("Word Cloud for Negative Reviews")
plt.axis("off")
plt.show()

merged_df.head()

# Bar Charts of Sentiment Counts:
sentiment_counts = merged_df['sentiment'].value_counts()

# Create a bar chart
plt.bar(sentiment_counts.index, sentiment_counts.values)
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.title('Sentiment Distribution')
plt.show()

# Histogram of Review Ratings:
positive_ratings = merged_df[merged_df['sentiment'] == 'positive']['review/score']
negative_ratings = merged_df[merged_df['sentiment'] == 'negative']['review/score']

# Create histograms for positive and negative ratings
plt.hist(positive_ratings, bins=5, alpha=0.5, label='Positive')
plt.hist(negative_ratings, bins=5, alpha=0.5, label='Negative')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.title('Distribution of Ratings for Positive and Negative Reviews')
plt.legend()
plt.show()

# Box Plots of Text Length:
positive_reviews = merged_df[merged_df['sentiment'] == 'positive']['review/summary']
negative_reviews = merged_df[merged_df['sentiment'] == 'negative']['review/summary']

# Calculate text lengths for positive and negative reviews
positive_lengths = positive_reviews.apply(len)
negative_lengths = negative_reviews.apply(len)

# Create box plots for text length for positive reviews
plt.figure(figsize=(8, 6))
sns.boxplot(x=positive_lengths, y='sentiment', data=merged_df[merged_df['sentiment'] == 'positive'], orient='h', palette='Set2')
plt.xlabel('Text Length')
plt.ylabel('Sentiment')
plt.title('Box Plots of Text Length for Positive Reviews')
plt.grid(True)
plt.show()

# Create box plots for text length for negative reviews
plt.figure(figsize=(8, 6))
sns.boxplot(x=negative_lengths, y='sentiment', data=merged_df[merged_df['sentiment'] == 'negative'], orient='h', palette='Set2')
plt.xlabel('Text Length')
plt.ylabel('Sentiment')
plt.title('Box Plots of Text Length for Negative Reviews')
plt.grid(True)
plt.show()

# Task 3: Prediction
# 3.1 Build a simple logistic regression model to predict the sentiment category based on a text-based review

# removing NA's from sentiment column
merged_df.dropna(subset=['sentiment'], inplace=True)   #dropping rows ensures that only positive and negative sentiments are considered

# Generate 80% training and 20% testing samples
train_size = int(0.8 * len(merged_df))
reviews_train = merged_df['review/summary'][:train_size]
reviews_test = merged_df['review/summary'][train_size:]

# Vectorize text data using TF-IDF
vectorizer = TfidfVectorizer(max_features=1000)  # You can adjust the number of features as needed
X_train_vec = vectorizer.fit_transform(reviews_train)
X_test_vec = vectorizer.transform(reviews_test)

y_train = merged_df['sentiment'][:train_size]
y_test = merged_df['sentiment'][train_size:]

# Train a logistic regression model
logistic_regression_model = LogisticRegression()
logistic_regression_model.fit(X_train_vec, y_train)

# Predict sentiment on the testing data
y_pred = logistic_regression_model.predict(X_test_vec)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 3.2 Build a multinomial logistic regression model to predict the rating of a book based on its text-based review
y_train = merged_df['review/score'][:train_size]
y_test = merged_df['review/score'][train_size:]

# Vectorize text data using TF-IDF
X_train_vec = vectorizer.fit_transform(reviews_train)
X_test_vec = vectorizer.transform(reviews_test)

# Train a multinomial logistic regression model
multinomial_regression_model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
multinomial_regression_model.fit(X_train_vec, y_train)

# Predict review/score on the testing data
y_pred = multinomial_regression_model.predict(X_test_vec)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")
print(f"Mean Absolute Error: {mae:.2f}")

# Loading the test data for applying both prediction models
testdata = pd.read_csv('testdata.csv')

# Predict Sentiment with Logistic Regression
# 1. Text Preprocessing
test_reviews = testdata['review/summary']
preprocessed_test_reviews = test_reviews.apply(preprocess_text)

# 2. Vectorization
X_test_vec = vectorizer.transform(preprocessed_test_reviews)

# 3. Prediction
y_pred_sentiment = logistic_regression_model.predict(X_test_vec)

# Print or use the sentiment predictions as needed
print("Predicted Sentiments:")
print(y_pred_sentiment)

# Predict Rating with Multinomial Regression
# 1. Text Preprocessing
test_reviews = testdata['review/summary']
preprocessed_test_reviews = test_reviews.apply(preprocess_text)

# 2. Vectorization
X_test_vec = vectorizer.transform(preprocessed_test_reviews)

# 3. Prediction
y_pred_rating = multinomial_regression_model.predict(X_test_vec)

# Print or use the rating predictions as needed
print("Predicted Ratings:")
print(y_pred_rating)