Word Cloud and Sentiment Analysis on Amazon Book Reviews

**Overview**
*
This GitHub repository contains code for a comprehensive analysis of a sizable dataset comprising 212,404 books on Amazon, spanning the years 1996 to 2014. The dataset includes a whopping 142.8 million reviews, forming the basis for insightful visualizations and predictive models.

**Objective**
*
The primary goal of this project is to extract valuable insights from the extensive book reviews dataset. Through analytical visualizations such as histograms and word clouds (general, positive, and negative), we aim to provide a nuanced understanding of the sentiments expressed in the reviews.

**Features**
*
***Word Clouds:** Generated for general sentiments, as well as positive and negative sentiments, offering a visual representation of the most frequent words in the reviews.

***Sentiment Analysis:** Reviews were classified into positive and negative sentiments using a compound score. The project leveraged a logistic regression model to predict sentiment based on text-based reviews.

***Rating Prediction:** A multinomial logistic regression model was employed to predict book ratings from text-based reviews.

**Results**
The general word cloud closely resembled the positive word cloud, indicating a predominance of positive reviews. The compound score function supported this observation. The logistic regression model demonstrated an impressive 93% accuracy, while the multinomial logistic regression model achieved a Mean Squared Error (MSE) of 1.33 and a Mean Absolute Error (MAE) of 0.58.
