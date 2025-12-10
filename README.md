# 2025-Fall---Language-Analytics

This project uses 6 part-of-speech (POS) tag features to predict the sentiment polarity of MOOC (Massive Open Online Course) reviews.

## Corpus
The corpus was obtained from [Kaggle](https://www.kaggle.com/datasets/imuhammad/course-reviews-on-coursera) and contains approximately 1.45 million Coursera user reviews. Due to the large size and positive review skew in the original dataset, a downsampled subset was created for analysis.

## MOOC_POS_tags
This file uses the [Bing customer lexicon](https://www.cs.uic.edu/~liub/FBS/sentiment-analysis.html) to calculate the number of positive and negative POS-tags, specifically adjectives, adverbs, and verbs. The counts are then normalized by the total word count.

## MOOC_tf_genism
The file calculates the most frequent adjectives, adverbs, and verbs in the downsampled MOOC reviews.

## XGBoost
The XGBoost model uses the six extracted features (positive and negative adjectives, adverbs, and verbs) to predict review polarity (positive: 4-5 stars; negative: 1-2 stars).
