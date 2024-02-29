Sentiment Analysis with Machine Learning
=====

This project focuses on sentiment analysis using machine learning algorithms on the IMDB movie review dataset. Sentiment analysis involves identifying and classifying the sentiment expressed in textual data, which in this case is movie reviews. The goal is to predict whether a given review is positive or negative or neutral

Dataset
==
The dataset used in this project is the IMDb dataset, which contains reviews along with their corresponding sentiment labels (positive or negative). The dataset is loaded using the pandas library.

Requirements
===
+ Python 3.x
+ pandas
+ scikit-learn

Preprocessing and Model Training
===
Preprocessing Pipeline: Text data is preprocessed using a pipeline that includes TF-IDF vectorization.

Classifiers:
===
+ Multinomial Naive Bayes: A classic choice for text classification tasks.
+ Random Forest Classifier: Ensemble learning method based on decision trees.
+ Logistic Regression: A linear model suitable for binary classification tasks.
+ Hyperparameter Tuning: Grid search with cross-validation is performed to find the optimal hyperparameters for each classifier.

Approach
===
The sentiment analysis task is approached using machine learning classifiers within a pipeline that includes text preprocessing and feature extraction using TF-IDF vectorization.

+ Classifiers Explored:
+ Multinomial Naive Bayes
+ Random Forest Classifier
+ Logistic Regression
+ Hyperparameters Tuned: For each classifier, hyperparameters were tuned using grid search with cross-validation to find the optimal combination for maximizing classification performance.

Results
===
The project outputs the best hyperparameters and corresponding scores for each classifier, along with the accuracy and classification report on the test set.

Conclusion
===
After evaluating the performance of different classifiers with tuned hyperparameters, the following conclusions were drawn:

+ Multinomial Naive Bayes: Achieved an accuracy of approximately 85.14% with the best parameters: alpha=0.1, max_features=3000, ngram_range=(1, 1).
+ Random Forest Classifier: This classifier was not considered due to issues encountered during parameter tuning.
+ Logistic Regression: This classifier was not considered due to issues encountered during parameter tuning.
==
In conclusion, the Multinomial Naive Bayes classifier demonstrated the best performance for sentiment analysis on the IMDB movie review dataset.

