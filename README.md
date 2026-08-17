# Global-Conflict-Classification

#Problem
Real-world conflict datasets lack structured severity labels, making classification a semi-supervised challenge requiring intelligent label generation before supervised learning can be applied.

#Solution
A hybrid pipeline that first applies K-Means clustering to generate severity labels from unlabeled data, then trains Logistic Regression and Naïve Bayes classifiers on those labels — achieving strong balanced accuracy on the 3,000-row UCDP conflict dataset.

#Pipeline Architecture
-> Data ingestion and preprocessing (missing values, normalization, encoding)
-> EDA with Matplotlib and Pandas
-> K-Means clustering for automatic severity label generation
-> Logistic Regression and Naïve Bayes training on clustered labels
-> Evaluation using balanced accuracy, confusion matrices, and classification reports Challenges
-> Choosing optimal K required elbow method analysis and domain reasoning
-> Label noise from unsupervised clustering introduced class imbalance in downstream classifiers
-> Balancing model interpretability vs. accuracy across both classifiers
