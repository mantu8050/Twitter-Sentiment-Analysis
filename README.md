# Twitter Sentiment Analysis using NLTK in Python

##  Project Overview
This project focuses on performing **sentiment analysis on Twitter data** using Natural Language Processing (NLP) and machine learning techniques. The objective is to classify tweets into **positive or negative sentiment** based on their textual content.

The project demonstrates a complete NLP pipeline—from data preprocessing to feature extraction and model evaluation—using classical machine learning algorithms.

---

##  Dataset Description
The dataset consists of a large collection of tweets labeled with sentiment values.  
Each tweet undergoes extensive preprocessing to remove noise and improve feature quality.

Download the dataset: [Project Dataset](https://drive.google.com/file/d/1k6ocHqPYYGJu2XyuMtcNaeYXSKFWP5ol/view?usp=sharing)
### Dataset Columns
- **target**: the polarity of the tweet (positive or negative)
- **ids**: Unique id of the tweet
- **date**: the date of the tweet
- **flag**: It refers to the query. If no such query exists, then it is NO QUERY.
- **user**: It refers to the name of the user that tweeted
- **text**: It refers to the text of the tweet

---

##  Text Preprocessing Steps
The following preprocessing steps were applied to the tweet text:

- Removal of stopwords
- Removal of punctuation marks
- Removal of URLs
- Removal of numeric characters
- Tokenization of text
- Stemming to reduce words to their root form

These steps help standardize the text and reduce vocabulary size.

---

##  Feature Extraction
To convert text data into numerical format, **TF-IDF (Term Frequency–Inverse Document Frequency)** vectorization was used.  
TF-IDF assigns higher weights to words that are important to a document but less frequent across the corpus.

---

##  Machine Learning Models Used
The following classification models were trained and evaluated:

1. **Bernoulli Naive Bayes**  
2. **Support Vector Machine (SVM)**  
3. **Logistic Regression**

Each model was evaluated using multiple performance metrics.

---

## Evaluation Metrics
Model performance was assessed using:

- **Accuracy**
- **F1-score** (for both classes)
- **ROC-AUC Score**

These metrics provide a balanced evaluation, especially for classification tasks.

---

## Results Summary

- Logistic Regression achieved the **highest accuracy**
- It produced the **best F1-scores** for both positive and negative classes
- It recorded the **highest ROC-AUC score (0.83)** among all models

---

## Final Conclusion
Based on empirical evaluation, **Logistic Regression outperformed SVM and Bernoulli Naive Bayes** across all metrics.  
The model also follows the principle of **Occam’s Razor**, making it the most suitable choice for this dataset.

Thus, Logistic Regression was selected as the final model for sentiment classification.

---

##  Technologies Used
- Python
- Pandas, NumPy
- NLTK
- Scikit-learn
- Matplotlib / Seaborn

---

##  Future Scope
- Use word embeddings such as Word2Vec or GloVe
- Apply deep learning models like LSTM or BERT
- Extend the project to multi-class sentiment classification
- Perform real-time sentiment analysis using live Twitter data

---

##  Author
**MahantaGouda**  
Sentiment Analysis Project
