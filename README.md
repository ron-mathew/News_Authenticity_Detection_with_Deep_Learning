# Fake News Detection Using ML & Deep Learning  
Python | BERT | LSTM | TensorFlow (2024–25)

## Overview
This project is an end-to-end Fake News Detection System built using a combination of Machine Learning and Deep Learning models. It processes real-world news articles and tweets, performs advanced text preprocessing, extracts meaningful features, and classifies news as real or fake with high accuracy.

This repository includes the full pipeline: data preprocessing, feature engineering, model building (ML + DL + Transformers), and evaluation.

## Key Features
- Achieved 94% accuracy using an ensemble of Naive Bayes, Logistic Regression, Random Forest, LSTM, and BERT.
- Hybrid ML + DL architecture combining TF-IDF features and contextual embeddings from BERT.
- Full preprocessing pipeline including tokenization, stopword removal, lemmatization, and tweet noise cleaning.
- Processed over 50,000 news articles and tweets.
- Includes evaluation metrics such as accuracy, precision, recall, F1-score, and confusion matrix.

## Project Architecture
Dataset → Preprocessing → Feature Extraction → Model Training → Ensemble → Evaluation

## Data Preprocessing  
- Lowercasing, punctuation removal  
- Tokenization  
- Stopword removal  
- Lemmatization  
- Noise cleaning for tweets (mentions, hashtags, URLs)

## Feature Engineering  
- TF-IDF vectorization for Machine Learning models  
- BERT transformer embeddings  
- Sequence padding and token indexing for LSTM  

## Models Implemented  
Machine Learning Models:  
- Naive Bayes  
- Logistic Regression  
- Random Forest  

Deep Learning Models:  
- LSTM  
- BERT (fine-tuned)

## Ensemble Model  
Weighted ensemble combining predictions from the top-performing ML and DL models.

## Repository Structure
Fake-News-Detection/  
├── README.md  
├── requirements.txt  
├── dataset/ (not included in repository)  
├── notebooks/  
│   └── fake-news-detection.ipynb  
├── src/  
│   ├── preprocessing.py  
│   ├── feature_extraction.py  
│   ├── train_models.py  
│   ├── bert_model.py  
│   └── ensemble.py  
└── results/  
    └── evaluation_metrics.json  

## Tech Stack
Python, TensorFlow, Keras, PyTorch (Transformers), Scikit-Learn, NLTK, Pandas, NumPy

## How to Run
1. Clone the repository  
2. Install dependencies using: pip install -r requirements.txt  
3. Open and run the Jupyter notebook or directly execute scripts from the src/ directory.

## Results
Naive Bayes: ~87%  
Logistic Regression: ~90%  
Random Forest: ~92%  
LSTM: ~93%  
BERT: ~94%  
Ensemble Model: 94%

## Future Improvements
- Add RoBERTa and DistilBERT variants  
- Deploy using Flask or Streamlit  
- Add LIME/SHAP explainability  
- Add real-time inference API  

## Contact
Ron Mathew  
AIML Student, Symbiosis Institute of Technology  
Email: your-email-here
