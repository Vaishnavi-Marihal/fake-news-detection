# Fake News Detection System

**Author:** Vaishnavi Marihal  
**Institution:** Jain College of Engineering and Technology, Hubballi  
**Year:** 2025  

## Overview
An NLP-based binary classification system to detect misinformation 
in news articles. Trained and evaluated four ML models (Logistic 
Regression, Naive Bayes, SVM, Random Forest) with Logistic Regression 
achieving the best performance on the ISOT Fake News Dataset.

## Results
| Metric | Score |
|--------|-------|
| Accuracy | 98% |
| F1-Score | 95.5% |
| Precision | 95% |
| Recall | 96% |

## Technologies Used
- Python
- Scikit-learn
- TF-IDF Vectorisation
- Logistic Regression
- Pandas, NumPy
- Streamlit (deployment)
- Matplotlib (evaluation plots)

## Features
- Classifies news articles as real or fake
- Supports both text input and URL-based predictions
- Deployed as a real-time web application via Streamlit
- Evaluated using confusion matrix, classification report

## How to Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Dataset
ISOT Fake News Dataset (University of Victoria)
- Real news collected from Reuters.com
- Fake news collected from unreliable sources flagged by Politifact
- Binary classification: Real (1) vs Fake (0)

## Author Note
This project was developed as part of B.E. Computer Science coursework at 
Jain College of Engineering and Technology, Hubballi, India.
