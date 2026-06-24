# Phishing Email Detection System using Natural Language Processing

This semester project is built directly from the proposal: an NLP-based phishing email detection system that preprocesses email text, extracts features, trains multiple machine learning and deep learning models, compares their performance, and provides a Flask prototype for real-time email classification. Source proposal: fileciteturn0file0

## 1. Project Coverage
This project includes all core items requested in the proposal:
- Text preprocessing pipeline
- Feature extraction using BoW, TF-IDF, Word2Vec-ready sequence pipeline, and BERT tokenization/embeddings flow
- Classical ML models: Naive Bayes, SVM, Random Forest
- Deep learning models: LSTM and BERT
- Evaluation using Accuracy, Precision, Recall, F1-score, Confusion Matrix, and ROC-AUC
- Dataset split: 70% train, 15% validation, 15% test
- Flask deployment prototype for real-time prediction
- Full environment setup and runnable code

## 2. Suggested Folder Structure
```bash
phishing_email_detection_system/
├── app.py
├── requirements.txt
├── .env.example
├── README.md
├── run_all.py
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── saved_models/
├── notebooks/
├── static/
│   └── css/
├── templates/
└── src/
```

## 3. Environment Setup
### Create virtual environment
```bash
python -m venv venv
```

### Activate virtual environment
Windows:
```bash
venv\Scripts\activate
```
Linux/macOS:
```bash
source venv/bin/activate
```

### Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
python -m nltk.downloader punkt stopwords wordnet omw-1.4
python -m spacy download en_core_web_sm
```

## 4. Dataset
Use a Kaggle phishing email dataset with columns similar to:
- `text` or `Email Text`
- `label` where phishing = 1 and legitimate = 0

Save dataset here:
```bash
data/raw/phishing_email.csv
```

If your dataset uses different column names, update them in the scripts or rename the columns.

## 5. Run the Full Pipeline
```bash
python run_all.py --csv data/raw/phishing_email.csv --text-column text --label-column label
```

## 6. Train Individual Models
### Classical ML
```bash
python -m src.train_ml --csv data/raw/phishing_email.csv --text-column text --label-column label
```

### LSTM
```bash
python -m src.train_lstm --csv data/raw/phishing_email.csv --text-column text --label-column label
```

### BERT
```bash
python -m src.train_bert --csv data/raw/phishing_email.csv --text-column text --label-column label
```

## 7. Launch Flask App
```bash
python app.py
```
Then open:
```bash
http://127.0.0.1:5000
```

## 8. Notes
- BERT training needs more RAM and time than other models.
- LSTM and BERT are optional to retrain every run; you can reuse saved models.
- If your system is low-end, first run the classical ML pipeline.

## 9. Expected Outcome
A fully functional phishing email detection system with comparative analysis and a real-time email classification prototype, aligned with the proposal target of high-accuracy phishing detection. fileciteturn0file0
