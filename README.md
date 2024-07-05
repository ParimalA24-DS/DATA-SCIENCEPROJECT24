# 🤖 NLP Chatbot Using Cosine Similarity

Welcome to the **NLP Chatbot Project**! This is a first  project demonstrates how to create a simple chatbot using **cosine similarity** for question answering.

![Chatbot Image 1](path/to/your/image1.png)
![Chatbot Image 2](path/to/your/image2.png)
## 🚀 Project Overview

- **Purpose:** A chatbot that matches user queries to predefined questions and returns the corresponding answers.
- **Technologies:** Python, NLTK, NumPy, scikit-learn

## 📝 Problem Statement

This chatbot:
- **Tokenizes** and **removes stopwords** from user input.
- **Matches** the input to a list of predefined questions using **cosine similarity**.
- Returns the corresponding answer if a match is found.
- Responds with `"I can't answer this question."` if no match is found.

## 🛠️ Requirements

- **Python 3.x**
- Google colab/jupyter notebbok 
- **Libraries:**
  - `nltk`
  - `numpy`
  - `scikit-learn`
  - `pandas`

## 📂 Dataset

- **Source:** CSV file containing questions and answersregarding data analytics.
- **Path:** `test.csv`as per your location 

## 🔧 Setup

1. **Mount Google Drive:**
    ```python
    from google.colab import drive
    drive.mount('/content/drive')
    ```

2. **Import Libraries:**
    ```python
    import numpy as np
    import pandas as pd
    import nltk
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    ```

3. **Download NLTK Data:**
    ```python
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('stopwords')
    ```

4. **Read Dataset:**
    ```python
    path = r"/content/drive/MyDrive/IMP1DS INTERVIEW PREP2024/15.DSPROJECT2024/1.NLPPROJECTS2024/test.csv"
    df = pd.read_csv(path, encoding='unicode_escape')
    questions_list = df['Questions'].tolist()
    answers_list = df['Answers'].tolist()
    ```

## 🔍 Preprocessing

1. **Initialize Tools:**
    ```python
    from nltk.stem import WordNetLemmatizer, PorterStemmer
    from nltk.corpus import stopwords
    import re
    ```

2. **Preprocess Function:**
    ```python
    def preprocess_with_stopwords(text):
        lemmatizer = WordNetLemmatizer()
        stemmer = PorterStemmer()
        text = re.sub(r'[^\w\s]', '', text)
        tokens = nltk.word_tokenize(text.lower())
        lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
        stemmed_tokens = [stemmer.stem(token) for token in lemmatized_tokens]
        return ' '.join(stemmed_tokens)
    ```

## 📈 Vectorization

1. **Setup Vectorizer:**
    ```python
    vectorizer = TfidfVectorizer(tokenizer=nltk.word_tokenize)
    X = vectorizer.fit_transform([preprocess_with_stopwords(q) for q in questions_list])
    ```

## 🤔 Response Generation

1. **Get Response Function:**
    ```python
    def get_response(text):
        processed_text = preprocess_with_stopwords(text)
        vectorized_text = vectorizer.transform([processed_text])
        similarities = cosine_similarity(vectorized_text, X)
        max_similarity = np.max(similarities)
        if max_similarity > 0.6:
            high_similarity_questions = [q for q, s in zip(questions_list, similarities[0]) if s > 0.6]
            target_answers = [answers_list[questions_list.index(q)] for q in high_similarity_questions]
            Z = vectorizer.fit_transform([preprocess_with_stopwords(q) for q in high_similarity_questions])
            final_similarities = cosine_similarity(vectorized_text, Z)
            closest = np.argmax(final_similarities)
            return target_answers[closest]
        else:
            return "I can't answer this question."
    ```

## 📊 Usage Example

- Example Query:
    ```python
    get_response('Who is MS Dhoni?')
    ```

## 📚 Additional Tools

- **GingerIt for Grammar Check:**
    ```python
    !pip install gingerit
    from gingerit.gingerit import GingerIt
    text = 'What is Data Anlytics'
    parser = GingerIt()
    corrected_text = parser.parse(text)
    print(corrected_text['result'])
    ```

- **TextBlob for Spelling Correction:**
    ```python
    !pip install textblob
    from textblob import TextBlob
    text = 'What is Data Anlytics'
    blob = TextBlob(text)
    corrected_text = blob.correct()
    print(corrected_text)
    ```

---

Feel free to explore and contribute to the project! 🚀

