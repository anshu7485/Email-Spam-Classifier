# Email Spam Classifier

## 📌 Overview
The **Email Spam Classifier** is a machine learning-based web application designed to classify emails as either **Spam** or **Not Spam**. It utilizes **Natural Language Processing (NLP)** techniques to analyze email text and predict whether an email is spam or legitimate.

## 🚀 Features
- Train a spam classification model using **Machine Learning & NLP**.
- Load and preprocess dataset (**spam.csv**).
- Convert text data into numerical format using **TF-IDF Vectorization**.
- Train and save the model using **Logistic Regression** (or any other ML model).
- Build a **Flask web app** to classify user-input emails.
- Deploy the model for real-world usage.


---

## 📷 Project Screenshot  
![Project UI](https://github.com/anshu7485/Email-Spam-Classifier/blob/main/Screenshot%202025-03-13%20165301.png)

---


## 🛠️ Tech Stack
- **Python** (pandas, numpy, scikit-learn, flask)
- **Machine Learning** (Logistic Regression, TF-IDF Vectorizer)
- **Natural Language Processing (NLP)**
- **Flask (Web Framework)**
- **HTML, CSS (Frontend for UI)**
- **Git & GitHub (Version Control & Deployment)**

---

## 📂 Project Structure
```
📦 Email-Spam-Classifier
├── 📂 app
│   ├── 📂 spam_classifier
│   │   ├── spam.csv                # Dataset for training
│   │   ├── train.py                # Script to train the model
│   │   ├── model.pkl               # Saved trained model
│   │   ├── vectorizer.pkl          # Saved TF-IDF Vectorizer
│   ├── 📂 templates
│   │   ├── index.html              # Web app frontend UI
│   ├── app.py                      # Flask web application
├── requirements.txt                 # Required Python dependencies
├── README.md                        # Project documentation
```

---

## 📊 Dataset Details
We use the **Spam/Ham dataset (spam.csv)**, which consists of labeled emails:
- **Spam Emails**: Unwanted or unsolicited messages.
- **Ham Emails**: Genuine and important messages.

The dataset is preprocessed using **Text Cleaning, Tokenization, and TF-IDF Vectorization**.

---

## 🎯 Model Training
1. Load dataset (`spam.csv`).
2. Preprocess text (remove stop words, punctuation, special characters).
3. Convert text to numerical format using **TF-IDF Vectorizer**.
4. Train ML model (Logistic Regression/SVM/Naive Bayes).
5. Save the trained model (`model.pkl`) and vectorizer (`vectorizer.pkl`).

---

## 🖥️ Running the Project Locally
### 📌 1. Clone the repository
```bash
git clone https://github.com/anshu7485/Email-Spam-Classifier.git
cd Email-Spam-Classifier
```

### 📌 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 📌 3. Run the Flask app
```bash
python app/app.py
```

### 📌 4. Open the web application
Go to **http://127.0.0.1:5000/** in your browser.

---

## 🏆 Results & Accuracy
- Model trained with **TF-IDF + Logistic Regression** achieves an accuracy of **95-98%**.
- Can be improved by using **Deep Learning (LSTMs, Transformers)**.

---

## 🔥 Future Enhancements
- Improve accuracy using **Neural Networks** (LSTMs, Transformers).
- Add **Email Header Analysis** for better classification.
- Deploy the app using **AWS, Heroku, or Render**.

---

## 🤝 Contribution
Feel free to contribute by **forking** this repository and submitting a **pull request (PR)**.

---

## 💡 Author
**Anshu**
- GitHub: [anshu7485](https://github.com/anshu7485)
- Email: anshu@example.com

---

## ⭐ Show Your Support
If you like this project, **give it a star ⭐ on GitHub**!
