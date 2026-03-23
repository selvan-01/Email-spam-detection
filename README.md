# 📩 Email Spam Detection Web App (Machine Learning + Flask)

An end-to-end Machine Learning project that classifies messages as **Spam** or **Not Spam** using **Natural Language Processing (NLP)** and a **Multinomial Naive Bayes model**, deployed through a **Flask web application**.

---

## 🚀 Project Overview

This project demonstrates a complete Machine Learning pipeline:

* 📊 Data preprocessing and cleaning
* 🧠 NLP-based feature extraction
* 🤖 Model training using Naive Bayes
* 🌐 Deployment using Flask
* 🎨 Interactive web interface

Users can input any message and instantly receive a prediction.

---

## 🧠 Machine Learning Workflow

1. **Data Collection**
   Dataset containing labeled SMS/Email messages (Spam or Ham)

2. **Text Preprocessing**

   * Lowercasing
   * Removing special characters
   * Tokenization
   * Stopword removal
   * Stemming

3. **Feature Extraction**

   * CountVectorizer (Bag of Words)

4. **Model Training**

   * Multinomial Naive Bayes

5. **Evaluation**

   * Accuracy Score

6. **Deployment**

   * Flask Web Application

---

## 🛠️ Tech Stack

* **Language:** Python
* **Libraries:** scikit-learn, pandas, numpy, nltk
* **Framework:** Flask
* **Frontend:** HTML, CSS
* **Visualization:** Matplotlib, Seaborn

---

## 📁 Project Structure

```
SPAM-DETECTION/
│
├── static/
│   ├── styles.css
│   ├── spam.gif
│   ├── not-spam.gif
│   └── spam-favicon.ico
│
├── templates/
│   ├── home.html
│   └── result.html
│
├── model.pkl
├── cv-transform.pkl
├── spam.py
├── app.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation & Setup

### 1. Clone the Repository

```
git clone https://github.com/selvan-01/Email-spam-detection.git
cd spam-detection
```

### 2. Install Dependencies

```
pip install -r requirements.txt
```

### 3. Download NLTK Data

```
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

### 4. Run the Application

```
python app.py
```

### 5. Open in Browser

```
http://127.0.0.1:5000/
```

---

## 💡 How It Works

* User enters a message in the web interface
* Text is transformed using CountVectorizer
* Model predicts:

  * **Spam (1)**
  * **Not Spam (0)**
* Result is displayed visually on the webpage

---

## 📊 Sample Predictions

| Message                | Prediction |
| ---------------------- | ---------- |
| Win a free iPhone now! | Spam       |
| Let's meet tomorrow    | Not Spam   |

---

## 🎯 Key Features

✔ End-to-End Machine Learning Pipeline
✔ Real-time Prediction
✔ Clean and Simple UI
✔ NLP-based Text Processing
✔ Lightweight and Fast Model

---

## 🔥 Future Improvements

* Deep Learning models (LSTM, BERT)
* Cloud deployment (AWS, Render)
* REST API integration
* UI/UX enhancements
* Multilingual support

---

## 📌 Conclusion

This project showcases how Machine Learning and NLP can be integrated with web technologies to build real-world intelligent applications.

---

## ⭐ Support

If you found this project useful:

* ⭐ Star this repository
* 🔁 Share it
* 💬 Provide feedback

---

## 📬 Contact

* 📧 Email: [senthamils445@gmail.com](mailto:senthamils445@gmail.com)
* 💼 LinkedIn: https://www.linkedin.com/in/senthamil45

---

🚀 Built with passion for AI, Machine Learning, and real-world problem solving.
