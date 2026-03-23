# ================= IMPORT LIBRARIES =================
from flask import Flask, render_template, request
import pickle

# ================= LOAD TRAINED MODEL =================

# Load the trained ML model (Multinomial Naive Bayes)
with open('model.pkl', 'rb') as model_file:
    classifier = pickle.load(model_file)

# Load the CountVectorizer used during training
with open('cv-transform.pkl', 'rb') as vectorizer_file:
    cv = pickle.load(vectorizer_file)

# ================= INITIALIZE FLASK APP =================
app = Flask(__name__)

# ================= HOME ROUTE =================
@app.route('/')
def home():
    """
    Renders the homepage where user inputs message
    """
    return render_template('home.html')


# ================= PREDICTION ROUTE =================
@app.route('/predict', methods=['POST'])
def predict():
    """
    Handles form submission and predicts whether
    the input message is SPAM or NOT SPAM
    """
    try:
        # Get message from form
        message = request.form.get('message')

        # Convert message into list format (required for vectorizer)
        data = [message]

        # Transform text using CountVectorizer
        vect = cv.transform(data).toarray()

        # Make prediction using trained model
        prediction = classifier.predict(vect)[0]

        # Send result to result.html
        return render_template('result.html', prediction=prediction)

    except Exception as e:
        # Handle errors gracefully
        return f"Error occurred: {str(e)}"


# ================= RUN APPLICATION =================
if __name__ == '__main__':
    # Run the Flask app in debug mode
    app.run(debug=True)