from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split

# Create a Flask web application
app = Flask(__name__)

# Initialize a TfidfVectorizer for text feature extraction
tfvect = TfidfVectorizer(stop_words='english', max_df=0.7)

# Load the trained model from the saved pickle file
loaded_model = pickle.load(open('model.pkl', 'rb'))

# Load the dataset from a CSV file
dataframe = pd.read_csv('output4.csv')

# Split the dataset into features (x) and labels (y)
x = dataframe['text']
y = dataframe['label']

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Function to perform fake news detection using the trained model
def fake_news_det(news):
    # Transform the training data using TfidfVectorizer
    tfid_x_train = tfvect.fit_transform(x_train)
    
    # Transform the test data using TfidfVectorizer
    tfid_x_test = tfvect.transform(x_test)
    
    # Prepare the input news for prediction
    input_data = [news]
    vectorized_input_data = tfvect.transform(input_data)
    
    # Make predictions using the loaded model
    prediction = loaded_model.predict(vectorized_input_data)
    
    return prediction

# Define the home route, rendering the index.html template
@app.route('/')
def home():
    return render_template('index.html')

# Define the route for prediction, which handles the form submission
@app.route('/predict', methods=['POST'])
def predict():
    # Check if the request method is POST
    if request.method == 'POST':
        # Retrieve the news text from the form
        message = request.form['message']
        
        # Perform fake news detection on the input text
        pred = fake_news_det(message)
        
        # Print the prediction (for debugging purposes)
        print(pred)
        
        # Render the index.html template with the prediction result
        return render_template('index.html', prediction=pred)
    else:
        # If the request method is not POST, render with an error message
        return render_template('index.html', prediction="Something went wrong")

# Run the Flask application if the script is executed
if __name__ == '__main__':
    app.run(debug=True)
