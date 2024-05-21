# app.py
from flask import Flask, render_template, request
import sys
sys.path.append("src/pipeline")  # Add the directory containing predict_pipeline.py to the Python path
from predict_pipeline import predict_pipeline


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')  # Render the home page template

@app.route('/predict_datapoint', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'POST':
        # Get form data
        gender = request.form['gender']
        ethnicity = request.form['ethnicity']
        parental_level_of_education = request.form['parental_level_of_education']
        lunch = request.form['lunch']
        test_preparation_course = request.form['test_preparation_course']
        reading_score = int(request.form['reading_score'])
        writing_score = int(request.form['writing_score'])

        # Pass form data to predict_pipeline function and get predictions
        features = {
            'gender': gender,
            'race_ethnicity': ethnicity,
            'parental_level_of_education': parental_level_of_education,
            'lunch': lunch,
            'test_preparation_course': test_preparation_course,
            'reading_score': reading_score,
            'writing_score': writing_score
        }
        predictions = predict_pipeline(features)

        return render_template('home.html', results=predictions[0][0])  # Render the prediction result template

    return render_template('index.html')  # Render the prediction form template

if __name__ == '__main__':
    app.run(debug=True)
