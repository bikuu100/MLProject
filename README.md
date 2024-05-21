
# Student Performance Indicator

This project aims to predict student performance based on various attributes such as gender, ethnicity, parental education level, lunch type, test preparation course, and scores in math, reading, and writing. The model is deployed as a web application using Flask and hosted on AWS.

## Table of Contents

- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Setup and Installation](#setup-and-installation)
- [Data Collection and Preprocessing](#data-collection-and-preprocessing)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Web Application](#web-application)
- [Deployment](#deployment)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Introduction

This project leverages machine learning techniques to predict student performance. The main goal is to identify students who may need additional support and intervention based on their predicted scores.

## Project Structure

```
├── app.py                     # Flask application file
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
├── data                       # Directory for storing data
│   └── student_performance.csv
├── notebooks                  # Jupyter notebooks for data exploration and analysis
│   └── EDA.ipynb & Model Training.ipynb
├── try                        # Source code for data processing and model training
│   ├── preprocess.py
│   ├── train_model.py
│   └── predict_pipeline.py
└── templates                  # HTML templates for Flask app
    └── index.html
```

## Setup and Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/bikuu100/MLProject.git
    cd student-performance-indicator
    ```

2. **Create and activate a virtual environment**:
    ```bash
    python -m venv venv
    source venv/bin/activate   # On Windows use `venv\Scripts\activate`
    ```

3. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Run the Flask app**:
    ```bash
    python app.py
    ```

## Data Collection and Preprocessing

- The dataset contains various features such as `gender`, `ethnicity`, `parental level of education`, `lunch type`, `test preparation course`, and scores in `math`, `reading`, and `writing`.
- Data preprocessing steps include handling missing values, encoding categorical variables, and normalizing numerical features.

## Exploratory Data Analysis

- Conducted thorough EDA using Pandas, Matplotlib, and Seaborn to uncover patterns and insights.
- Visualizations and findings are documented in the `notebooks/EDA.ipynb` file.

## Model Training and Evaluation

- Trained multiple models including Linear Regression, Decision Trees, and Random Forests.
- Evaluated models using performance metrics such as Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).
- Selected the best-performing model for deployment.

## Web Application

- Developed a web application using Flask.
- The app provides an interface to input student attributes and get the predicted performance scores.

## Deployment

- The Flask web application is deployed on AWS.
- Follow the instructions in the `deployment` directory for details on setting up the AWS environment and deploying the app.

## Usage

1. **Start the Flask app**:
    ```bash
    python app.py
    ```

2. **Access the web application**:
    Open a web browser and navigate to `http://127.0.0.1:5000`.

3. **Input student details** and get the predicted performance scores.

## Results

- The deployed model provides accurate predictions of student performance based on the input features.
- Detailed results and performance metrics are documented in the `notebooks/EDA.ipynb` file.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License.

## Contact

For any questions or suggestions, please contact Bikash Kushwaha(mailto:kushwahabikash875@gmail.com).

---

