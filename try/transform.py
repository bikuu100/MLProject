import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import joblib
import numpy as np
import os

numerical_feautres = ['reading_score', 'writing_score']
categorical_feautres = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']



def custom_preprocessor(numerical_feautres, categorical_feautres):

    
    numerical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore')),
        ('scaler', StandardScaler(with_mean=False))
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_pipeline, numerical_feautres),
            ('cat', categorical_pipeline, categorical_feautres)
        ]
    )
    return preprocessor


import train_test_split


# Get data from train_test_split.py
x_train = pd.read_csv(r'try_train_test_split_function\x_train.csv')
x_test = pd.read_csv(r'try_train_test_split_function\x_test.csv')
y_train = pd.read_csv(r'try_train_test_split_function\y_train.csv')
y_test = pd.read_csv(r'try_train_test_split_function\x_test.csv')

# Now you can use x_train, x_test, y_train, y_test in your script

preprocessor=custom_preprocessor(numerical_feautres,categorical_feautres)



# Fit and transform x_train
x_train_preprocessed = preprocessor.fit_transform(x_train)

# Transform x_test
x_test_preprocessed = preprocessor.transform(x_test)

train_arr = np.c_[
                x_train_preprocessed, np.array(y_train)
            ]
test_arr = np.c_[x_test_preprocessed, np.array(y_test)]

directory = 'try_train_test_split_function'

if not os.path.exists(directory):
    os.makedirs(directory)


# Save the preprocessor as .pkl file in the specified directory
output_file_path = os.path.join(directory, 'preprocessor.pkl')
joblib.dump(preprocessor, output_file_path)

def main():
    # Your main code here
    pass


if __name__ == "__main__":
    main()
