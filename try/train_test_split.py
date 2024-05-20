# train_test_split_to_csv.py
import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

input_file_path = 'Notebook\data\stud.csv' 
df = pd.read_csv(input_file_path)
x = df.drop(columns=["math_score"],axis=1)
y = df["math_score"]


def perform_train_test_split(x , y , test_size=0.2, random_state=42):
    
    x_train, x_test, y_train, y_test = train_test_split(x , y , test_size=test_size, random_state=random_state)
    return x_train, x_test , y_train , y_test

def save_data_to_csv(x_train, x_test, y_train, y_test, x_train_file_path, x_test_file_path , y_train_file_path, y_test_file_path):
    x_train.to_csv(x_train_file_path, index=False)
    x_test.to_csv(x_test_file_path, index=False)
    y_train.to_csv(y_train_file_path, index=False)
    y_test.to_csv(y_test_file_path, index=False)


def main():
    # Path to the input CSV file
    input_file_path = 'Notebook\data\stud.csv'  # Change this to your input CSV file path

    # Read the input data from CSV
    df = pd.read_csv(input_file_path)

    x = df.drop(columns=["math_score"],axis=1)
    y = df["math_score"]

    
    # Perform train-test split
    x_train,x_test,y_train,y_test = perform_train_test_split(x , y, test_size=0.2, random_state=42)

    output_dir = 'try_train_test_split_function'
    
    # Create the directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the train and test sets to CSV files in the specified directory
    x_train_file_path = os.path.join(output_dir, 'x_train.csv')
    x_test_file_path = os.path.join(output_dir, 'x_test.csv')
    y_train_file_path = os.path.join(output_dir, 'y_train.csv')
    y_test_file_path = os.path.join(output_dir, 'y_test.csv')
    save_data_to_csv(x_train, x_test, y_train, y_test, x_train_file_path, x_test_file_path , y_train_file_path, y_test_file_path)
    
    print(f"x Training data saved to {x_train_file_path}")
    print(f"x Testing data saved to {x_test_file_path}")
    print(f"y Training data saved to {y_train_file_path}")
    print(f"y Testing data saved to {y_test_file_path}")
# train_test_split.py

def get_data_paths():
    """
    Returns the file paths for x_train, x_test, y_train, and y_test.

    Returns:
    tuple: A tuple containing the file paths for x_train, x_test, y_train, and y_test.
    """
    x_train_path = 'path/to/x_train.csv'
    x_test_path = 'path/to/x_test.csv'
    y_train_path = 'path/to/y_train.csv'
    y_test_path = 'path/to/y_test.csv'

    return x_train_path, x_test_path, y_train_path, y_test_path


 



if __name__ == "__main__":
    main()
    
    
