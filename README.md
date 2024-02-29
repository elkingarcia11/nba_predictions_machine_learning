# NBA Champion Predictor - Machine Learning Model

## Overview

This repository contains a Python script for predicting team outcomes based on historical data. The script utilizes a neural network implemented using TensorFlow and Keras. The data includes features for each team, such as performance metrics, and the target variable represents the success of the team in a given year.

## Files

- `binary_champ.py`: The main Python script that imports data, trains the model, and makes predictions.
- `data/`: Directory containing CSV files with historical team data.
- `data/simple/`: Directory containing CSV files with simplified team data.
- `data/advanced/`: Directory containing CSV files with advanced team data.

## Dependencies

- numpy
- csv
- random
- tensorflow
- tensorflow.keras

Install dependencies using:

```bash
pip install numpy csv random tensorflow
```

## Usage

1. Ensure you have the required dependencies installed.
2. Adjust the parameters in the script as needed (e.g., `latest_year`, `excluded_year`, etc.).
3. Run the script using:

    ```bash
    python binary_champ.py
    ```

4. The script will import training data, train the neural network, and make predictions for a specified year.

## Functions

### `import_data(latest_year, excluded_year)`

Import training data and target values from multiple years, excluding a specified year.

### `import_random_data(latest_year, excluded_year)`

Import training data with random samples and target values, excluding a specified year.

### `import_prediction(excluded_year)`

Import data for prediction from a specified year.

### `import_advanced_prediction(excluded_year)`

Import advanced data for prediction from a specified year.

### `import_advanced_data(latest_year, excluded_year)`

Import advanced training data and target values from multiple years, excluding a specified year.

## Data Format

- `X_years`: List of data arrays for each year (excluding the excluded_year).
- `y`: List of target values corresponding to the teams and years.

## Model Training

1. The script imports data using the provided functions.
2. The data is converted to NumPy arrays and normalized using TensorFlow's Normalization layer.
3. A Sequential model is defined with two Dense layers.
4. The model is compiled with a binary cross-entropy loss function and Adam optimizer.
5. The model is trained using the training data and target values.

## Example Usage

```python
# Example Usage:
X, y = import_advanced_data(2024, 2009)

# ... (model training)

# Import advanced data for prediction from a specified year
x_test = import_advanced_prediction("2009")

# Make predictions using the trained model
predictions = model.predict(x_test)

# Display the predictions
print("predictions = \n", predictions)
```

Feel free to modify the script and parameters based on your specific use case and dataset.