"""
X_years (ndarray) (m,n)) : Data, m (15) years with n (30) teams 
y (ndarray (m,))   : target values (teams*years)

year_data (ndarray) (n,i)) : n (30) teams with i (22) features
W (ndarray) (i,)) : 1d array with i features

Each team  has (i) [22] features 

X = [[]]
"""

import numpy as np
import csv
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import plot_model
import tensorflow as tf

def import_data(latest_year, excluded_year):
    # Training set: Every year
    X_years = []
    # Target values
    y = []
    # Import each years data
    year = 2005
    while year < latest_year:
        if year != excluded_year:
            yr = str(year)
            with open(f"data/simple/{yr}.csv", 'r') as file:
                csv_reader = csv.reader(file)
                row_count = 0
                for row in csv_reader:
                    if row_count != 0:
                        y.append(row.pop())
                        X_years.append(row[3:])
                    row_count+=1
        year +=1
    return X_years,y

def import_random_data(latest_year, excluded_year):
     # Training set: Every year
    X_years = []
    X_o = []
    # Target values
    y = []
    y_o = []
    # Import each years data
    year = 2005
    while year < latest_year:
        if year != excluded_year:
            yr = str(year)
            with open(f"data/{yr}.csv", 'r') as file:
                csv_reader = csv.reader(file)
                row_count = 0
                for row in csv_reader:
                    if row_count != 0:
                        champ = row.pop()
                        if float(champ) == 1:
                            X_years.append(row[3:])
                            y.append(champ)
                        else:
                            y_o.append(champ)
                            X_o.append(row[3:])
                    row_count+=1
        year +=1
    
    # Set the random seed for reproducibility (optional)
    random.seed(1234)

    # Randomly select 18 samples from the dataset
    random_samples = random.sample(X_o, 18)
    X_years += random_samples
    y += [0]*18

    return X_years,y

def import_prediction(exluded_year):
    X = []
    with open(f"data/simple/{exluded_year}.csv", 'r') as file:
        csv_reader = csv.reader(file)
        row_count = 0
        for row in csv_reader:
            if row_count != 0:
                row.pop()
                X.append(row[3:])
            row_count+=1
    Xn = np.array(X)
    float_X = Xn.astype(float)
    return float_X

def import_advanced_prediction(excluded_year):
    X = []
    with open(f"data/advanced/{excluded_year}.csv", 'r') as file:
        csv_reader = csv.reader(file)
        row_count = 0
        for row in csv_reader:
            if row_count != 0:
                r = row[2:27]
                r = [element for element in r if element != '']
                X.append(r)
            row_count+=1
    X = list(filter(lambda x: x != '', X))
    Xn = np.array(X)
    float_X = Xn.astype(float)
    return float_X

def import_advanced_data(latest_year, excluded_year):
     # Training set: Every year
    X_years = []
    # Target values
    y = []
    # Import each years data
    year = 2005
    while year < latest_year:
        if year != excluded_year:
            yr = str(year)
            with open(f"data/advanced/{yr}.csv", 'r') as file:
                csv_reader = csv.reader(file)
                row_count = 0
                for row in csv_reader:
                    if row_count != 0:
                        y.append(row[27])
                        r = row[2:27]
                        r = [element for element in r if element != '']
                        X_years.append(r)
                        
                    row_count+=1
        year +=1
    return X_years,y

X,y = import_advanced_data(2024,2009)
Xn = np.array(X)
Yn = np.array(y)
float_X = Xn.astype(float)
float_Y = Yn.astype(float)

norm_l = tf.keras.layers.Normalization(axis=-1)
norm_l.adapt(float_X)  # learns mean, variance
normalized_x = norm_l(float_X)
tf.random.set_seed(1234)
model = Sequential(
    [   
        Dense(16, activation='relu', name = 'layer1'),
        Dense(1, activation='sigmoid', name = 'layer3'),
     ]
)

# defines a loss function and specifies a compile optimization.
model.compile(
    loss = tf.keras.losses.BinaryCrossentropy(),
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001),
)

# runs gradient descent and fits the weights to the data.
model.fit(
    float_X,float_Y,epochs=300
)

x_test = import_advanced_prediction("2009")
predictions = model.predict(x_test)
print("predictions = \n", predictions)

"""

X,y = import_data(2025,2017)

"""