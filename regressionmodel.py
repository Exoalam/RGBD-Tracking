import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.layers import Input, Dense, Concatenate
import pandas as pd
import numpy as np
from keras.callbacks import ModelCheckpoint

# Sample data (dummy data for illustration)
data = pd.read_csv('path_to_your_file.csv')

X1_linear = data['x1'].values
X_poly = data[['x2', 'x3', 'x4']].values
Y = data[['y1', 'y2']].values

# Define input layers
input_linear = Input(shape=(1,), name="linear_input")
input_poly = Input(shape=(3,), name="poly_input")

# Linear relationship for x1
linear_output = Dense(2, activation='linear', name="linear_output")(input_linear)

# Polynomial relationship for x2, x3, x4
poly_hidden = Dense(64, activation='relu')(input_poly)
poly_hidden2 = Dense(32, activation='relu')(poly_hidden)
poly_output = Dense(2, activation='linear', name="poly_output")(poly_hidden2)

# Combine linear and polynomial outputs
combined_output = Concatenate()([linear_output, poly_output])

# Final output layer for y1 and y2
final_output = Dense(2, activation='linear')(combined_output)

# Construct the model
model = Model(inputs=[input_linear, input_poly], outputs=final_output)

model.compile(optimizer='adam', loss='mse')  # mean squared error for regression

filepath = "model_at_epoch_{epoch:02d}.h5"
checkpoint = ModelCheckpoint(filepath, period=10)  # Save every 10 epochs


model.fit([X1_linear, X_poly], Y, epochs=100, verbose=1, callbacks=[checkpoint])

