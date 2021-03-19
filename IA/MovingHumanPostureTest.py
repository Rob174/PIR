import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Input, Flatten

#Construction de l'input: un vecteur caractéristique 100x1, puis 25x1 puis 10x1

input = Input(shape=(400, 100,1)) # width, height, channel: 1 = black/white
layer = Flatten()(input) # to obtain a 400*1 vector

layer = Dense(units=100)(input)
layer = Dense(units=25)(layer)
layer = Dense(units=10)(layer)

model = Model(inputs=[input], outputs=[layer])

# SGD: Gradient Descent (with momentum) optimizer.
# MSE: Mean Squared Error
model.compile(optimizer='sgd', loss='mse')