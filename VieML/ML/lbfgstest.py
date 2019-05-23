import numpy as np
from scipy.optimize import minimize
from keras.models import Sequential
from keras.layers.core import Dense

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([[0], [0], [0], [1]])

model = Sequential()
model.add(Dense(4, activation='sigmoid', input_dim=2))
model.add(Dense(4, activation='sigmoid', input_dim=2))
model.add(Dense(1, activation='sigmoid', input_dim=2))

model.compile("Adam", "binary_crossentropy", metrics=["accuracy"])
# model.fit(X, Y, epochs=3000)

index = 0


def loss(W):
    global model, index
    weights = model.get_weights()
    convert(weights, W)
    index = 0
    model.set_weights(weights)
    error = model.evaluate(X, Y)[0]
    print(error)
    return error


def convert(weights, W):
    global index
    for i in range(len(weights)):
        e = weights[i]
        if isinstance(e, np.ndarray):
            convert(e, W)
        else:
            weights[i] = W[index]
            index += 1


tmp = np.array(model.get_weights())
x0 = []
for i in range(tmp.shape[0]):
    x0 = np.append(x0, tmp[i].flatten())
loss(x0)
res = minimize(loss, x0, method='L-BFGS-B', options={'eps': 1e-3, 'disp': True})
print(model.predict(X).round())
