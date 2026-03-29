import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

cols = ["fLength","fWidth","fSize","fConc","fConc1","fAsym","fM3Long","fM3Trans","fAlpha","fDist","class"]

df = pd.read_csv("magic04.data", names=cols)
df["class"] = (df["class"] == "g").astype(int)

train, val, split = np.split(df, [int(0.6*len(df)), int(0.8*len(df))])

train = pd.DataFrame(train, columns = cols)

train_X = train[["fLength","fWidth","fSize","fConc","fConc1","fAsym","fM3Long","fM3Trans","fAlpha","fDist"]].values
train_Y = train["class"].values

m,n = train_X.shape
epsilon = 1e-15

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def cost_func(X, Y, w, b):
    cost_sum = 0
    for i in range(m):
        z = np.dot(w, X[i]) + b
        g = sigmoid(z)
        cost_sum += -Y[i]*np.log(g+epsilon) - (1-Y[i])*np.log(1-g+epsilon)

    return (1/m)*cost_sum

def gradient(X, Y, w, b):
    grad_w = np.zeros(n)
    grad_b = 0
    for i in range(m):
        z = np.dot(w, X[i]) + b
        g = sigmoid(z)
        grad_b += (g - Y[i])
        for j in range(n):
            grad_w[j] += (g - Y[i])*X[i,j]

    grad_w = (1/m)*grad_w
    grad_b = (1/m)*grad_b
    return grad_w, grad_b

def gradient_descent(X, Y, alpha, epochs):
    w = np.zeros(n)
    b = 0
    for i in range(epochs):
        grad_w, grad_b = gradient(X, Y, w, b)
        w = w - alpha*grad_w
        b = b - alpha*grad_b
        if (i % 1000 == 0):
            print(f"Epoch: {i}")
            print(f"Cost: {cost_func(X, Y, w, b)}")
    return w,b

def predict(X, w, b):
    preds = np.zeros(m)
    for i in range(m):
        z = np.dot(w, X[i]) + b
        g = sigmoid(z)
        if  (g >= 0.5):
            preds[i] = 1
        else:
            preds[i] = 0

    return preds

learning_rate = 0.01
epochs = 10000

final_w, final_b = gradient_descent(train_X, train_Y, learning_rate, epochs)

predictions = predict(train_X, final_w, final_b)
accuracy = np.mean(predictions == train_Y)*100
print(f"training accuracy: {accuracy:.2f}%")


