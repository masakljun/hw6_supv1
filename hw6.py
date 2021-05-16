import numpy as np
import csv
from sklearn.preprocessing import StandardScaler

class ANNRegression:

    def __init__(self, units, lambda_):
        self.units = units
        self.lambda_ = lambda_


class ANNClassification:

    def __init__(self, units, lambda_):
        self.units = units
        self.lambda_ = lambda_



class ANN:
    def __init__(self, type, units, X, y):
        self.type = type
        self.units = units
        self.X = X
        self.y = y

        self.weights = []
        self.bias = []

        ## initialize weights and bias
        # weights
        num_features = X.shape[1]
        self.weights.append(np.random.rand(self.units[0], num_features))
        for i in range(0, len(self.units)-1):
            self.weights.append(np.random.rand(self.units[i+1], self.units[i]))

        if self.type == "reg":
            self.weights.append(np.random.rand(1, self.units[-1]))
        else: # type = "clas"
            outputs = len(np.unique(self.y))
            self.weights.append(np.random.rand(outputs, self.units[-1]))

        # bias
        for i in range(0, len(self.units)):
            self.bias.append(np.random.rand(self.units[i], 1))
        if self.type == "reg":
            self.bias.append(np.random.rand(1, 1))
        else: # type = "clas"
            outputs = len(np.unique(self.y, 1))
            self.bias.append(np.random.rand(outputs))


    def feedforward(self, act):
        act = act.T
        for i in range(0, len(self.weights)):
            z = np.dot(self.weights[i], act) + self.bias[i]
            #print(f"{i}: {z}")
            act = sigmoid(z)
            #print(f"{i}: {act}")

        act = act.T

        if self.type == "reg":
            act = act
        else: # type = "clas"
            act = softmax(act)

        return act

def read_csv(fn):
    content = list(csv.reader(open(fn, "rt")))
    legend = content[0][:-1]
    data = content[1:]
    X = np.array([d[:-1] for d in data], dtype=np.float)
    y = np.array([d[-1] for d in data], dtype=np.float)
    return legend, X, y


# Sigmoid function
def sigmoid(x):
    res = 1.0/(1.0 + np.exp(-x))
    return res

# Softmax function
def softmax(x):
    res = np.exp(x) / (np.sum(np.exp(x), axis=1, keepdims= True))
    return res

if __name__ == "__main__":
    legend, X, y = read_csv("housing2r.csv")
    X = StandardScaler().fit_transform(X)

    test = ANN("reg", units = [3], X = X, y = y)
    print(test.feedforward(X[0:10, :]))