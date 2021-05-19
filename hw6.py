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
        if self.type == "reg":
            outputs = 1
        else:
            outputs = len(np.unique(self.y))

        # weights and biases
        num_features = X.shape[1]

        # no hidden layers
        if len(self.units) == 0:
            self.weights.append(np.random.rand(outputs, num_features))
            self.bias.append(np.random.rand(outputs, 1))
        else: # hidden layers
            # weights
            self.weights.append(np.random.rand(self.units[0], num_features))
            for i in range(0, len(self.units)-1):
                self.weights.append(np.random.rand(self.units[i+1], self.units[i]))
            self.weights.append(np.random.rand(outputs, self.units[-1]))

            # bias
            for i in range(0, len(self.units)):
                self.bias.append(np.random.rand(self.units[i], 1))
            self.bias.append(np.random.rand(outputs, 1))

        # layers lenghts
        self.layer_lengths = [num_features, *self.units, outputs]

    def fit(self):

        pass
        # fmin_lbfgs(feedforward, wb, backprop, ...)

    def feedforward(self, weights, info = False):
        weights, bias = from1D(weights, self.layer_lengths)
        #print(f"W: {weights}")
        #print(f"b: {bias}")

        activations = []
        zs = []

        act = self.X.T
        activations.append(act)

        for i in range(0, len(weights)-1):
            z = np.dot(weights[i], act) + bias[i]
            act = sigmoid(z)
            activations.append(act)
            zs.append(z)

        z = np.dot(weights[-1], act) + bias[-1]
        zs.append(z)

        if self.type == "reg":
            act = z
        else: # type = "clas"
            act = softmax(z.T)
            # in the same shape as previous
            act = act.T

        activations.append(act)

        if type == "reg":
            loss = mse(y, act)
        else: # type = "clas"
            loss = cross_entropy(y, act)

        if info:
            return zs, activations, loss
        else:
            return loss


    def backpropagation(self, weights):
        zs, activations, loss = self.feedforward(weights, True)
        w, b = from1D(weights, self.layer_lengths)

        wgs = [None] * len(w)    # weight gradients
        bgs = [None] * len(w)
        ds = [None] * len(w)    # deltas

        # first step
        if self.type == "reg":
            ds[-1] = 2 * (activations[-1] - y.flatten())/len(y.flatten())
        else:
            ds[-1] = activations[-1]
            ds[-1][y.flatten(), range(len(y.flatten()))] -= 1
            ds[-1] = ds[-1] / len(y.flatten())
        wgs[-1] = ds[-1].dot(activations[-2].T)
        bgs[-1] = np.sum(ds[-1], axis = 1)

        # other steps
        lvl = len(w)-2

        while lvl >= 0:
            print(lvl)
            sigmoid_der = activations[lvl+1] * (1. - activations[lvl+1])
            print(f"ds+1: {w[lvl].shape}")
            ds[lvl] = sigmoid_der * ds[lvl+1].T.dot(w[lvl+1]).T
            wgs[lvl] = ds[lvl].dot(activations[lvl].T)
            bgs[lvl] = np.sum(ds[lvl], axis = 1)
            lvl = lvl - 1

        print(ds)
        print(wgs)
        print(bgs)

        #for i in range(2, len(self.units)-1):


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

# Mean squared error for regression
def mse(true, pred):
    res = np.square(np.subtract(true, pred)).mean()
    return res

# Cross entropy for classification
def cross_entropy(true, pred):
    el = np.choose(true, pred)
    log_like = -np.log(el)
    loss = np.sum(log_like)/len(true)
    return loss



# Put everything (biases, weights) in 1D vector
def to1D(weights, biases):
    res = []
    for w in weights:
        res.extend(w.flatten())
    for b in biases:
        res.extend(b.flatten())
    return res

# Put out of 1D vector back to "matrices"
def from1D(wb, layers):
    #[5, 3, 1]  #[3, 1]
    weights = []
    biases = []
    start = 0
    for i in range(1, len(layers)):
        finish = start + (layers[i] * layers[i-1])
        w = np.array(wb[start:finish]).reshape((layers[i], layers[i-1]))
        weights.append(w)
        start = finish

    for i in range(1, len(layers)):
        finish = start + layers[i]
        b = np.array(wb[start:finish]).reshape(layers[i], 1)
        biases.append(b)
        start = finish

    return weights, biases

if __name__ == "__main__":
    #legend, X, y = read_csv("housing2r.csv")
    #X = StandardScaler().fit_transform(X)

    #test = ANN("reg", units = [3], X = X, y = y)
    #print(test.feedforward(X[0:10, :]))

    X = np.array([[0, 0, 1],
       [1, 1, 1],
       [1, 0, 1],
       [0, 1, 1]])
    y = np.array([0, 1, 1, 0])


    #weights = np.array([[ 0.93405968,  0.0944645 ],
    #   [ 0.94536872,  0.42963199],
    #   [ 0.39545765, -0.56782101],
    #   [ 0.95254891, -0.98753949]])

    #weights = weights.T
    #biases = [weights[:, -1]]
    #weights = [weights[:, :-1]]
    np.random.seed(1)
    weights = [np.random.rand(2,4), np.random.rand(2,3)]
    biases = [w[:, -1] for w in weights]
    weights = [w[:, :-1] for w in weights]

    print(biases)

    w = to1D(weights, biases)

    test = ANN("clas", units = [2], X = X, y = y)
    print(test.backpropagation(w))
