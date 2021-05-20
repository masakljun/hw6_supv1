import numpy as np
import csv
from sklearn.preprocessing import StandardScaler
from scipy.optimize import fmin_l_bfgs_b

class ANNRegression:

    def __init__(self, units, lambda_):
        self.units = units
        self.lambda_ = lambda_

    def fit(self, X, y):
        model = ANN(type = "reg", units = self.units, X = X, y = y, lambda_= self.lambda_)
        model.fit()
        return model



class ANNClassification:

    def __init__(self, units, lambda_):
        self.units = units
        self.lambda_ = lambda_

    def fit(self, X, y):
        model = ANN(type = "clas", units = self.units, X = X, y = y, lambda_= self.lambda_)
        model.fit()
        return model

class ANN:
    def __init__(self, type, units, X, y, lambda_):
        self.type = type
        self.units = units
        self.X = X
        self.y = y
        self.lambda_ = lambda_

        self.weights_ = []
        self.bias_ = []

        ## initialize weights and bias
        if self.type == "reg":
            outputs = 1
        else:
            outputs = len(np.unique(self.y))

        # weights and biases
        num_features = X.shape[1]

        np.random.seed(0)
        # no hidden layers
        if len(self.units) == 0:
            self.weights_.append(np.random.rand(outputs, num_features))
            self.bias_.append(np.random.rand(outputs, 1))
        else: # hidden layers
            # weights
            self.weights_.append(np.random.rand(self.units[0], num_features))
            for i in range(0, len(self.units)-1):
                self.weights_.append(np.random.rand(self.units[i+1], self.units[i]))
            self.weights_.append(np.random.rand(outputs, self.units[-1]))

            # bias
            for i in range(0, len(self.units)):
                self.bias_.append(np.random.rand(self.units[i], 1))
            self.bias_.append(np.random.rand(outputs, 1))

        # layers lengths
        self.layer_lengths = [num_features, *self.units, outputs]

    def fit(self):
        wb = to1D(self.weights_, self.bias_)
        if self.verify_gradient(wb, 1e-5, 1e-3):
            print("Verified: the gradient and cost are compatible")
        else:
            print("Warning: the gradient and cost are NOT compatible")

        wb_new, u, v = fmin_l_bfgs_b(func = self.feedforward, x0 = wb, fprime = self.backpropagation, maxiter = 1000)

        w, b = from1D(wb_new, self.layer_lengths)
        self.weights_ = w
        self.bias_ = b

    def feedforward(self, weights, info = False, X_new = None):
        # regularize weights
        add = self.lambda_ / np.sum(2 * (np.array(weights) ** 2))
        #print(add)
        weights, bias = from1D(weights, self.layer_lengths)
        #print(f"W: {weights}")
        #print(f"b: {bias}")

        activations = []
        zs = []

        if X_new is None:
            act = self.X.T
        else:
            act = X_new.T

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

        if self.type == "reg":
            loss = mse(self.y, act)
        else: # type = "clas"
            loss = cross_entropy(self.y, act)


        if info:
            return zs, activations, loss+add
        else:
            return loss+add


    def backpropagation(self, weights):
        zs, activations, loss = self.feedforward(weights, True)
        w, b = from1D(weights, self.layer_lengths)

        wgs = [None] * len(w)    # weight gradients
        bgs = [None] * len(w)
        ds = [None] * len(w)    # deltas

        # first step
        if self.type == "reg":
            ds[-1] = 2 * (activations[-1] - self.y.flatten())/len(self.y.flatten())
        else: # type = clas
            ds[-1] = activations[-1]
            ds[-1][self.y.flatten(), range(len(self.y.flatten()))] -= 1
            ds[-1] = ds[-1] / len(self.y.flatten())
        wgs[-1] = ds[-1].dot(activations[-2].T)
        bgs[-1] = np.sum(ds[-1], axis = 1)

        # other steps
        lvl = len(w)-2

        while lvl >= 0:
            #print(lvl)
            sigmoid_der = activations[lvl+1] * (1. - activations[lvl+1])
            #print(f"ds+1: {w[lvl].shape}")
            ds[lvl] = sigmoid_der * ds[lvl+1].T.dot(w[lvl+1]).T
            wgs[lvl] = ds[lvl].dot(activations[lvl].T)
            bgs[lvl] = np.sum(ds[lvl], axis = 1)
            lvl = lvl - 1

        # regularize weights
        for i in range(len(wgs)):
            wgs[i] = wgs[i] + self.lambda_ * w[i]



        #print(ds)
        #print(wgs)
        #print(bgs)
        return to1D(wgs, bgs)

    # Verify gradient
    def verify_gradient(self, weights, h, threshold):
        # always the same value
        same = self.feedforward(weights, False)
        # value for comparison
        comp = self.backpropagation(weights)

        for i in range(0, len(weights)):
            # change one weight
            weights[i] = weights[i] + h
            value = self.feedforward(weights, False)
            # compute the difference
            diff = (value - same) / h

            # check whether the difference between comp[i] and diff exceed the threshold
            if abs(diff - comp[i]) >= threshold:
                return False

            # correct weight to be same as before
            weights[i] = weights[i] - h
        return True

    def predict(self, X):
        wb = to1D(self.weights_, self.bias_)
        z, a, l = self.feedforward(wb, info = True, X_new=X)
        predictions = a[-1].T

        if self.type == "reg":
            predictions = predictions.flatten()

        return predictions

    def weights(self):
        w = []
        for i in range(len(self.weights_)):
            w.append(np.c_[self.weights_[i], self.bias_[i]].T)
        return w


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


def housing(X, y, type):
    #if type == "reg":

    pass
    #else:


if __name__ == "__main__":
    pass
    #legend, X, y = read_csv("housing2r.csv")
    #X = StandardScaler().fit_transform(X)

    #test = ANN("reg", units = [3], X = X, y = y)
    #print(test.feedforward(X[0:10, :]))

    #X = np.array([[0, 0, 1],
      # [1, 1, 1],
      # [1, 0, 1],
      # [0, 1, 1]])
    #y = np.array([0, 1, 1, 0])


    #weights = np.array([[ 0.93405968,  0.0944645 ],
     #  [ 0.94536872,  0.42963199],
      # [ 0.39545765, -0.56782101],
       #[ 0.95254891, -0.98753949]])

    #np.random.seed(1)
    #weights = np.random.rand(4,1)

    #weights = weights.T
    #biases = [weights[:, -1]]
    #weights = [weights[:, :-1]]
    #np.random.seed(1)
    #weights = [np.random.rand(2,4), np.random.rand(2,3)]
    #biases = [w[:, -1] for w in weights]
    #weights = [w[:, :-1] for w in weights]

    #print(biases)

    #w = to1D(weights, biases)

    #test = ANN("reg", units = [], X = X, y = y, lambda_= 0.0)
    #print(test.backpropagation(w))
    #print(test.verify_gradient(w, 1e-5, 1e-3))

