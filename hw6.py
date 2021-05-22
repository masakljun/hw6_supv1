import numpy as np
import csv
import time
from sklearn.preprocessing import StandardScaler
from scipy.optimize import fmin_l_bfgs_b
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import pandas as pd
import sys
sys.path.append("../HW2")
sys.path.append("../HW4")
from hw_svr import SVR, RBF
from main import MultinomialLogReg

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
        if self.verify_gradient(wb, 1e-06, 1e-02):
            print("Verified: the gradient and cost are compatible")
        else:
            print("Warning: the gradient and cost are NOT compatible")

        wb_new, u, v = fmin_l_bfgs_b(func = self.feedforward, x0 = wb, fprime = self.backpropagation, maxiter = 1000)

        w, b = from1D(wb_new, self.layer_lengths)
        self.weights_ = w
        self.bias_ = b

    def feedforward(self, weights, info = False, X_new = None):
        #print(add)
        weights, bias = from1D(weights, self.layer_lengths)
        #print(f"W: {weights}")
        #print(f"b: {bias}")

        # regularize weights
        bias0 = [np.zeros(b.shape) for b in bias]
        weights_reg = to1D(weights, bias0)
        add = (self.lambda_ / 2) * np.sum((np.array(weights_reg) ** 2))

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

        if X_new is None:
            if self.type == "reg":
                loss = mse(self.y, act)
            else: # type = "clas"
                loss = cross_entropy(self.y, act)
        else:
            # do not compute loss for predictions
            loss, add = 0, 0

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


def read_csv_reg(fn):
    content = list(csv.reader(open(fn, "rt")))
    legend = content[0][:-1]
    data = content[1:]
    X = np.array([d[:-1] for d in data], dtype=np.float)
    y = np.array([d[-1] for d in data], dtype=np.float)
    return legend, X, y

def read_csv_clas(fn):
    content = list(csv.reader(open(fn, "rt")))
    legend = content[0][:-1]
    data = content[1:]
    X = np.array([d[:-1] for d in data], dtype=np.float)
    y = np.array([d[-1] for d in data], dtype=np.str)
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


def housing2r(fn):
    legend, X, y = read_csv_reg(fn)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, shuffle=True)

    X_train = StandardScaler().fit_transform(X_train)
    X_test = StandardScaler().fit_transform(X_test)

    # grid search
    lambda_candidates = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1]
    units_candidates = [[], [10], [5, 10], [10, 10], [10, 20], [20, 20]]

    best_lambda, best_units, _ = grid_search(lambda_candidates, units_candidates, X_train, y_train, t="reg")
    best_model = ANNRegression(best_units, best_lambda)
    fitted = best_model.fit(X_train, y_train)
    best_res = mse(y_test, fitted.predict(X_test).T)
    print(f"[ANN] The best MSE is: {best_res}. It was obtained with units: {best_units} and lambda: {best_lambda}")

    fitter = SVR(kernel=RBF(sigma=5), lambda_=1, epsilon=1)
    model = fitter.fit(X_train, y_train)
    pred = model.predict(X_test)
    svr_res = mse(y_test, pred)
    print(f"[SVR w/ RBF] The MSE is: {svr_res}")


def housing3(fn):
    legend, X, y = read_csv_clas(fn)
    y = np.unique(y, return_inverse = True)[1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0, shuffle= True)

    X_train = StandardScaler().fit_transform(X_train)
    X_test = StandardScaler().fit_transform(X_test)


    # grid search
    lambda_candidates = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1]
    units_candidates = [[], [5], [10], [8, 8], [5, 10], [10, 15], [20, 20]]

    best_lambda, best_units, _ = grid_search(lambda_candidates, units_candidates, X_train, y_train, t = "clas")
    best_model = ANNClassification(best_units, best_lambda)
    fitted = best_model.fit(X_train, y_train)
    best_res = cross_entropy(y_test, fitted.predict(X_test).T)
    print(f"[ANN] The best cross entropy score is: {best_res}. It was obtained with units: {best_units} and lambda: {best_lambda}")

    fitter = MultinomialLogReg()
    model = fitter.build(X_train, y_train)
    pred = model.predict1(X_test)
    mlr_res = cross_entropy(y_test, pred.T)
    print(f"[Multinomial log.reg.] The cross entropy is: {mlr_res}")


def grid_search(lambda_candidates, units_candidates, X_train, y_train, t):
    best_res = np.inf
    best_lambda = 0
    best_units = 0
    for l in lambda_candidates:
        for u in units_candidates:
            print(f"units {u}, lambda {l}")
            res_cv = []
            kf = KFold(n_splits=5, random_state=None, shuffle=False)
            for train_index, test_index in kf.split(X_train):
                X_train_k, X_test_k = X_train[train_index, :], X_train[test_index, :]
                y_train_k, y_test_k = y_train[train_index], y_train[test_index]
                if t == "clas":
                    model = ANNClassification(u, l)
                    fitted = model.fit(X_train_k, y_train_k)
                    res_cv.append(cross_entropy(y_test_k, fitted.predict(X_test_k).T))

                else: #t = reg
                    model = ANNRegression(u, l)
                    fitted = model.fit(X_train_k, y_train_k)
                    res_cv.append(mse(y_test_k, fitted.predict(X_test_k)))

            # check if the current setting of parameters is better then the best so far
            if np.average(res_cv) < best_res:
                best_res = np.average(res_cv)
                best_lambda = l
                best_units = u
    return best_lambda, best_units, best_res

def read_final_data():
    # read train data
    df = pd.read_csv('train.csv.gz', sep=',', index_col="id")
    df = df.to_numpy()

    X = df[:, :-1]
    y = df[:, -1]

    # convert to integers
    y = np.array(to_int(y))

    # shuffle
    np.random.seed(0)
    idx = list(range(len(y)))
    np.random.shuffle(idx)

    X = X[idx, :]
    y = y[idx]
    X = X.astype(np.float64)
    y = y.astype(np.int)

    # read test data
    df1 = pd.read_csv('test.csv.gz', sep=',', index_col="id")
    X_test = df1.to_numpy()

    X = StandardScaler().fit_transform(X)
    X_test = StandardScaler().fit_transform(X_test)

    return X, y, X_test

def create_final_predictions():
    X, y, X_test = read_final_data()


    # grid search
    lambda_candidates = [0.0001, 0.001, 0.01, 0.1]
    units_candidates = [[], [5, 18], [5,10,18], [10,10,10]]

    start_time = time.time()
    best_lambda, best_units, best_res = grid_search(lambda_candidates, units_candidates, X, y, t="clas")
    end_time_cv = time.time() - start_time

    # fit the model
    start_time = time.time()
    model = ANNClassification(best_units, best_lambda)
    fitted_model = model.fit(X, y)
    end_time_fit = time.time() - start_time


    # create final predictions
    start_time = time.time()
    final_predictions = fitted_model.predict(X_test)
    end_time_pred = time.time() - start_time

    df_result = pd.DataFrame(final_predictions, columns= ['Class_1', 'Class_2', 'Class_3', 'Class_4','Class_5', 'Class_6', 'Class_7', 'Class_8', 'Class_9'])
    df_result.index += 1
    df_result.to_csv("final.txt", index_label= "id", sep=",")

    print(f"Best lambda: {best_lambda}")
    print(f"Best units: {best_units}")
    print(f"Best res: {best_res}")
    print(f"CV time: {end_time_cv}")
    print(f"Fit time:  {end_time_fit}")
    print(f"Predict time: {end_time_pred}")

    # predict on train
    train_predictions = fitted_model.predict(X)
    train_res = cross_entropy(y, train_predictions.T)
    print(f"Train prediction: {train_res}")


def to_int(y):
    dictio = {'Class_1': 0, 'Class_2': 1, 'Class_3': 2, 'Class_4': 3, 'Class_5': 4, 'Class_6': 5, 'Class_7': 6, 'Class_8': 7, 'Class_9': 8}
    num_y = [dictio[i] for i in y]
    return num_y


if __name__ == "__main__":
    ### problem 3
    #housing3("housing3.csv")
    #housing2r("housing2r.csv")

    ### problem 4
    create_final_predictions()
    #X, y, _ = read_final_data()
    #size = int(0.7 * X.shape[0])

    #X_train = X[:size, :]
    #y_train = y[:size]
    #X_test = X[size:, :]
    #y_test = y[size:]

    #model = ANNClassification([5, 18],0.001)
    #fitted = model.fit(X_train, y_train)
    #res = cross_entropy(y_test, fitted.predict(X_test).T)
    #print( f"[ANN] The cross entropy score is: {res}.")


