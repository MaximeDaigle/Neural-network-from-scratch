import pickle
import numpy as np


class NN(object):
    def __init__(self,
                 hidden_dims=(512, 256),
                 datapath='cifar10.pkl',
                 n_classes=10,
                 epsilon=1e-6,
                 lr=7e-4,
                 batch_size=1000,
                 seed=None,
                 activation="relu",
                 init_method="glorot"
                 ):

        self.hidden_dims = hidden_dims
        self.n_hidden = len(hidden_dims)
        self.datapath = datapath
        self.n_classes = n_classes
        self.lr = lr
        self.batch_size = batch_size
        self.init_method = init_method
        self.seed = seed
        self.activation_str = activation
        self.epsilon = epsilon

        self.train_logs = {'train_accuracy': [], 'validation_accuracy': [], 'train_loss': [], 'validation_loss': []}

        if datapath is not None:
            u = pickle._Unpickler(open(datapath, 'rb'))
            u.encoding = 'latin1'
            self.train, self.valid, self.test = u.load()
        else:
            self.train, self.valid, self.test = None, None, None

    def get_weight(self,key):
        return self.weights[key].T

    def initialize_weights(self, dims):
        if self.seed is not None:
            np.random.seed(self.seed)

        self.weights = {}
        # self.weights is a dictionary with keys W1, b1, W2, b2, ..., Wm, Bm where m - 1 is the number of hidden layers
        all_dims = [dims[0]] + list(self.hidden_dims) + [dims[1]]
        for layer_n in range(1, self.n_hidden + 2):
            self.weights[f"W{layer_n}"] = np.random.uniform(-1 / (all_dims[layer_n-1] ** (0.5)), 1 / (all_dims[layer_n-1]  ** (0.5)), (all_dims[layer_n-1], all_dims[layer_n]))
            self.weights[f"b{layer_n}"] = np.zeros((1, all_dims[layer_n]))

    def relu(self, x, grad=False):
        if grad:
            return np.greater(x, 0).astype(float)
        return np.maximum(x, 0)

    def sigmoid(self, x, grad=False):
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        if grad:
            sig = self.sigmoid(x, False)
            return sig - sig ** 2
        return 1 / (1 + np.exp(-x))

    def tanh(self, x, grad=False):
        if grad:
            return 1 - np.tanh(x) ** 2
        return np.tanh(x)

    def activation(self, x, grad=False):
        if self.activation_str == "relu":
            return self.relu(x,grad)
        elif self.activation_str == "sigmoid":
            return self.sigmoid(x, grad)
        elif self.activation_str == "tanh":
            return self.tanh(x, grad)
        else:
            raise Exception("invalid")
        return 0

    def softmax(self, x):
        # x is a vector or batch_size × n_dimensions
        # Remember that softmax(x-C) = softmax(x) when C is a constant.
        max = np.max(x, axis=-1)  # find the max of each rows
        if len(x.shape) > 1:  # if x is batch_size × n_dimensions, add a dimension to broadcast substration
            max = max[:, np.newaxis]
        x = x - max
        x = np.exp(x)
        sum = np.sum(x, axis=-1)
        if len(x.shape) > 1:  # if x is batch_size × n_dimensions, add a dimension to broadcast division
            sum = sum[:, np.newaxis]
        x = x / sum
        return x

    def forward(self, x):
        cache = {"Z0": x}
        # cache is a dictionary with keys Z0, A0, ..., ZL, AL where L - 1 is the number of hidden layers
        # Ai corresponds to the preactivation at layer i, Zi corresponds to the activation at layer i
        for layer_n in range(1, self.n_hidden+1): #each hidden layers
            cache[f"A{layer_n}"] = self.weights[f"b{layer_n}"] + np.dot(cache[f"Z{layer_n-1}"], self.weights[f"W{layer_n}"])
            cache[f"Z{layer_n}"] = self.activation(cache[f"A{layer_n}"])
        #output layer
        cache[f"A{self.n_hidden+1}"] = self.weights[f"b{self.n_hidden+1}"] + np.dot(cache[f"Z{self.n_hidden}"], self.weights[f"W{self.n_hidden+1}"])
        cache[f"Z{self.n_hidden+1}"] = self.softmax(cache[f"A{self.n_hidden+1}"])
        return cache

    def backward(self, cache, labels):
        output = cache[f"Z{self.n_hidden + 1}"] #(nb samples, m)
        grads = {}
        if len(np.squeeze(cache["Z0"]).shape) == 1:
            nb_samples = 1
            cache["Z0"] = cache["Z0"][np.newaxis, :] #change shape (d,) to (1,d) so code below works for any nb_samples (even 1) the same way by using the standard shape (nb samples, d)
        else:
            nb_samples = cache["Z0"].shape[0]
        # grads is a dictionary with keys dAL, dWL, dbL, dZ(L-1), dA(L-1), ..., dW1, db1
        grads[f"dA{self.n_hidden + 1}"] = output - labels #(nb samples, m)
        grads[f"dW{self.n_hidden + 1}"] = np.dot(cache[f"Z{self.n_hidden}"].T, grads[f"dA{self.n_hidden + 1}"]) / nb_samples #correct? #mean where: np.dot(Z(L-1).T, dAL) where Z(L-1)=(nb samples, d_{L-1}), dAL=(nb samples, m) and result is (d_{L-1}, m)
        grads[f"db{self.n_hidden + 1}"] = np.mean(grads[f"dA{self.n_hidden + 1}"], axis=0)[np.newaxis, :] #average on samples of dAL=(nb samples, m)

        for layer_n in range(self.n_hidden, 0, -1):
            grads[f"dZ{layer_n}"] = np.dot(grads[f"dA{layer_n+1}"], self.weights[f"W{layer_n+1}"].T) # np.dot(dA(Layer+1), W(Layer+1).T) where dA(layer+1)=(nb samples, d_{Layer+1}), W(Layer+1)=(d_{Layer},d_{Layer+1}) and results=(nb samples, d_{Layer})
            grads[f"dA{layer_n}"] = np.multiply(self.activation(cache[f"A{layer_n}"],grad=True), grads[f"dZ{layer_n}"]) #element wise
            grads[f"dW{layer_n}"] = np.dot(cache[f"Z{layer_n-1}"].T, grads[f"dA{layer_n}"]) / nb_samples
            grads[f"db{layer_n}"] = np.mean(grads[f"dA{layer_n}"], axis=0)[np.newaxis, :] #add a dimension because gradescope check if it is the same shape. So, vectors of shape (n,) != (n,1) are rejected, but (1,n) is accepted even if they are all valid

        cache["Z0"] = np.squeeze(cache["Z0"]) #remove dimension added to compute grads
        return grads

    def update(self, grads):
        for layer in range(1, self.n_hidden + 2):
            self.weights[f"W{layer}"] -= self.lr * grads[f"dW{layer}"]
            self.weights[f"b{layer}"] -= self.lr * grads[f"db{layer}"]

    def one_hot(self, y):
        return np.eye(self.n_classes)[y]

    def loss(self, prediction, labels):
        prediction[np.where(prediction < self.epsilon)] = self.epsilon
        prediction[np.where(prediction > 1 - self.epsilon)] = 1 - self.epsilon
        return -np.sum(labels * np.log(prediction))/labels.shape[0]

    def compute_loss_and_accuracy(self, X, y):
        one_y = self.one_hot(y)
        cache = self.forward(X)
        predictions = np.argmax(cache[f"Z{self.n_hidden + 1}"], axis=1)
        accuracy = np.mean(y == predictions)
        loss = self.loss(cache[f"Z{self.n_hidden + 1}"], one_y)
        return loss, accuracy, predictions

    def train_loop(self, n_epochs):
        X_train, y_train = self.train
        y_onehot = self.one_hot(y_train)
        dims = [X_train.shape[1], y_onehot.shape[1]]
        self.initialize_weights(dims)

        n_batches = int(np.ceil(X_train.shape[0] / self.batch_size))

        for epoch in range(n_epochs):
            for batch in range(n_batches):
                minibatchX = X_train[self.batch_size * batch:self.batch_size * (batch + 1), :]
                minibatchY = y_onehot[self.batch_size * batch:self.batch_size * (batch + 1), :]

                cache = self.forward(minibatchX)
                grads = self.backward(cache, minibatchY)
                self.update(grads)

            X_train, y_train = self.train
            train_loss, train_accuracy, _ = self.compute_loss_and_accuracy(X_train, y_train)
            X_valid, y_valid = self.valid
            valid_loss, valid_accuracy, _ = self.compute_loss_and_accuracy(X_valid, y_valid)

            self.train_logs['train_accuracy'].append(train_accuracy)
            self.train_logs['validation_accuracy'].append(valid_accuracy)
            self.train_logs['train_loss'].append(train_loss)
            self.train_logs['validation_loss'].append(valid_loss)

        return self.train_logs

    def evaluate(self):
        X_test, y_test = self.test
        return self.compute_loss_and_accuracy(X_test, y_test)[0:2]

if __name__ == "__main__":
    model = NN(hidden_dims=(512, 256),
                 datapath='cifar10.pkl',
                 lr=0.003,
                 batch_size=100,
                 seed=0
                 )
    train_logs = model.train_loop(n_epochs=50)
    for k in train_logs:
        print(k, train_logs[k])
    import matplotlib.pyplot as plt

    plt.plot(list(range(50)), train_logs['validation_accuracy'], label='Validation')
    plt.plot(list(range(50)), train_logs['train_accuracy'], label='Training')
    plt.legend()
    plt.ylabel('Accuracy')
    plt.xlabel('epochs')
    plt.show()

    plt.plot(list(range(50)), train_logs['validation_loss'], label='Validation')
    plt.plot(list(range(50)), train_logs['train_loss'], label='Training')
    plt.legend()
    plt.ylabel('Loss')
    plt.xlabel('epochs')
    plt.show()