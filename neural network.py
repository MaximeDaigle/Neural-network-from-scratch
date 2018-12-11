import numpy as np
import matplotlib.pyplot as plt

np.random.seed(3)

#remove error when divide 0 by 0 when doing the finite difference gradient check
np.seterr(divide='ignore', invalid='ignore')


circles = np.loadtxt('circles.txt')

n_classes = 2
n_train = 880
target_ind = [circles.shape[1] - 1]
inds = np.arange(circles.shape[0])
np.random.shuffle(inds)
train_inds = inds[:n_train]
test_inds = inds[n_train:]
train_set = circles[train_inds, :]
test_set = circles[test_inds, :]
train_inputs = train_set[:,:-1]
train_labels = train_set[:,-1]
test_inputs = test_set[:,:-1]
test_labels = test_set[:,-1]



class Neural_network:

    def __init__(self, d, dh, m, epoch = 100, step_size= 0.01, k = 1, lambda11 = 0, lambda12 = 0, lambda21 = 0, lambda22 = 0):
        self.d = d # number of features
        self.dh = dh # number of neurons in the hidden layer
        self.m = m # number of class = number of neurons in the last layer (output layer)
        self.step_size = step_size
        self.epoch = epoch
        self.b1 = np.zeros((dh,1))
        self.w1 = initialize_weight(d,dh)
        self.b2 = np.zeros((m,1))
        self.w2 = initialize_weight(dh,m)
        self.minibatch_size = k  # size of minibatch
        self.lambda11 = lambda11
        self.lambda12 = lambda12
        self.lambda21 = lambda21
        self.lambda22 = lambda22

    def weight_decay(self,w):
        if w == 'w1':
            return self.lambda11*np.sign(self.w1) + 2*self.lambda12*self.w1
        elif w == 'w2':
            return self.lambda21*np.sign(self.w2) + 2*self.lambda22*self.w2

    def fprop(self,x,y = None):
        self.ha = self.b1 + np.dot(self.w1,x)
        self.hs = self.ha.clip(min=0)
        self.oa = self.b2 + np.dot(self.w2,self.hs)
        self.os = softmax(self.oa)

    def bprop(self,x,y):
        grad_b2 = self.os - np.eye(self.m)[y].reshape((self.m,1))
        grad_w2 = np.outer(self.os, self.hs)
        for i in range(self.hs.shape[0]):
            grad_w2[y][i] = grad_w2[y][i] - self.hs[i]
        grad_hs = np.dot(self.w2.T,self.os) - self.w2[y,:].reshape((self.dh,1))
        indicator = np.sign(self.hs).clip(min=0)
        grad_ha = np.multiply(grad_hs,indicator)
        grad_b1 = grad_ha
        grad_w1 = np.outer(grad_ha,x)
        return np.array([grad_b1, grad_w1, grad_b2, grad_w2])

    def gradient_descent(self,grad_b1, grad_w1, grad_b2, grad_w2):
        self.w1 = self.w1 - self.step_size*(grad_w1 + self.weight_decay('w1'))
        self.b1 = self.b1 - self.step_size*grad_b1
        self.w2 = self.w2 - self.step_size * (grad_w2 + self.weight_decay('w2'))
        self.b2 = self.b2 - self.step_size * grad_b2

    # only use to debug the gradient_descent and back prop
    def gradient_check(self,x,y, debug = False):
        # use two-sided difference estimate instead of the one-sided difference
        epsilon = 0.00001
        y = int(y)
        x = x.reshape((self.d, 1))

        self.fprop(x,y)
        computed = self.bprop(x,y)

        #gradient check for b1
        grad_b1_check = np.zeros((self.dh,1))
        for i in range(self.b1.shape[0]):
            b1plus = np.copy(self.b1)
            b1plus[i] = self.b1[i] + epsilon
            b1minus = np.copy(self.b1)
            b1minus[i] = self.b1[i] - epsilon
            grad_b1_check[i] = (self.loss_check(x=x, y=y, b1=b1plus) - self.loss_check(x=x, y=y, b1=b1minus)) / (2*epsilon)

        # gradient check for b2
        grad_b2_check = np.zeros(self.b2.shape)
        for i in range(self.b2.shape[0]):
            b2plus = np.copy(self.b2)
            b2plus[i] = self.b2[i] + epsilon
            b2minus = np.copy(self.b2)
            b2minus[i] = self.b2[i] - epsilon
            grad_b2_check[i] = (self.loss_check(x=x, y=y, b2=b2plus) - self.loss_check(x=x, y=y, b2=b2minus)) / (2 * epsilon)

        #gradient check for w1
        grad_w1_check = np.zeros(self.w1.shape)
        for i in range(self.w1.shape[0]):
            for j in range(self.w1.shape[1]):
                w1plus = np.copy(self.w1)
                w1plus[i,j] = self.w1[i,j] + epsilon
                w1minus = np.copy(self.w1)
                w1minus[i,j] = self.w1[i,j] - epsilon
                grad_w1_check[i,j] = (self.loss_check(x=x, y=y, w1=w1plus) - self.loss_check(x=x, y=y, w1=w1minus)) / (2*epsilon)

        # gradient check for w2
        grad_w2_check = np.zeros(self.w2.shape)
        for i in range(self.w2.shape[0]):
            for j in range(self.w2.shape[1]):
                w2plus = np.copy(self.w2)
                w2plus[i, j] = self.w2[i, j] + epsilon
                w2minus = np.copy(self.w2)
                w2minus[i, j] = self.w2[i, j] - epsilon
                grad_w2_check[i, j] = (self.loss_check(x=x, y=y, w2=w2plus) - self.loss_check(x=x, y=y, w2=w2minus)) / (2*epsilon)

        grads = np.array([grad_b1_check, grad_w1_check, grad_b2_check, grad_w2_check])

        if debug:
            show_grads(computed=computed, estimate=grads)

        return grads

    def loss_check(self, x, y, b1 = None, b2 = None, w1 = None, w2 = None):
        if b1 is None:
            b1 = self.b1
        if b2 is None:
            b2 = self.b2
        if w1 is None:
            w1 = self.w1
        if w2 is None:
            w2 = self.w2
        ha = b1 + np.dot(w1, x)
        hs = ha.clip(min=0)
        oa = b2 + np.dot(w2, hs)
        os = softmax(oa)
        return - np.log(os[y])

    def gradient_check_minibatch(self, minibatch_inputs,minibatch_labels):
        computed = self.train_minibatch(minibatch_inputs, minibatch_labels.astype(int), True)

        total_b1 = np.zeros((self.dh, 1))
        total_w1 = np.zeros((self.dh, self.d))
        total_b2 = np.zeros((self.m, 1))
        total_w2 = np.zeros((self.m, self.dh))
        grads = np.array([total_b1, total_w1, total_b2, total_w2])

        # sum gradient from the k examples
        for i in range(minibatch_inputs.shape[0]):
            grads += self.gradient_check(x=minibatch_inputs[i], y=minibatch_labels[i])

        # use average estimated gradients of the minibatch
        grads = grads / minibatch_inputs.shape[0]

        show_grads(computed = computed, estimate = grads)

    def train(self,train_inputs, train_labels):
        train_labels = train_labels.astype(int)
        for l in range(self.epoch):
            #separate the dataset in minibatch with k examples
            for i in range(train_inputs.shape[0]//self.minibatch_size):
                s = i * self.minibatch_size #start index of the current minibatch
                e = (i + 1) * self.minibatch_size #end index of the current minibatch
                self.train_minibatch( minibatch_inputs = train_inputs[s:e, :], minibatch_labels  = train_labels[s:e])

            #if the numbers of examples in the trainning dataset is not a multiple of minibatch_size
            # take care of the last minibatch that cant have minibatch_size examples
            if (train_inputs.shape[0] % self.minibatch_size) != 0:
                s = (train_inputs.shape[0] // self.minibatch_size) * self.minibatch_size
                self.train_minibatch(minibatch_inputs = train_inputs[s:, :], minibatch_labels = train_labels[s:])

    def train_minibatch(self, minibatch_inputs, minibatch_labels, debug = False):
        total_b1 = np.zeros((self.dh,1))
        total_w1 = np.zeros((self.dh,self.d))
        total_b2 = np.zeros((self.m,1))
        total_w2 = np.zeros((self.m,self.dh))
        grads = np.array([total_b1, total_w1, total_b2, total_w2])

        # sum gradient from the k examples
        for j in range(minibatch_inputs.shape[0]):
            input = minibatch_inputs[j].reshape((self.d,1))
            self.fprop(input, minibatch_labels[j])
            grads += self.bprop(input, minibatch_labels[j])

        # use average gradients of the minibatch
        grads = grads / minibatch_inputs.shape[0]
        if debug:
            return grads
        self.gradient_descent(grads[0], grads[1], grads[2], grads[3])

    def predict(self, x):
        x =  x.reshape((self.d, 1))
        self.fprop(x)
        # print(self.os, y)
        return float(np.argmax(self.os))

    def score(self,test_inputs, test_labels):
        # Calcul du pourcentage d'exemple qui se fait correctement classifier
        correct = 0
        for i in range(test_inputs.shape[0]):
            if self.predict(test_inputs[i]) == test_labels[i]:
                correct = correct + 1
        result = 'Taux de classification correcte: ' + str((correct / test_inputs.shape[0]) * 100) + '%'
        return result

def show_grads(computed,estimate):
    s = ['grad_b1', 'grad_w1', 'grad_b2', 'grad_w2']
    for j in range(4):
        print("ratio computed " + s[j] + " / estimation " + s[j])
        print(computed[j] / estimate[j])
        print('computed ' + s[j])
        print(computed[j])
        print(s[j])
        print(estimate[j])
        print()

def softmax(a):
    """
    numerically stable softmax
    :param a: vector dx1 or matrix dxn containing n vectors of length d
    :return: vector dx1 or matrix dxn
    """
    # find the max for each column
    max = np.max(a, axis=0)
    a = a - max
    a = np.exp(a)
    sum = np.sum(a, axis=0)
    a = a / sum
    return a

def initialize_weight(n_input,n2):
    """
    initialize weights of a layer with an uniform distribution [-1/n_input,1/n_input]
    n_input = number of inputs for the layer
    (bias is initialized to 0, so it's not initialize with this function)
    :param n_input: number of columns = number of neurons in the layer before = number of neurons that give inputs
    :param n2: number of rows = number of neurons in the layer 'after' or current layer = number of neurons that receive inputs
    :return: weights matrix
    """
    n_input_sqrt = n_input**(0.5)
    return np.random.uniform(-1/n_input_sqrt,1/n_input_sqrt,(n2,n_input))

#decision regions
def plot_decision_region(model, params, n=50):
    fig = plt.figure()
    n=n
    x = np.linspace(-1,1,n)
    y = np.linspace(-1,1,n)
    xx, yy = np.meshgrid(x,y)
    grid_dataset = np.column_stack((xx.ravel(),yy.ravel()))
    pred = []
    for e in grid_dataset:
        pred.append(model.predict(e))
    pred = np.array(pred)
    pred = pred.reshape(xx.shape)
    plt.contourf(xx,yy,pred)
    fig.suptitle('decision region')
    plt.title(params)
    plt.show()



model = Neural_network(2,10,2, epoch=10, step_size=0.01, lambda11=0,lambda12=0,lambda21=0,lambda22=0)
model.train(train_inputs, train_labels)


print('Finite difference gradient check for 1 example')
model_check_gradient = Neural_network(2,3,2, epoch=10, step_size=0.01, lambda11=0,lambda12=0,lambda21=0,lambda22=0)
model_check_gradient.train(train_inputs, train_labels)
result_gradient_check = model_check_gradient.gradient_check(test_inputs[0],test_labels[0], True)
print()

# Give the average estimated gradient on 10 examples, the average computed gradient on the same examples
# and the ratio (computed average / estimation average)
print('Finite difference gradient check for 10 examples')
model_check_minibatch_gradient = Neural_network(2,3,2, epoch=10, k=10, step_size=0.01, lambda11=0,lambda12=0,lambda21=0,lambda22=0)
model_check_minibatch_gradient.train(train_inputs,train_labels)
minibatch_inputs = train_inputs[0:10]
minibatch_labels = train_labels[0:10]

model_check_minibatch_gradient.gradient_check_minibatch(minibatch_inputs,minibatch_labels)

# 5)
print('Model 1: 10 hidden neurons, 10 epoch, step size = 0.01')
print(model.score(test_inputs, test_labels))
print()
plot_decision_region(model, '10 hiddens, 10 epoch, step size = 0.01')

model2 = Neural_network(2,4,2, epoch=10, step_size=0.01, lambda11=0,lambda12=0,lambda21=0,lambda22=0)
model2.train(train_inputs, train_labels)
print('Model 2: 4 hidden neurons, 10 epoch, step size = 0.01')
print(model2.score(test_inputs, test_labels))
print()
plot_decision_region(model2, '4 hidden neurons')

model3 = Neural_network(2,10,2, epoch=10, step_size=0.01, lambda11=0.01,lambda12=0,lambda21=0,lambda22=0)
model3.train(train_inputs, train_labels)
print('Model 3: 10 hidden neurons, 10 epoch, step size = 0.01, lambda11 = 0.01')
print(model3.score(test_inputs, test_labels))
print()
plot_decision_region(model3, '10 hidden neurons, lambda11 = 0.01')

model4 = Neural_network(2,10,2, epoch=10, step_size=0.01, lambda11=0,lambda12=0.01,lambda21=0,lambda22=0)
model4.train(train_inputs, train_labels)
print('Model 4: 10 hidden neurons, 10 epoch, step size = 0.01, lambda12 = 0.01')
print(model4.score(test_inputs, test_labels))
print()
plot_decision_region(model4, 'lambda12 = 0.01')

model5 = Neural_network(2,10,2, epoch=10, step_size=0.01, lambda11=0,lambda12=0,lambda21=0.01,lambda22=0)
model5.train(train_inputs, train_labels)
print('Model 5: 10 hidden neurons, 10 epoch, step size = 0.01, lambda21 = 0.01')
print(model5.score(test_inputs, test_labels))
print()
plot_decision_region(model5, 'lambda21 = 0.01')

model6 = Neural_network(2,10,2, epoch=10, step_size=0.01, lambda11=0,lambda12=0,lambda21=0,lambda22=0.01)
model6.train(train_inputs, train_labels)
print('Model 6: 10 hidden neurons, 10 epoch, step size = 0.01, lambda22 = 0.01')
print(model6.score(test_inputs, test_labels))
print()
plot_decision_region(model6, 'lambda22 = 0.01')