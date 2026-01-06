import numpy as np

class SGD:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.history = []

    def step(self, point, gradient):
        new_point = point - self.learning_rate * gradient
        self.history.append(new_point.copy())
        return new_point
    
    def reset(self):
        self.history = []

class MomentumSGD:
    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity = None
        self.history = []

    def step(self, point, gradient):
        if self.velocity is None:
            self.velocity = np.zeros_like(point)

        self.velocity = self.momentum * self.velocity - self.learning_rate * gradient
        new_point = point + self.velocity

        self.history.append(new_point.copy())
        return new_point

    def reset(self):
        self.velocity = None
        self.history = []

class RMSProp:
    def __init__(self, learning_rate=0.01, decay=0.9, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.decay = decay
        self.epsilon = epsilon
        self.squared_grad = None
        self.history = []

    def step(self, point, gradient):
        if self.squared_grad is None:
            self.squared_grad = np.zeros_like(point)

        self.squared_grad = (self.decay * self.squared_grad + (1 - self.decay) * gradient**2)
        new_point = point - self.learning_rate * gradient / (self.squared_grad**0.5 + self.epsilon)

        self.history.append(new_point.copy())
        return new_point

    def reset(self):
        self.history = []
        self.squared_grad = None


class Adam:
    def __init__(self, learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0
        self.history = []

    def step(self, point, gradient):
        if self.m is None:
            self.m = np.zeros_like(point)
            self.v = np.zeros_like(point)

        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradient
        self.v = self.beta2 * self.v + (1 - self.beta2) * gradient**2

        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)

        new_point = point - self.learning_rate * m_hat / (v_hat**0.5 + self.epsilon)
        self.history.append(new_point.copy())
        return new_point

    def reset(self):
        self.history = []
        self.m = None
        self.v = None
        self.t = 0

def get_optimizer(name, learning_rate=0.01):
    optimizers = {
        'adam': Adam(learning_rate),
        'sgd': SGD(learning_rate),
        'momentumsgd': MomentumSGD(learning_rate),
        'rmsprop': RMSProp(learning_rate)
    }
    return optimizers[name.lower()]

