import numpy as np

class Adam:
    def __init__(self, learning_rate=0.1, beta1=0.9, beta2=0.999, epsilon=1e-8):
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

def get_optimizer(name, learning_rate=0.1):
    optimizers = {
        'adam': Adam(learning_rate)
    }
    return optimizers[name.lower()]

