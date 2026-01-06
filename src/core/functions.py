import numpy as np

class RosenbrockFunction:
    def __call__(self, x, y):
        return (1 - x) ** 2 + 100 * (y - x**2) ** 2

    def gradient(self, x, y):
        dx = -2 * (1 - x) - 400 * x * (y - x**2)
        dy = 200 * (y - x**2)
        return np.array([dx, dy])

    @property 
    def bounds(self):
        return (-2, 2), (-1, 3)

    @property 
    def optimum(self):
        return (1.0, 1.0)

class SphereFunction:
    def __call__(self, x, y):
        return x**2 + y**2

    def gradient(self, x, y):
        return np.array([2 * x, 2 * y])

    @property 
    def bounds(self):
        return (-2, 2), (-2, 2)

    @property 
    def optimum(self):
        return (0.0, 0.0)

class BealeFunction:
    def __call__(self, x, y):
        term1 = (1.5 - x + x * y) ** 2
        term2 = (2.25 - x + x * y**2) ** 2
        term3 = (2.625 - x + x * y**3) ** 2
        return term1 + term2 + term3

    def gradient(self, x, y):
        dx = (
            2 * (1.5 - x + x * y) * (y - 1)
            + 2 * (2.25 - x + x * y**2) * (y**2 - 1)
            + 2 * (2.625 - x + x * y**3) * (y**3 - 1)
        )
        dy = (
            2 * (1.5 - x + x * y) * x
            + 2 * (2.25 - x + x * y**2) * 2 * x * y
            + 2 * (2.625 - x + x * y**3) * 3 * x * y**2
        )
        return np.array([dx, dy])

    @property 
    def bounds(self):
        return (-4.5, 4.5), (-4.5, 4.5)

    @property 
    def optimum(self):
        (3.0, 0.5)

class HimmelblauFunction:
    def __call__(self, x, y):
        return (x**2 + y - 11) ** 2 + (x + y**2 - 7) ** 2

    def gradient(self, x, y):
        dx = 4 * x * (x**2 + y - 11) + 2 * (x + y**2 - 7)
        dy = 2 * (x**2 + y - 11) + 4 * y * (x + y**2 - 7)
        return np.array([dx, dy])

    @property 
    def bounds(self):
        return (-5, 5), (-5, 5)

    @property 
    def optimum(self):
        return (3.0, 2.0)

def get_function(name):
    functions = {
            'himmelblau': HimmelblauFunction(),
            'rosenbrock': RosenbrockFunction(),
            'sphere': SphereFunction(),
            'beale': BealeFunction()
    }
    return functions[name.lower()]
