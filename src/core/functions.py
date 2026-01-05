import numpy as np

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
            'himmelblau': HimmelblauFunction()
    }
    return functions[name.lower()]
