import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go

class GradientDescentVisualizer:
    def __init__(self, objective_function, optimizer, start_point, iterations=100):
        self.func = objective_function
        self.optimizer = optimizer
        self.start_point = np.array(start_point)
        self.iterations = iterations
        self.trajectory = None

    def run_optimization(self):
        self.optimizer.reset()
        point = self.start_point.copy()
        trajectory = [point.copy()]

        for _ in range(self.iterations):
            grad = self.func.gradient(point[0], point[1])
            point = self.optimizer.step(point, grad)
            trajectory.append(point.copy())

        self.trajectory = np.array(trajectory)
        return self.trajectory

    def visualize_3d(self):
        if self.trajectory is None:
            self.run_optimization()

        x_bounds, y_bounds = self.func.bounds
        x = np.linspace(x_bounds[0], x_bounds[1], 100)
        y = np.linspace(y_bounds[0], y_bounds[1], 100)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)

        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = self.func(X[i, j], Y[i, j])

        traj_z = np.array([self.func(p[0], p[1]) for p in self.trajectory])

        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')

        ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6, edgecolor='none')

        ax.plot(
                self.trajectory[:, 0],
                self.trajectory[:, 1],
                traj_z,
                'r.-',
                linewidth=2,
                markersize=6,
                label='Optimization Path'
        )

        ax.scatter(
            [self.trajectory[0, 0]],
            [self.trajectory[0, 1]],
            [traj_z[0]],
            c='green',
            s=100,
            label='Start'
        )

        ax.scatter(
            [self.trajectory[-1, 0]],
            [self.trajectory[-1, 1]],
            [self.trajectory[-1]],
            c='red',
            s=150,
            marker='*',
            label='End'
        )

        ax.set_xlabel('x', fontsize=10)
        ax.set_ylabel('y', fontsize=10)
        ax.set_zlabel('f(x, y)', fontsize=10)

        ax.set_title('3D Gradient Descent Visualization', fontsize=14)
        ax.legend()

        plt.tight_layout()
        plt.show()

    def get_convergence_data(self):
        if self.trajectory is None:
            self.run_optimization()

        values = np.array([self.func(p[0], p[1]) for p in self.trajectory])
        distances = np.array([np.linalg.norm(p - self.func.optimum) for p in self.trajectory])

        return {
            'iterations': len(self.trajectory),
            'final_value': values[-1],
            'optimal_value': self.func(*self.func.optimum),
            'final_distance': distances[-1],
            'trajectory': self.trajectory,
            'values': values,
            'distances': distances
        }
