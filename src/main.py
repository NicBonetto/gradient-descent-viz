from src.core import GradientDescentVisualizer, get_function, get_optimizer
import argparse

def main():
    parser = argparse.ArgumentParser(description='Visualize gradient descent algorithms')
    parser.add_argument(
        '--function',
        type=str,
        default='himmelblau',
        choices=['himmelblau'],
        help='Test function to optimize'
    )
    parser.add_argument(
        '--optimizer',
        type=str,
        default='adam',
        choices=['adam'],
        help='Optimization algorithm'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=0.01,
        help='Learning rate'
    )
    parser.add_argument(
        '--iter',
        type=int,
        default=100,
        help='Number of iterations'
    )
    parser.add_argument(
        '--start-x',
        type=float,
        default=-1.0,
        help='Starting x coordinate'
    )
    parser.add_argument(
        '--start-y',
        type=float,
        default=-1.0,
        help='Starting y coordinate'
    )
    parser.add_argument(
        '--mode',
        type=str,
        default='3d',
        choices=['3d', '2d', 'interactive'],
        help='Visualization mode'
    )

    args = parser.parse_args()

    optimizer = get_optimizer(args.optimizer, learning_rate=args.lr)
    func = get_function(args.function)

    viz = GradientDescentVisualizer(
        objective_function=func,
        optimizer=optimizer,
        start_point=(args.start_x, args.start_y),
        iterations=args.iter
    )

    if args.mode == '3d':
        viz.visualize_3d()
    elif args.mode == '2d':
        viz.visualize_2d()
    else:
        viz.visualize_interactive()

if __name__ == '__main__':
    main()
