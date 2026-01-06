# gradient-descent-viz
Gradient descent vizualizer.

## Installation
1. Clone the repository or download the project
2. Install dependencies
```bash
python -m pip install -r requirements.txt
```

## Usage
Run a visualization with default settings:
```bash
python -m src.main
```

Available options:
- `--function`: Gradient function. Choose from `rosenbrock`, `sphere`, `beale`, or `himmelblau` (default: `himmelblau`)
- `--optimizer`: Optimization function. Choose from `sgd`, `momentumsgd`, `rmsprop`, or `adam` (default: `adam`)
- `--lr`: Learning rate (default: 0.01)
- `--iter`: Number of optimization steps (default: 100)
- `--start-x`, `--start-y`: Starting coordinates
- `--mode`: Visualization mode. Choose from `2d`, `3d`, or `interactive` (default: `3d`)

## Examples
Visualize SGD on Rosenbrock function on a 2D graph:
```bash
python -m src.main --function rosenbrock --optimizer sgd --mode 2d
```

Visualize Adam optimizer on Beale function:
```bash
python -m src.main --function beale --optimizer adam
```
