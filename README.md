# Connect Four Reinforcement Learning

This project implements a reinforcement learning agent for the game of Connect Four.

## Setup

### Prerequisites
- [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)

### Environment Setup

1. Clone this repository:
```bash
git clone <repository-url>
cd connect_four_rl
```

2. Create and activate the conda environment:
```bash
conda env create -f environment.yml
conda activate c4-env
```

This will install all required dependencies including:
- PyTorch and TorchVision
- NumPy
- Matplotlib
- Jupyter

## Project Structure

The project is organized as follows:
- `environment/`: Contains the Connect Four game environment
- `agents/`: Contains implementations of RL agents
- `models/`: Contains neural network models
- `notebooks/`: Jupyter notebooks for experiments and visualizations

## Usage

To start experimenting with the project, you can run one of the Jupyter notebooks:

```bash
jupyter notebook
```

## License

[Specify your license here] 