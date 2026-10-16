# 🐍 SnakeDQN

A Deep Q-Network (DQN) agent that learns to play the classic Snake game using reinforcement learning, built with PyTorch and Pygame.

## Overview

This project implements a reinforcement learning agent that teaches itself to play Snake through trial and error. The agent uses a simple feedforward neural network (Linear Q-Network) trained with the DQN algorithm, including experience replay and an epsilon-greedy exploration strategy.

### How It Works

1. **State Representation (11 features):** The agent observes danger in 3 directions (straight, left, right), its current movement direction (4 values), and the relative food position (4 values).
2. **Action Space:** The agent chooses from 3 actions — go straight, turn right, or turn left.
3. **Reward Signal:** +10 for eating food, -10 for dying (hitting a wall or itself).
4. **Training:** Uses short-term (single step) and long-term (experience replay from memory) training with MSE loss and the Adam optimizer.

### Project Structure

```
snakeDQN/
├── agent.py              # DQN agent with training loop
├── model.py              # Linear Q-Network and Q-Trainer
├── snake_game.py         # Playable Snake game (human-controlled)
├── snake_game_agent.py   # Snake game environment for the AI agent
├── helpers.py            # Live training plot with matplotlib
├── arial.ttf             # Font for in-game score display
├── models/
│   └── model.pth         # Saved model weights (best score)
├── pyproject.toml        # Project metadata and dependencies
├── uv.lock               # Locked dependency versions
└── .python-version       # Python 3.12
```

## Prerequisites

- **Python 3.12+**
- **[uv](https://docs.astral.sh/uv/)** — a fast Python package and project manager
- A display server (X11/Wayland) for the Pygame window

### Install uv

If you don't have `uv` installed:

```bash
# Linux / macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or with pip
pip install uv
```

## Setup & Installation

```bash
# Clone the repository
git clone https://github.com/rickmwasofficial/snakeDQN.git
cd snakeDQN

# Create virtual environment and install all dependencies
uv sync
```

That's it — `uv sync` reads `pyproject.toml` and `uv.lock`, creates a `.venv`, and installs all pinned dependencies.

## Usage

### Train the AI Agent

```bash
uv run agent.py
```

This starts the training loop. A Pygame window will open showing the snake playing in real time, and a matplotlib plot will track the score and mean score over games. The best model is automatically saved to `models/model.pth`.

Training output:

```
Game: 1, Score: 0, Record: 0
Game: 2, Score: 1, Record: 1
Game: 3, Score: 0, Record: 1
...
```

> **Tip:** The agent uses epsilon-greedy exploration — it makes random moves for roughly the first 80 games, then increasingly relies on the learned policy. Noticeable improvement typically starts around game 50–100.

### Play Snake Yourself

```bash
uv run snake_game.py
```

Use the **arrow keys** to control the snake. Your final score is printed when the game ends.

## Configuration

Key hyperparameters in [`agent.py`](agent.py):

| Parameter      | Default   | Description                                |
| -------------- | --------- | ------------------------------------------ |
| `MAX_MEMORY`   | 100,000   | Experience replay buffer size              |
| `BATCH_SIZE`   | 1,000     | Mini-batch size for long-term training     |
| `LR`           | 0.001     | Learning rate (Adam optimizer)             |
| `gamma`        | 0.9       | Discount factor for future rewards         |
| `epsilon`      | 80 - n    | Exploration rate (decays with game count)  |

Network architecture in [`model.py`](model.py):

- Input layer: 11 neurons (state features)
- Hidden layer: 256 neurons (ReLU activation)
- Output layer: 3 neurons (Q-values for each action)

## Dependencies

- **[PyTorch](https://pytorch.org/)** — neural network and training
- **[Pygame](https://www.pygame.org/)** — game rendering and input
- **[Matplotlib](https://matplotlib.org/)** — live training plots
- **[IPython](https://ipython.org/)** — interactive display for plots
- **[torchvision](https://pytorch.org/vision/)** — PyTorch vision utilities

## License

This project is provided as-is for educational purposes.
