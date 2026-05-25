# World's Hardest Game AI

A reinforcement learning project that trains an AI agent to play a Pygame version of **World's Hardest Game**. The project compares **Deep Q-Network (DQN)** and **Double DQN** agents on maze-like levels with moving enemies, wall collisions, checkpoints, rewards, and saved training results.

## Overview

The goal of this project is to build an intelligent game-playing agent that can learn how to move through dangerous areas, avoid enemies, collect targets, and complete levels through trial and error. Instead of hard-coding the solution path, the agent learns a policy from game states and rewards.

Main objectives:

- Build a playable Pygame environment based on tile maps.
- Implement human-playable levels and AI-compatible game levels.
- Train reinforcement learning agents using DQN and Double DQN.
- Save trained models, experience replay buffers, evaluation results, and training plots.
- Compare the performance of DQN and Double DQN on Level 1.

## Key Features

- **Pygame game environment** with tiled maps and moving enemies.
- **3 levels**: Level 1, Level 2, and Level 3.
- **Human play mode** through standalone level scripts.
- **DQN and Double DQN agents** for reinforcement learning.
- **Enhanced Dueling DQN network** with Layer Normalization and separate value/advantage streams.
- **22-dimensional state representation** including danger detection, direction, target location, distance features, enemy risk, safe direction count, path efficiency, area control, and movement momentum.
- **Experience replay**, priority memory, success memory, target network updates, adaptive epsilon decay, and reward shaping.
- **Saved models and results** for trained Level 1 agents.
- **Training plots and demo video** included in the repository.

## Tech Stack

- Python
- PyTorch
- Pygame
- NumPy
- Matplotlib
- OpenCV
- PyTMX

## Project Structure

```text
CS106-maci/
├── source_code/
│   ├── Tiles/                         # Tile sets and TMX level maps
│   │   ├── level1.tmx
│   │   ├── level2.tmx
│   │   └── level3.tmx
│   ├── saved_models/                  # Trained PyTorch models
│   │   ├── DQN_Level1.pth
│   │   ├── DQN_Level1_BEST.pth
│   │   ├── Double_DQN_Level1.pth
│   │   └── Double_DQN_Level1_BEST.pth
│   ├── saved_experiences/             # Replay buffers
│   ├── training_results/              # JSON training and evaluation logs
│   ├── plots/                         # Training curves
│   ├── videos1/                       # Demo video
│   ├── advanced_trainer_fixed.py       # Main training manager and RL agent
│   ├── model.py                       # Enhanced Dueling DQN model
│   ├── trainer.py                     # DQN training step
│   ├── trainer2_fixed.py              # Double DQN training step
│   ├── game_level1.py                 # AI-compatible Level 1 environment
│   ├── game_level2.py                 # AI-compatible Level 2 environment
│   ├── game_level3.py                 # AI-compatible Level 3 environment
│   ├── level1.py                      # Human-playable Level 1
│   ├── level2.py                      # Human-playable Level 2
│   ├── level3.py                      # Human-playable Level 3
│   ├── train_single.py                # CLI script for training one model
│   ├── play_game.py                   # CLI script for testing trained models
│   ├── play_trained_agent.py          # Interactive trained-agent player
│   ├── setup_environment.py           # Dependency checking and setup helper
│   └── requirements.txt
└── .gitignore
```

## Installation

Clone the repository and move into the source folder:

```bash
git clone <your-repository-url>
cd CS106-maci/source_code
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

You can also use the setup helper:

```bash
python setup_environment.py
```

## How to Run

> Run all commands inside the `source_code/` folder because the project uses relative paths such as `Tiles/`, `saved_models/`, and `arial.ttf`.

### 1. Play manually

Run one of the human-playable levels:

```bash
python level1.py
python level2.py
python level3.py
```

### 2. List available trained models

```bash
python play_game.py list
```

### 3. Play using a trained AI agent

Run DQN on Level 1:

```bash
python play_game.py Level1 DQN 5 True
```

Run the best Double DQN model on Level 1:

```bash
python play_game.py Level1 Double_DQN 5 True best
```

Arguments:

```text
python play_game.py <level> <algorithm> [num_games] [show_game] [use_best]
```

Example:

```bash
python play_game.py Level1 Double_DQN 10 True best
```

### 4. Train a model

Train DQN:

```bash
python train_single.py Level1 DQN 3000 300
```

Train Double DQN:

```bash
python train_single.py Level1 Double_DQN 3000 300
```

Arguments:

```text
python train_single.py <level> <algorithm> [episodes] [eval_interval]
```

Supported values:

- `level`: `Level1`, `Level2`, `Level3`
- `algorithm`: `DQN`, `Double_DQN`
- `episodes`: number of training episodes
- `eval_interval`: evaluate the agent every N episodes

After training, the project saves:

- trained model: `saved_models/`
- replay buffer: `saved_experiences/`
- evaluation logs: `training_results/`
- training plot: `plots/`

## Current Training Results

The repository includes trained Level 1 models and evaluation logs. Based on the saved JSON result files:

| Model | Level | Best Episode | Win Rate | Average Score | Games Won |
|---|---:|---:|---:|---:|---:|
| DQN | Level 1 | 10000 | 36% | 3.70 | 18/50 |
| Double DQN | Level 1 | 9000 | 74% | 5.38 | 37/50 |

Double DQN achieved a higher win rate and average score on Level 1 in the saved experiments.

## Demo Assets

The project includes the following visual outputs:

```text
source_code/plots/DQN_Level1_training.png
source_code/plots/Double_DQN_Level1_training.png
source_code/videos1/best_gamelv1_db_dqn_125.mp4
```

These files can be used in a report, slide deck, or GitHub README to demonstrate training progress and gameplay behavior.

## Reinforcement Learning Design

### State Representation

The agent observes a 22-feature state vector containing:

- danger in four directions;
- current movement direction;
- target position relative to the player;
- normalized distance to the target;
- current progress score;
- nearest enemy distance;
- predicted enemy collision risk;
- number of safe movement directions;
- path efficiency;
- area control;
- movement momentum.

### Action Space

The agent chooses one of four actions:

```text
Right, Down, Left, Up
```

### Reward Strategy

The reward system encourages the agent to:

- move toward the target;
- avoid enemies and walls;
- collect objectives;
- reduce unnecessary oscillation;
- complete the level efficiently.

### Model Architecture

The project uses an enhanced Dueling DQN-style network:

- input layer with 22 features;
- hidden layers with 256 and 128 units;
- Layer Normalization;
- Dropout;
- separate value stream and advantage stream;
- Q-value aggregation using the Dueling DQN formulation.

## Notes and Known Issues

- The stable training entry point is `train_single.py`.
- The stable trained-agent testing entry point is `play_game.py`.
- Some older scripts such as `main.py` and `mainscreen.py` reference legacy modules named `agent.py` and `agent2.py`, which are not included in the current ZIP. If these files are missing, use `train_single.py`, `play_game.py`, `level1.py`, `level2.py`, and `level3.py` instead.
- If Pygame display fails on non-Windows systems, check the line that sets `SDL_VIDEODRIVER` to `windib` in the level files. `windib` is Windows-specific; on macOS/Linux, you may need to remove that line or use a compatible driver.
- Level 1 has saved trained models. Level 2 and Level 3 have environment code, but additional training may be required before they can be played well by the AI.

## Future Improvements

- Train and compare models on Level 2 and Level 3.
- Add a cleaner unified menu for human play and AI play.
- Add more evaluation metrics such as average survival time and collision count.
- Save experiment configurations automatically for reproducibility.
- Improve reward shaping for harder levels.
- Add a Streamlit or web dashboard for visualizing training results.

## License

This repository contains an MIT License file. If the project uses or adapts code from external tutorials or repositories, keep proper attribution in the final submission.
