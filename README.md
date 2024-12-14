# Winning at 2048 Using Reinforcement Learning

This project investigates the use of reinforcement learning (RL) techniques to play the game of 2048. The implementation involves various baseline models, including simple heuristics, Monte Carlo Tree Search (MCTS), and policy gradient methods. An expert agent was trained to enhance the performance of these models, utilizing gameplay data to inform decision-making. The goal is to explore effective strategies for consistently achieving high tiles and understanding the decision-making process in stochastic environments.

## Key Features

- Game Implementation: A custom implementation of 2048 optimized for computational efficiency using Python and Tkinter.
- Baseline Models: Random action policy, heuristic-based models, and their evaluation.
- Expert Agent: A CNN-based policy network trained with expert gameplay data.
- Advanced RL Methods: Integration of policy gradient algorithms and MCTS for strategic gameplay.

## Installation

```bash
git clone https://github.com/mathilde-cros/RL-for-2048.git RL-for-2048
cd RL-for-2048
pip install -r requirements.txt
```

## Usage

### Run the game

The code provide a fully comprehensive CLI to run the game and test different algoritms with different parameters.

The following command will display all the options possible
```
python main.py --help
```

To run the Monte Carlo Tree Search algorithm:
```
python main.py --algo mcts --gui --delay 100
```

To run the Policy Gradien algorith:
```
python main.py --algo policy_gradient --gui --delay 100
```
You also can run several run of an algo to see how it performs in average:
``` 
python main.py --algo heuristic --heuristic combined --delay 1 --runs 100
```

### Train the expert agent
If you want to train the expert agent again:

```
python algorithms/train_policy_network.py
```

## Acknowledgments

This project was inspired by or builds upon work from the following repositories:

- [Robert Xiao's 2048 AI](https://github.com/nneonneo/2048-ai): Provides the expectimax optimization algorithm used for expert gameplay.
- [Anders Qiu's Python 2048](https://github.com/andersqiu/python-tk-2048): Offers a Python implementation of the 2048 game, which served as a base for this project.

Special thanks to the creators for making these resources available!
