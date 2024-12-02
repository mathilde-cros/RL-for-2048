# RL-for-2048

This is a simple implementation of the game 2048 using reinforcement learning. The game is implemented in Python and the reinforcement learning is done algorithms inspired from the content of Harvard's class STAT1840 and the papers:
- https://arxiv.org/pdf/2110.10374
- https://web.stanford.edu/class/aa228/reports/2020/final41.pdf 

The initial code for the 2048 layout is inspired from the logic.py and 2048.py files from https://www.geeksforgeeks.org/2048-game-in-python/ .

To run the code and play the game yourself, simply run python 2048.py in the terminal. The game will start and you can move the tiles in the grid using the commands as follows :
- 'u' : Move Up
- 'd' : Move Down
- 'l' : Move Left
- 'r' : Move Right

To run the code and for the reinforcement learning model to play, run python RL_2048.py in the terminal. The model will start training and will play the game itself.


ABOVE NEEDS TO BE UPDATED

To run the gradient policy algorithm with the gridsearch for 50 runs per combination, run:
python main.py --algo policy_gradient --grid_search --runs_per_combo 50
The results will then be saved in the results folder under the file name grid_search_results.csv.