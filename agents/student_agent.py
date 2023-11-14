# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time


@register_agent("student_agent")
class StudentAgent(Agent):
    """
    A dummy class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """

    def __init__(self):
        super(StudentAgent, self).__init__()
        self.name = "StudentAgent"
        self.dir_map = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
        }

    def step(self, chess_board, my_pos, adv_pos, max_step):
        
        """
        MAIN LOGIC:
                1. Gather all viable moves (move code from starter)
                2. If any of the immediately available moves result in a win, take them. If any result in a loss, remove them.
                3. If the game has more than n moves to be made (Determine n later):
                    monte carlo search: simulate n random games from each possible move, return the one with the highest average utility
                4. If the game has less than n moves to be made (endgame)
                    minimax search on the endgame

            FUNCTIONS TO BUILD:

            1. get_viable_moves(): get list of possible moves EMMA
            2. is_suicide(): boolean does the move result in defeat EMMA
            3. wins_game(): boolean does the move result in a win EMMA
            4. monte_carlo(): simulates game randomly from a given state REX
            5. minimax(): builds minimax tree and returns optimal move EMMA
        """

        """
        Implement the step function of your agent here.
        You can use the following variables to access the chess board:
        - chess_board: a numpy array of shape (x_max, y_max, 4)
        - my_pos: a tuple of (x, y)
        - adv_pos: a tuple of (x, y)
        - max_step: an integer

        You should return a tuple of ((x, y), dir),
        where (x, y) is the next position of your agent and dir is the direction of the wall
        you want to put on.

        Please check the sample implementation in agents/random_agent.py or agents/human_agent.py for more details.
        """

        # Some simple code to help you with timing. Consider checking 
        # time_taken during your search and breaking with the best answer
        # so far when it nears 2 seconds.
        start_time = time.time()
        time_taken = time.time() - start_time
        
        print("My AI's turn took ", time_taken, "seconds.")

        # dummy return
        return my_pos, self.dir_map["u"]

    def eval_node_montecarlo(chess_board, my_pos, adv_pos, max_step):
        # create a new world and place barriers based on existing chess_board
        # create a two random agents
        # run the game 
        # return the win/loss margin as int
        return 1; # remove