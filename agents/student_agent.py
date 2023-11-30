from agents.agent import Agent
from store import register_agent
import time
import numpy as np
from copy import deepcopy
import random


@register_agent("student_agent")
class StudentAgent(Agent):
    def __init__(self):
        super(StudentAgent, self).__init__()
        self.name = "StudentAgent"
        self.dir_map = {"u": 0, "r": 1, "d": 2, "l": 3}

    def step(self, chess_board, my_pos, adv_pos, max_step):
        start_time = time.time()

        # # Implement your alpha-beta search here
        # best_move = self.alpha_beta_search(chess_board, my_pos, adv_pos, max_step)

        time_taken = time.time() - start_time
        # print("My AI's turn took ", time_taken, "seconds.")

        moves = self.get_viable_moves(chess_board, my_pos, adv_pos, max_step)

        #return moves[random.randint(0, len(moves) - 1)]
        farthest = 0
        best_move = my_pos
        (a, b) = my_pos
        best_move = my_pos
        for i in range(len(moves)):
            (x, y) = moves[i][0] #x,y
            if ((abs(a-x)**2+abs(b-y)**2)**(1/2) > farthest):
                farthest = (abs(a-x)**2+abs(b-y)**2)**(1/2)
                best_move = moves[i]

        return best_move[0], best_move[1]

    def alpha_beta_search(self, chess_board, my_pos, adv_pos, max_step):
        # Implement alpha-beta search algorithm with a maximum time of 2 seconds

        # Set initial values for alpha and beta
        alpha = float('-inf')
        beta = float('inf')

        # Get all viable moves
        moves = self.get_viable_moves(chess_board, my_pos, adv_pos, max_step)

        # Initialize best move to a dummy value
        best_move = (my_pos, self.dir_map["u"])

        # Iterate through each move and evaluate using alpha-beta pruning
        for move in moves:
            _, value = self.min_value(
                chess_board, my_pos, adv_pos, max_step - 1, alpha, beta
            )

            # Update best move if the current move has a higher value
            if value > alpha:
                alpha = value
                best_move = move

            # Break if time exceeds 2 seconds
            # if time.time() - start_time >= 1.8:
            #     break

        return best_move

    def max_value(self, chess_board, my_pos, adv_pos, max_step, alpha, beta):
        # Implement max_value function of alpha-beta search

        if max_step == 0 or self.wins_game(chess_board, my_pos, adv_pos):
            return self.evaluate(chess_board, my_pos, adv_pos), None

        value = float('-inf')
        best_move = None

        moves = self.get_viable_moves(chess_board, my_pos, adv_pos)

        for move in moves:
            _, min_val = self.min_value(
                chess_board, my_pos, adv_pos, max_step - 1, alpha, beta
            )

            if min_val > value:
                value = min_val
                best_move = move

            if value >= beta:
                return value, best_move

            alpha = max(alpha, value)

        return value, best_move

    def min_value(self, chess_board, my_pos, adv_pos, max_step, alpha, beta):
        if max_step == 0 or self.wins_game(chess_board, my_pos, adv_pos):
            return self.evaluate(chess_board, my_pos, adv_pos), None

        value = float('inf')
        best_move = None

        moves = self.get_viable_moves(chess_board, my_pos, adv_pos)

        for move in moves:
            _, max_val = self.max_value(
                chess_board, my_pos, adv_pos, max_step - 1, alpha, beta
            )

            if max_val < value:
                value = max_val
                best_move = move

            if value <= alpha:
                return value, best_move

            beta = min(beta, value)

        return value, best_move
    
    def get_viable_moves(self, chess_board, my_pos, adv_pos, max_step):
        moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
        x, y = my_pos
        allowed_moves = []
        for d in range(0, 4):
            for l in range(0, max_step):  # 0 to the max amount you can move
                new_x, new_y = x + l * moves[d][0], y + l * moves[d][1]
                # check if we are at the end of the board
                if (new_x < 0 or new_y < 0 or new_x >= len(chess_board) or new_y >= len(chess_board)):
                    break
                # now we are at a certain cell, first check to see if we can go any farther
                # check opponent position
                if (d == 0 and adv_pos == (new_x - 1, new_y)):
                    for place in range(0, 4):
                        if not chess_board[new_x, new_y, place]:
                            allowed_moves.append(((new_x, new_y), place))
                    break
                if (d == 1 and adv_pos == (new_x + 1, new_y)):
                    for place in range(0, 4):
                        if not chess_board[new_x, new_y, place]:
                            allowed_moves.append(((new_x, new_y), place))
                    break
                if (d == 2 and adv_pos == (new_x, new_y + 1)):
                    for place in range(0, 4):
                        if not chess_board[new_x, new_y, place]:
                            allowed_moves.append(((new_x, new_y), place))
                    break
                if (d == 3 and adv_pos == (new_x, new_y - 1)):
                    for place in range(0, 4):
                        if not chess_board[new_x, new_y, place]:
                            allowed_moves.append(((new_x, new_y), place))
                    break
                # check for barrier
                if chess_board[new_x, new_y, d]:
                    # cannot move farther
                    for place in range(4):
                        if not chess_board[new_x, new_y, place]:
                            allowed_moves.append(((new_x, new_y), place))
                    break
                for place in range(4):
                        if not chess_board[new_x, new_y, place]:
                            allowed_moves.append(((new_x, new_y), place))
        return allowed_moves


    def is_suicide(self, chess_board, my_pos, adv_pos, move):
        # Implement is_suicide function
        # Returns True if the move results in defeat, else False
        pass

    def wins_game(self, chess_board, my_pos, adv_pos):
        # Implement wins_game function
        # Returns True if the current state results in a win, else False
        pass

    def monte_carlo(self, chess_board, my_pos, adv_pos, max_step):
        # Implement monte_carlo function
        # Simulates game randomly from a given state and returns the win/loss margin as an integer
        # Call random_step function below for now, later we can use heuristic
        pass

    def minimax(self, chess_board, my_pos, adv_pos, max_step):
        # Implement minimax function
        # Builds minimax tree and returns the optimal move
        pass

    def evaluate(self, chess_board, my_pos, adv_pos):
        # Implement a dummy evaluation function
        # Returns a dummy value for evaluation
        return 0

    def random_step(self, chess_board, my_pos, adv_pos, max_step):
        moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
        steps = np.random.randint(0, max_step + 1)

        # Pick steps random but allowable moves
        for _ in range(steps):
            r, c = my_pos

            # Build a list of the moves we can make
            allowed_dirs = [ d                                
                for d in range(0,4)                           # 4 moves possible
                if not chess_board[r,c,d] and                 # chess_board True means wall
                not adv_pos == (r+moves[d][0],c+moves[d][1])] # cannot move through Adversary

            if len(allowed_dirs)==0:
                # If no possible move, we must be enclosed by our Adversary
                break

            random_dir = allowed_dirs[np.random.randint(0, len(allowed_dirs))]

            # This is how to update a row,col by the entries in moves 
            # to be consistent with game logic
            m_r, m_c = moves[random_dir]
            my_pos = (r + m_r, c + m_c)

        # Final portion, pick where to put our new barrier, at random
        r, c = my_pos
        # Possibilities, any direction such that chess_board is False
        allowed_barriers=[i for i in range(0,4) if not chess_board[r,c,i]]
        # Sanity check, no way to be fully enclosed in a square, else game already ended
        assert len(allowed_barriers)>=1 
        dir = allowed_barriers[np.random.randint(0, len(allowed_barriers))]

        return my_pos, dir