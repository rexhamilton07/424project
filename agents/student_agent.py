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
        move = self.monte_carlo(chess_board, my_pos, adv_pos, max_step)
        return move

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
        #check kill and suicide for all moves
        i = 0
        filtered_moves = []
        for (x, y), d in allowed_moves:
            i+=1
            potential_move = ((x,y),d)
            new_board = self.add_move_to_board(chess_board, potential_move)
            new_pos = (x, y)
            if self.is_suicide(new_board, new_pos, adv_pos):
                continue
            if self.wins_game(new_board, new_pos, adv_pos):
                return [potential_move]
            filtered_moves.append(allowed_moves[i-1])
        return filtered_moves

    def add_move_to_board(self, chess_board, move):
        new_board = deepcopy(chess_board)
        (x,y),d = move
        new_board[x, y, d] = True
        if d == 0:
            new_board[x-1,y,2] = True
        if d ==1:
            new_board[x,y+1,3] = True
        if d == 2:
            new_board[x+1,y,0] = True
        if d == 3:
            new_board[x,y-1,1] = True
        return new_board

    def is_endgame(self, my_pos, adv_pos, chess_board):
        # Union-Find
        moves = moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
        
        father = dict()
        for r in range(len(chess_board)):
            for c in range(len(chess_board)):
                father[(r, c)] = (r, c)

        def find(pos):
            if father[pos] != pos:
                father[pos] = find(father[pos])
            return father[pos]

        def union(pos1, pos2):
            father[pos1] = pos2

        for r in range(len(chess_board)):
            for c in range(len(chess_board)):
                for dir, move in enumerate(
                    moves[1:3]
                ):  # Only check down and right
                    if chess_board[r, c, dir + 1]:
                        continue
                    pos_a = find((r, c))
                    pos_b = find((r + move[0], c + move[1]))
                    if pos_a != pos_b:
                        union(pos_a, pos_b)

        for r in range(len(chess_board)):
            for c in range(len(chess_board)):
                find((r, c))
        p0_r = find(tuple(my_pos))
        p1_r = find(tuple(adv_pos))
        p0_score = list(father.values()).count(p0_r)
        p1_score = list(father.values()).count(p1_r)
        if p0_r == p1_r:
            return False, p0_score, p1_score
        player_win = None
        win_blocks = -1
        if p0_score > p1_score:
            player_win = 0
            win_blocks = p0_score
        elif p0_score < p1_score:
            player_win = 1
            win_blocks = p1_score
        else:
            player_win = -1
        return True, p0_score, p1_score
    
    def is_suicide(self, chess_board, my_pos, adv_pos):
        result, x, y = self.is_endgame(my_pos, adv_pos, chess_board)
        if result and x < y:
            return True
        return False

    def wins_game(self, chess_board, my_pos, adv_pos):
        result, x, y = self.is_endgame(my_pos, adv_pos, chess_board)
        if result and x >= y:
            return True
        return False

    def monte_carlo(self, chess_board, my_pos, adv_pos, max_step):
        # Implement monte carlo 
        start_time = time.time()
        options = self.get_viable_moves(chess_board, my_pos, adv_pos, max_step)
        if (len(options) > 10):
            filtered_options = []
            for i in range(5):
                filtered_options.append(options[random.randint(0, len(options)-1)])
            options = filtered_options
        best_move = options[0]
        best_move_count = 0
        for a in options:
            win_count = 0
            for i in range(10):
                if self.mc_step(chess_board, my_pos, adv_pos, True, max_step):
                    win_count += 1
            if win_count > best_move_count:
                best_move_count = win_count
                best_move = a
        return best_move
                

    def mc_step(self, chess_board, my_pos, adv_pos, my_turn, max_step):
        # implement step function to check if is_endgame and if not select a random move and call itself
        res, x, y = self.is_endgame(my_pos, adv_pos, chess_board)
        if res:
            win = (x > y)
            return win
        if (my_turn):
            options = self.get_viable_moves(chess_board, my_pos, adv_pos, max_step)
            if len(options) <= 0:
                return False
            move = options[random.randint(0, len(options)-1)] # implement heuristic here later on
            new_board = self.add_move_to_board(chess_board, move)
            (a, b), d = move
            self.mc_step(new_board, (a, b), adv_pos, False, max_step)
        else:
            options = self.get_viable_moves(chess_board, adv_pos, my_pos, max_step)
            if len(options) <= 0:
                return False
            move = options[random.randint(0, len(options)-1)] # implement heuristic here later on
            new_board = self.add_move_to_board(chess_board, move)
            (a, b), d = move
            self.mc_step(new_board, my_pos, (a, b), True, max_step)

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