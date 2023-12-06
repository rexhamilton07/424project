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
        new_node = Node(chess_board, my_pos, adv_pos, 0, False, 2)
        if (self.minimax_build_tree(chess_board, my_pos, adv_pos, max_step, new_node, True)):
            self.evaluate_node(new_node, True)
            node = self.find_move_from_minmax(new_node)
            if node.direction != new_node.direction and node.my_pos != new_node.my_pos:
                print("yes")
                if node.minmaxvalue == 1:
                    print("woohoo")
                    return node.my_pos, node.direction   
            else:
                moves = self.get_viable_moves(chess_board, my_pos, adv_pos, max_step)
                if len(moves) > 0:
                    sorted_moves = sorted(moves, key=lambda move: self.heuristic(chess_board, my_pos, adv_pos, move, max_step), reverse=True)
                    return sorted_moves[0]
                return self.random_step(chess_board, my_pos, adv_pos, max_step)

        moves = self.get_viable_moves(chess_board, my_pos, adv_pos, max_step)
        if len(moves) > 0:
            sorted_moves = sorted(moves, key=lambda move: self.heuristic(chess_board, my_pos, adv_pos, move, max_step), reverse=True)
            sorted_moves = sorted_moves[:10]
            best_move = self.monte_carlo(chess_board, my_pos, adv_pos, max_step, sorted_moves, start_time)
            return best_move
        return self.random_step(chess_board, my_pos, adv_pos, max_step)

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

    def heuristic(self, chess_board, my_pos, adv_pos, move, max_step):
        score = 0  
        score += self.move_distance(my_pos, move)
        dis = self.adv_distance(move, adv_pos)
        #if (not self.three_walls(chess_board, my_pos, adv_pos, move)):
        score -= dis[0] #want to move closer to the adversary but not if the position is a trap
        if dis[1] == True:
            score += 2
        # 
        #     score -= 5 
        return score
    #distance between move and adv_pos
    # self.dir_map = {"u": 0, "r": 1, "d": 2, "l": 3}
    def adv_distance(self, move, adv_pos):
        my_x, my_y = move[0]
        adv_x, adv_y = adv_pos
        wall_pos = False
        if my_x > adv_x:
            if move[1] == 3:
                wall_pos = True
        else:
            if move[1] == 1:
                wall_pos = True
        if my_y > adv_y :
            if move[1] == 2:
                wall_pos == True
        else:
            if move[1] == 0:
                wall_pos == True
    
        return abs(my_x - adv_x) + abs(my_y - adv_y), wall_pos
    
    # how far does the agent travel for the move
    def move_distance(self, my_pos, move):
        (x,y),d = move
        (a, b) = my_pos
        return np.sqrt((x - a)**2 + (y - b)**2)

    # contained by three walls
    def three_walls(self, chess_board, my_pos, adv_pos, move):
        numwalls = 0
        # self.dir_map = {"u": 0, "r": 1, "d": 2, "l": 3}
        (x, y), d = move
        chess_board = self.add_move_to_board(chess_board, move)
        if (chess_board[x, y, 0]):
            numwalls+=1
        if (chess_board[x, y, 1]):
            numwalls+=1
        if (chess_board[x, y, 2]):
            numwalls+=1
        if (chess_board[x, y, 3]):
            numwalls+=1
        if numwalls == 2:
            return True
        if numwalls == 3:
            return True
        return False

    # continues a wall

    def continues_wall(self, chess_board, move): # only counts straight walls, not corners (will box itself in three times and then move)
        (x, y), d = move
        # self.dir_map = {"u": 0, "r": 1, "d": 2, "l": 3}
        if d == 0 and ((x > 0 and chess_board[x-1, y, 0]) or (x < len(chess_board) -1 and chess_board[x+1, y, 0])):
            return True
        if d == 2 and ((x > 0 and chess_board[x-1, y, 2]) or (x < len(chess_board) -1 and chess_board[x+1, y, 2])):
            return True
        if d == 1 and ((y > 0 and chess_board[x, y -1, 1]) or (y < len(chess_board) -1 and chess_board[x, y+1, 1])):
            return True
        if d == 3 and ((y > 0 and chess_board[x, y -1, 3]) or (y < len(chess_board) -1 and chess_board[x, y+1, 3])):
            return True
        return False

    def add_move_to_board(self, chess_board, move):
        (x, y), d = move
        temp_board = np.copy(chess_board)
        temp_board[x, y, d] = True
        if d == 0:
            temp_board[x-1, y, 2] = True
        elif d == 1:
            temp_board[x, y+1, 3] = True
        elif d == 2:
            temp_board[x+1, y, 0] = True
        elif d == 3:
            temp_board[x, y-1, 1] = True
        return temp_board

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

    def monte_carlo(self, chess_board, my_pos, adv_pos, max_step, options, start_time):
        best_move = options[0]
        best_move_count = 0
        for a in options:
            if (time.time() - start_time > 1.5):
                return best_move
            win_count = 0
            for i in range(30):
                if self.mc_step(chess_board, my_pos, adv_pos, True, max_step):
                    win_count += 1
            if win_count > best_move_count:
                best_move_count = win_count
                best_move = a
        return best_move                

    def mc_step(self, chess_board, my_pos, adv_pos, my_turn, max_step):
        res, x, y = self.is_endgame(my_pos, adv_pos, chess_board)
        if res:
            win = (x > y)
            return win
        if (my_turn):
            moves = []
            for i in range(0, 5): # can decide how many we want
                moves.append(self.random_step(chess_board, my_pos, adv_pos, max_step))
            sorted_moves = sorted(moves, key=lambda move: self.heuristic(chess_board, my_pos, adv_pos, move, max_step), reverse=True)
            new_board = self.add_move_to_board(chess_board, moves[0])
            (a, b), d = moves[0]
            return self.mc_step(new_board, (a, b), adv_pos, False, max_step)
        else:
            moves = []
            for i in range(0, 5): # can decide how many we want
                moves.append(self.random_step(chess_board, adv_pos, my_pos, max_step))
            sorted_moves = sorted(moves, key=lambda move: self.heuristic(chess_board, adv_pos, my_pos, move, max_step), reverse=True)
            new_board = self.add_move_to_board(chess_board, moves[0])
            (a, b), d = moves[0]
            return self.mc_step(new_board, my_pos, (a, b), True, max_step)

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

    def minimax_build_tree(self, chess_board, my_pos, adv_pos, max_step, root, maxplayer):
        tmp_my_pos = my_pos
        tmp_adv_pos = adv_pos
        #alternate between maxplayer and minplayer
        if maxplayer: 
            new_moves = self.get_viable_moves(chess_board, my_pos, adv_pos, max_step) #get all moves from current position
            # check to see if the tree is too big
            if len(new_moves) > 5:
                return False
            if len(new_moves) == 0: #make sure you can move otherwise return
                return True
            if len(new_moves) == 1:
                #check if this move wins the game
                new_chess_board = self.add_move_to_board(chess_board, new_moves[0])
                tmp_my_pos = new_moves[0][0]
                #make move and update my_pos and chessboard (move my_pos based on 'move')
                if self.wins_game(new_chess_board, tmp_my_pos, adv_pos):
                    #update chess board and new position
                    new_node = Node(new_chess_board, tmp_my_pos, adv_pos, 1, True, new_moves[0][1]) #gets 1 because you wins the game
                    root.add_move(new_node)
                else:
                    #update chess board and new position
                    new_node = Node(new_chess_board, tmp_my_pos, adv_pos, 0, False, new_moves[0][1])
                    root.add_move(new_node)
                    self.minimax_build_tree(new_chess_board, tmp_my_pos, adv_pos, max_step, new_node, False)
                    self.evaluate_node(new_node, True)
                return True
            for move in new_moves:
                #make move and update my_pos and chessboard (move my_pos based on 'move')
                new_chess_board = self.add_move_to_board(chess_board, move)
                tmp_my_pos = move[0]
                new_node = Node(new_chess_board, tmp_my_pos, adv_pos, 0, False, move[1])
                root.add_move(new_node)
                #call minimax on this node as root
                self.minimax_build_tree(new_chess_board, tmp_my_pos, adv_pos, max_step, new_node, False)
                self.evaluate_node(new_node, True)
        else: #minplayer
            new_moves = self.get_viable_moves(chess_board, adv_pos, my_pos, max_step)
            # check to see if the tree is too big
            if len(new_moves) > 5:
                return False
            if len(new_moves) == 0:
                return True
            if len(new_moves) == 1:
                new_chess_board = self.add_move_to_board(chess_board, new_moves[0])
                tmp_adv_pos = new_moves[0][0]
                #check if this move wins the game
                if self.wins_game(new_chess_board, tmp_adv_pos, my_pos):
                    
                    new_node = Node(new_chess_board, my_pos, tmp_adv_pos, -1, False, new_moves[0][1]) #gets -1 because adv wins the game
                    root.add_move(new_node)
                else: 
                    new_node = Node(new_chess_board, my_pos, tmp_adv_pos, 0, False, new_moves[0][1])
                    root.add_move(new_node)
                    self.minimax_build_tree(new_chess_board, my_pos, tmp_adv_pos, max_step, new_node, True)
                    self.evaluate_node(new_node, False)
                return True
            for move in new_moves: #get all moves from current position
                #make move and update adv_pos and chessboard (move the adv_pos based on 'move')
                new_chess_board = self.add_move_to_board(chess_board, move)
                tmp_adv_pos = move[0]
                new_node = Node(new_chess_board, my_pos, tmp_adv_pos, 0, False, move[1])
                root.add_move(new_node)
                #call minimax on this node as root
                self.minimax_build_tree(new_chess_board, my_pos, tmp_adv_pos, max_step, new_node, True)
                self.evaluate_node(new_node, False)
        return True
    def evaluate_node(self,eval_node, maxplayer):
        nodes = eval_node.nodes
        valuemax = -10
        valuemin = 10
        if len(nodes) != 0:
            for node in nodes:
                if maxplayer:
                    if node.minmaxvalue >= valuemax :
                        eval_node.minmaxvalue = node.minmaxvalue
                else:
                    if node.minmaxvalue <= valuemin :
                        eval_node.minmaxvalue = node.minmaxvalue
    def find_move_from_minmax(self,root): 
        child_nodes = root.nodes
        if len(child_nodes) == 0 :
            return root
        tmp_node = child_nodes[0]
        for node in child_nodes:
            if node.minmaxvalue == 1 :
                tmp_node = node
                if node.winsgame == True:
                    return node
        return tmp_node 
class Node:
    def __init__(self, chess_board, my_pos, adv_pos, minmaxvalue, winsgame, direction):
        self.chess_board = chess_board
        self.my_pos = my_pos
        self.adv_pos = adv_pos
        self.nodes = []
        self.minmaxvalue = minmaxvalue
        self.winsgame = winsgame
        self.direction = direction
    def add_move(self, child):
        self.nodes.append(child)