"""PlayerSkeletonA.py
The beginnings of an agent that might someday play Baroque Chess.

"""
import time
import BC_state_etc as BC
from random import randint

USE_CUSTOM_STATIC_EVAL_FUNCTION = True
import sys

CURRENT_PLAYER = 0
OTHER_PLAYER = 1
pruning = True
zobristnum = []


def parameterized_minimax(currentState, alphaBeta=True, ply=3, useBasicStaticEval=True, useZobristHashing=False):
    """Implement this testing function for your agent's basic
    capabilities here."""
    global CURRENT_PLAYER, pruning, start, time_limit
    CURRENT_PLAYER = currentState.whose_move
    global OTHER_PLAYER
    if CURRENT_PLAYER == 0:
        OTHER_PLAYER = 1
    else:
        OTHER_PLAYER = 0

    # pass
    pruning = alphaBeta

    curr_static_eval = minimax_basic(currentState, ply)
    DATA = {'CURRENT_STATE_STATIC_VAL': curr_static_eval,
            'N_STATES_EXPANDED': 0,
            'N_STATIC_EVALS': 0,
            'N_CUTOFFS': 0}
    return DATA


DIRECTIONS = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
BOARD_LENGTH = 8

# map direction to its 180 degree counterpart
DIRECTION_OPPOSITES = {'N': 'S', 'NE': 'SW', 'E': 'W', 'SE': 'NW', 'S': 'N', 'SW': 'NE', 'W': 'E', 'NW': 'SE'}

DIRECTION_TO_CODE_DICT = {'N': 0, 'NE': 1, 'E': 2, 'SE': 3, 'S': 4, 'SW': 5, 'W': 6, 'NW': 7}

# Maps the following:
# Piece:
PIECES_MOVES_DICT = []


def get_successors(current_state):
    """Return a list of possible successor states"""
    board = current_state.board

    global CURRENT_PLAYER
    CURRENT_PLAYER = current_state.whose_move

    # for j in range(3):
    #     for i in range(2, -1, -1):
    #         print(matrix[i][j])
    global PIECES_MOVES_DICT

    # initialize dictionary to be empty for all pieces
    # i = 0
    # for piece in ['F', 'L', 'K', 'W', 'C', 'P', 'f', 'l', 'k',
    #             'w', 'p']:

    #   i += 1
    #  PIECES_MOVES_DICT[piece] = []
    # print(str(piece) + ' ' + str(PIECES_MOVES_DICT[piece]))

    # print('IN GET SUCCESSORS AFTER INITIALIZING DICT TO EMPTY')
    for j in range(BOARD_LENGTH):
        for i in range(BOARD_LENGTH - 1, -1, -1):
            piece = BC.CODE_TO_INIT[board[i][j]]

            if CURRENT_PLAYER == BC.WHITE:

                if piece == 'F':  # white freezer
                    # a = get_F_moves(board,i,j)
                    # print("F i,j: ",(i,j))
                    if get_F_moves(board, i, j):
                        PIECES_MOVES_DICT.append([piece, get_F_moves(board, i, j)[0][0], get_F_moves(board, i, j)[0][1],
                                                  get_F_moves(board, i, j)[0][2]])

                    # PIECES_MOVES_DICT[piece].append(get_F_moves(board, i, j))
                elif piece == 'L':  # leaper
                    # print("L i,j: ",(i,j))
                    if get_L_moves(board, i, j) != []:
                        PIECES_MOVES_DICT.append([piece, get_L_moves(board, i, j)[0][0], get_L_moves(board, i, j)[0][1],
                                                  get_L_moves(board, i, j)[0][2]])

                elif piece == 'K':  # king
                    # print("K i,j: ",(i,j))
                    if get_K_moves(board, i, j):
                        PIECES_MOVES_DICT.append([piece, get_K_moves(board, i, j)[0][0], get_K_moves(board, i, j)[0][1],
                                                  get_K_moves(board, i, j)[0][2]])

                elif piece == 'W':  # withdrawer
                    if get_W_moves(board, i, j):
                        PIECES_MOVES_DICT.append([piece, get_W_moves(board, i, j)[0][0], get_W_moves(board, i, j)[0][1],
                                                  get_W_moves(board, i, j)[0][2]])

                elif piece == 'C':  # Coordinator
                    if get_C_moves(board, i, j):
                        PIECES_MOVES_DICT.append([piece, get_C_moves(board, i, j)[0][0], get_C_moves(board, i, j)[0][1],
                                                  get_C_moves(board, i, j)[0][2]])

                elif piece == 'P':  # Pincer
                    print("P i,j: ", (i, j))
                    p_moves = get_P_moves(board, i, j)

                    if p_moves != []:
                        PIECES_MOVES_DICT.append([piece, p_moves[0][0], p_moves[0][1], p_moves[0][2]])

            else:  # CURRENT_PLAYER == BC.BLACK
                if piece == 'f':  # BLACK freezer
                    if get_F_moves(board, i, j):
                        PIECES_MOVES_DICT.append([piece, get_F_moves(board, i, j)[0][0], get_F_moves(board, i, j)[0][1],
                                                  get_F_moves(board, i, j)[0][2]])

                elif piece == 'l':  # leaper
                    if get_L_moves(board, i, j):
                        PIECES_MOVES_DICT.append([piece, get_L_moves(board, i, j)[0][0], get_L_moves(board, i, j)[0][1],
                                                  get_L_moves(board, i, j)[0][2]])

                elif piece == 'k':  # king
                    if get_K_moves(board, i, j):
                        PIECES_MOVES_DICT.append([piece, get_K_moves(board, i, j)[0][0], get_K_moves(board, i, j)[0][1],
                                                  get_K_moves(board, i, j)[0][2]])

                elif piece == 'w':  # black withdrawer
                    if get_W_moves(board, i, j) != []:
                        PIECES_MOVES_DICT.append([piece, get_W_moves(board, i, j)[0][0], get_W_moves(board, i, j)[0][1],
                                                  get_W_moves(board, i, j)[0][2]])

                elif piece == 'c':  # coordinator
                    if get_C_moves(board, i, j) != []:
                        PIECES_MOVES_DICT.append([piece, get_C_moves(board, i, j)[0][0], get_C_moves(board, i, j)[0][1],
                                                  get_C_moves(board, i, j)[0][2]])

                    # PIECES_MOVES_DICT.append([piece,get_C_moves(board, i, j)[0][0],get_C_moves(board, i, j)[0][1],get_C_moves(board, i, j)[0][2]])
                elif piece == 'p':  # pincer
                    if get_P_moves(board, i, j) != []:
                        PIECES_MOVES_DICT.append([piece, get_P_moves(board, i, j)[0][0], get_P_moves(board, i, j)[0][1],
                                                  get_P_moves(board, i, j)[0][2]])

                    # PIECES_MOVES_DICT.append([piece,get_P_moves(board, i, j)[0][0],get_P_moves(board, i, j)[0][1],get_P_moves(board, i, j)[0][2]])
    return PIECES_MOVES_DICT


def is_vacant(board, direction, i, j):
    """Returns the following list:
   index 0: whether or not the spot in the given direction is vacant and/or legal
   index 1: tuple representing coordinates of new spot in given direction (if not legal, return original coords)
   index 2: whether or not the spot in the given direction is occupied by an opposing piece
   """
    max_index = 7
    min_index = 0
    # print('a')
    # check if tile north of given tile is vacant, Repeat for every direction until input direction found
    if direction == 'N':
        # print('b')
        if i - 1 < min_index:
            # print('bb')
            return [False, (), False]
        else:
            # print('bbb')
            # print('i' + str(i))
            # print('j' + str(j))
            is_opponent = board[i - 1][j] % 2 != CURRENT_PLAYER and board[i - 1][j] != 0
            # print('after is opp')
            # print(board[i-1][j] == 0)
            return [board[i - 1][j] == 0, (i - 1, j), is_opponent]

    if direction == 'NE':
        # print('c')
        if i - 1 < min_index or j + 1 > max_index:
            return [False, (), False]
        else:
            is_opponent = board[i - 1][j + 1] % 2 != CURRENT_PLAYER and board[i - 1][j + 1] != 0
            return [board[i - 1][j + 1] == 0, (i - 1, j + 1), is_opponent]

    if direction == 'E':
        # print('d')
        if j + 1 > max_index:
            return [False, (), False]
        else:
            is_opponent = (board[i][j + 1] % 2 != CURRENT_PLAYER and board[i][j + 1] != 0)
            return [board[i][j + 1] == 0, (i, j + 1), is_opponent]

    if direction == 'SE':
        # print('e')
        if i + 1 > max_index or j + 1 > max_index:
            # print('ee')
            return [False, (), False]
        else:
            # print('eee')
            is_opponent = (board[i + 1][j + 1] % 2 != CURRENT_PLAYER and board[i + 1][j + 1] != 0)
            # print('after e opppp')
            return [board[i + 1][j + 1] == 0, (i + 1, j + 1), is_opponent]

    if direction == 'S':
        # print('f')
        if i + 1 > max_index:
            return [False, (), False]
        else:
            is_opponent = (board[i + 1][j] % 2 != CURRENT_PLAYER and board[i + 1][j] != 0)
            return [board[i + 1][j] == 0, (i + 1, j), is_opponent]

    if direction == 'SW':
        # print('g')
        if i + 1 > max_index or j - 1 < min_index:
            return [False, (), False]
        else:
            is_opponent = (board[i + 1][j - 1] % 2 != CURRENT_PLAYER and board[i + 1][j - 1] != 0)
            return [board[i + 1][j - 1] == 0, (i + 1, j - 1), is_opponent]

    if direction == 'W':
        # print('h')
        if j - 1 < min_index:
            return [False, (), False]
        else:
            is_opponent = (board[i][j - 1] % 2 != CURRENT_PLAYER and board[i][j - 1] != 0)
            return [board[i][j - 1] == 0, (i, j - 1), is_opponent]

    if direction == 'NW':
        # print('i')
        if i - 1 < min_index or j - 1 < min_index:
            return [False, (), False]
        else:
            is_opponent = (board[i - 1][j - 1] % 2 != CURRENT_PLAYER and board[i - 1][j - 1] != 0)
            return [board[i - 1][j - 1] == 0, (i - 1, j - 1), is_opponent]


def get_P_moves(board, i, j):
    """generate list of possible moves for a pincer
       Return the following list with interior lists of the following definition:
       index 0: original position
       index 1: target position
       index 2: tuple list containing indices of pieces captured, empty tuple if no capture took place
       """
    move_list = []
    captive_list = []
    if frozen(board, i, j, CURRENT_PLAYER, 0):
        # move_list.append([(i,j),(i,j),captive_list])
        return move_list
    for direction_index in range(0, 8, 2):  # Explore N, E, S, W
        direction = DIRECTIONS[direction_index]
        # print('PAWN DIR: ' + str(direction) + ' ====================================')
        target_tile = is_vacant(board, direction, i, j)
        # print(direction, (i,j), target_tile)
        while target_tile[0]:
            target_tile_i = target_tile[1][0]
            target_tile_j = target_tile[1][1]
            captive_list = []  # list of captured pieces for move from i, j to target_i, target_j

            # Target_tile is vacant, now check for adjacent opponents and if allies behind them to see if we capture
            directions_to_check = [0, 2, 4, 6]  # Explore N, E, S, W except for direction we came from
            directions_to_check.remove(DIRECTION_TO_CODE_DICT[DIRECTION_OPPOSITES[direction]])
            for adjacent_index in directions_to_check:
                adjacent_tile = is_vacant(board, DIRECTIONS[adjacent_index], target_tile_i, target_tile_j)
                if adjacent_tile[2]:  # if tile adjacent to target_tile is occupied by enemy
                    adjacent_opponent_i = adjacent_tile[1][0]
                    adjacent_opponent_j = adjacent_tile[1][1]
                    tile_behind_opponent = is_vacant(board, DIRECTIONS[adjacent_index], adjacent_opponent_i,
                                                     adjacent_opponent_j)
                    if not tile_behind_opponent[0] and not tile_behind_opponent[2]:  # if tile behind opponent is ally
                        # If we can perform a pincer attack on an enemy piece, capture it and add it to captive list
                        captive_list.append((adjacent_opponent_i, adjacent_opponent_j))

            # Update/add to possible moves with the following list:
            # [original_index, target/destination_index, list of pieces captured by move from original to target]
            move_list.append([(i, j), (target_tile_i, target_tile_j), captive_list])

            # update target tile to continue down path of current direction
            target_tile = is_vacant(board, direction, target_tile_i, target_tile_j)
    # print('get p moves: ', move_list)
    return move_list


def get_C_moves(board, i, j):
    """generate list of possible moves for a coordinator
       Return the following list with interior lists of the following definition:
       index 0: original position
       index 1: target position
       index 2: tuple list indexes of pieces captured, empty tuple if no capture took place
       """
    move_list = []
    capture_list = []
    if frozen(board, i, j, CURRENT_PLAYER, 0):
        # move_list.append([(i,j),(i,j),captive_list])
        return move_list
    # GOAL 1: get position of current player's king so that c and k can coordinate attacks
    my_king = ''
    if CURRENT_PLAYER == BC.WHITE:
        my_king = 'K'
    else:
        my_king = 'k'
    king_location = ()
    for j in range(BOARD_LENGTH):
        if not king_location:  # if king_location not found yet
            for i in range(BOARD_LENGTH - 1, -1, -1):
                if board[i][j] == my_king:
                    king_location = (i, j)
                    break  # we found king, break out of inner loop
        else:  # we found the king, break out of outer loop
            break

    # GOAL 2: generate all possible moves, check if piece(s) captured via rules of coordinator
    for direction in DIRECTIONS:
        target_tile = is_vacant(board, direction, i, j)
        while target_tile[0]:
            target_tile_i = target_tile[1][0]
            target_tile_j = target_tile[1][1]

            capture_list = []
            upper_corner_code = board[target_tile_i][king_location[1]]
            if upper_corner_code % 2 != CURRENT_PLAYER and upper_corner_code != 0:
                capture_list.append((target_tile_i, king_location[1]))

            bottom_corner_code = board[king_location[0]][target_tile_j]
            if bottom_corner_code % 2 != CURRENT_PLAYER and bottom_corner_code != 0:
                capture_list.append((target_tile_i, target_tile_j))

            move_list.append([(i, j), (target_tile_i, target_tile_j), capture_list])

            # update target_tile to continue down current path
            target_tile = is_vacant(board, direction, target_tile_i, target_tile_j)

    return move_list


# Imitator is intimidating
# def get_I_moves(board, i, j):
#     """generate list of possible moves for an imitator
#         Return the following list with interior lists of the following definition:
#         index 0: original position
#         index 1: target position
#         index 2: tuple list of pieces captured, empty tuple if no capture took place
#         """
#     move_list = []


def get_W_moves(board, i, j):
    """generate list of possible moves for a withdrawer
       Return the following list with interior lists of the following definition:
       index 0: original position
       index 1: target position
       index 2: tuple list of pieces captured, empty tuple if no capture took place
       """
    move_list = []
    captive_list = []
    if frozen(board, i, j, CURRENT_PLAYER, 0):
        # move_list.append([(i,j),(i,j),captive_list])
        return move_list
    for direction in DIRECTIONS:
        target_tile = is_vacant(board, direction, i, j)
        withdrawee_tile = is_vacant(board, DIRECTION_OPPOSITES[direction], i, j)  # tile being withdrawn from for
        if withdrawee_tile[0] or withdrawee_tile[2]:
            withdrawee_tile_i = withdrawee_tile[1][0]
            withdrawee_tile_j = withdrawee_tile[1][1]
        while target_tile[0]:  # while target tile is vacant; also check tile in opposite direction has enemy
            target_tile_i = target_tile[1][0]
            target_tile_j = target_tile[1][1]

            if withdrawee_tile[2]:  # if there is a victim of the withdrawer, record said victim and move
                move_list.append([(i, j), (target_tile_i, target_tile_j), [(withdrawee_tile_i,
                                                                            withdrawee_tile_j)]])  # should be assigned
            else:  # no victim of withdrawer, moving as normal
                move_list.append([(i, j), (target_tile_i, target_tile_j), []])

            # update target_tile to continue down path of current direction
            target_tile = is_vacant(board, direction, target_tile_i, target_tile_j)

            # if no move for given direction, break out of loop
    return move_list


def get_K_moves(board, i, j):
    """generate list of possible moves for king
   Return the following list with interior lists of the following definition:
   index 0: original position
   index 1: target position
   index 2: list index of piece captured, empty tuple if no capture took place
   """
    move_list = []
    captive_list = []
    if frozen(board, i, j, CURRENT_PLAYER, 0):
        # move_list.append([(i,j),(i,j),captive_list])
        return move_list
    for direction in DIRECTIONS:
        target_tile = is_vacant(board, direction, i, j)
        if target_tile[0]:  # if target space is empty
            move_list.append([(i, j), (target_tile[1][0], target_tile[1][1]), ()])
        elif target_tile[2]:  # if target space occupied by enemy piece
            move_list.append([(i, j), (target_tile[1][0], target_tile[1][1]), [(target_tile[1][0], target_tile[1][1])]])
            # King occupies space of captured piece
    return move_list


def get_L_moves(board, i, j):
    """generate list of possible moves for leaper
   Return the following list with interior lists of the following definition:
   index 0: original position
   index 1: target position
   index 2: list index of piece captured, empty if no capture took place
   """
    move_list = []
    captive_list = []
    if frozen(board, i, j, CURRENT_PLAYER, 0):
        # move_list.append([(i,j),(i,j),captive_list])
        return move_list
    for direction in DIRECTIONS:
        # print('LEAPER DIR: ' + str(direction) + ' ====================================')
        target_tile = is_vacant(board, direction, i, j)
        while True:
            if target_tile[0]:
                target_tile_i = target_tile[1][0]
                target_tile_j = target_tile[1][1]
                # if the target tile is actually vacant and legal, add it to possible moves
                move_list.append([(i, j), (target_tile_i, target_tile_j), ()])

                # pursue direction to find more vacancies along a direction
                target_tile = is_vacant(board, direction, target_tile_i, target_tile_j)
            elif target_tile[2]:  # if blocked by opposing piece
                target_tile_i = target_tile[1][0]
                target_tile_j = target_tile[1][1]
                tile_behind_enemy = is_vacant(board, direction, target_tile_i, target_tile_j)
                if tile_behind_enemy[0]:
                    # print('check here akjfhasjdhflskjdfhslkjdf')
                    move_list.append([(i, j), (tile_behind_enemy[1][0], tile_behind_enemy[1][1]),
                                      [(target_tile_i, target_tile_j)]])
                    # Leaper captures piece at target_tile_i, target_tile_j;
                    # moves from i, j to vacant space behind target_tile_i, target_tile_j
            else:
                break  # no more possible moves, get out of while loop
    return move_list


def get_F_moves(board, i, j):
    """generate list of possible moves for immobilizer/freezer
   Return the following list with interior lists of the following definition:
   index 0: original position
   index 1: target position
   index 2: tuple index of piece captured, empty tuple if no capture took place"""
    move_list = []
    captive_list = []
    if frozen(board, i, j, CURRENT_PLAYER, 0):
        # move_list.append([(i,j),(i,j),captive_list])
        return move_list
    for direction in DIRECTIONS:
        # print('direction: ' + str(direction) + ' ====================================')
        target_tile = is_vacant(board, direction, i, j)
        ## EXAMPLE OF TARGET_TILE: [False, (6,0), False] [False, (), False]
        # print("target_tile: ", target_tile)
        while target_tile[0]:
            target_tile_i = target_tile[1][0]
            target_tile_j = target_tile[1][1]
            move_list.append([(i, j), (target_tile_i, target_tile_j), []])
            target_tile = is_vacant(board, direction, target_tile_i, target_tile_j)
    # print('get F moves: ', move_list)
    return move_list


def frozen(board, i, j, player, num_freezers):
    # print('FREEZING')
    for direction in DIRECTIONS:
        opponent_check = is_vacant(board, direction, i, j)
        if opponent_check[2]:  # if tile in direction from i, j is occupied by opposing piece
            # print('THERE IS OPPONENT')
            opponent_i = opponent_check[1][0]
            opponent_j = opponent_check[1][1]
            if player == BC.BLACK:  # if current player is black team
                if board[opponent_i][opponent_j] == BC.WHITE_FREEZER:
                    # if opposing piece is white freezer
                    # piece frozen but we should check for other immobilizer to see if freeze effect is canceled
                    # check if immobilizer is frozen by our frozen

                    # if opposing immobilizer is frozen, return False i.e. return that original
                    # piece being checked is free to move
                    if num_freezers == 2:  # if there are two freezers, they cancel each other
                        return False
                    else:
                        return frozen(board, opponent_i, opponent_j, other_player(player), num_freezers + 1)
            else:  # player == BC.WHITE
                # print('NO OPPOEMT')
                if board[opponent_i][opponent_j] == BC.BLACK_FREEZER:  # if opposing piece is freezer
                    if num_freezers == 2:
                        return False
                    else:
                        return frozen(board, opponent_i, opponent_j, other_player(player), num_freezers + 1)
    return num_freezers == 1  # if there is only one freezer, its effect takes hold


def other_player(player):
    if player == BC.WHITE:
        return BC.BLACK
    else:
        return BC.WHITE


# TODO: Generate moves for queen, coordinator, pincers
#       complete parameterized minimax, IDDFS, alpha beta


# TODO Make compatible/ integrate with the rest of project

def get_win(current_state):
    King = False
    king = True
    for i in range(8):
        for j in range(8):
            if current_state.board[i][j] == 13:
                King = True
            if current_state.board[j][j] == 12:
                king = False
    return King & king


def minimax_basic(current_state, plyLeft):
    """Uses alpha-beta pruning to find the optimal route."""
    global EXPLORED_STATES, alpha, beta, n_cutoffs
    if plyLeft == 0:
        return current_state.static_eval()
    if current_state.whose_move == 'W':
        provisional = -100000
    else:
        provisional = 100000
    # available_moves = PIECES_MOVES_DICT
    if len(PIECES_MOVES_DICT) == 0:
        return current_state.static_eval()

    alpha = -100000
    beta = 100000
    for m in PIECES_MOVES_DICT:
        new_state = move(current_state, m)
        if get_win(new_state) != "No win":
            return update(current_state)
        newVal = minimax_basic(new_state, plyLeft - 1)
        if current_state.whose_move == BC.WHITE and newVal > provisional:
            provisional = newVal
            alpha = max(provisional, alpha)
            if pruning and alpha >= beta:
                n_cutoffs += n_cutoffs
                break
            return alpha
        elif current_state.whose_move == BC.BLACK and newVal < provisional:
            provisional = newVal
            beta = min(provisional, beta)
            if pruning and alpha >= beta:
                n_cutoffs = n_cutoffs + 1
                break
            return beta
    return provisional


def update(current_state):
    global n_states_expanded, n_static_evals, EXPLORED_STATES
    if current_state in EXPLORED_STATES:
        return EXPLORED_STATES[current_state]
    else:
        n_states_expanded += 1
        value = current_state.static_eval()
        n_static_evals += 1
        EXPLORED_STATES[current_state] = value
        return value


def minimax(current_state, plyLeft, start, time_limit):
    """Uses alpha-beta pruning to find the optimal route."""
    global EXPLORED_STATES, alpha, beta, n_cutoffs, PIECES_MOVES_DICT
    if plyLeft == 0:
        return update(current_state)
    if current_state.whose_move == BC.WHITE:
        provisional = -100000
    else:
        provisional = 100000
    # available_moves = get_successors(current_state)
    if len(PIECES_MOVES_DICT) == 0:
        return update(current_state)

    alpha = -100000
    beta = 100000

    PIECES_MOVES_DICT = get_successors(current_state)
    for m in PIECES_MOVES_DICT:

        if time.time() - start < time_limit:
            new_state = move(current_state, m)
            if get_win(new_state) != "No win":
                return update(current_state)

            newVal = minimax(new_state, plyLeft - 1, start, time_limit)
            if current_state.whose_move == BC.WHITE and newVal > provisional:
                provisional = newVal
                alpha = max(provisional, alpha)
                if pruning and alpha >= beta:
                    n_cutoffs += n_cutoffs
                    break
                return alpha
            elif current_state.whose_move == BC.BLACK and newVal < provisional:
                provisional = newVal
                beta = min(provisional, beta)
                if pruning and alpha >= beta:
                    n_cutoffs = n_cutoffs + 1
                    break
                return beta
    return provisional


def move(current_state, li):
    new_s = BC.BC_state()  # deeply copy state by using constructor
    new_s.board = current_state.board
    new_s.whose_move = CURRENT_PLAYER
    new_s.__class__ = BC.BC_state

    # dictionary = PIECES_MOVES_DICT
    # print(li)

    original_index_i = li[1][0]
    original_index_j = li[1][1]
    # print(current_state.board[original_index_i][original_index_j])
    destination_index_i = li[2][0]
    destination_index_j = li[2][1]
    capture_list = li[3]
    # print(capture_list)
    if li[0] != 'f' or 'F':
        for captured_piece in capture_list:
            cpi = captured_piece[0]
            cpj = captured_piece[1]
            new_s.board[cpi][cpj] = 0

    new_s.board[destination_index_i][destination_index_j] = current_state.board[original_index_i][original_index_j]
    new_s.board[original_index_i][original_index_j] = 0
    # print(new_s.board[destination_index_i][destination_index_j])
    if current_state.whose_move == BC.WHITE:
        new_s.whose_move = BC.BLACK
    else:
        new_s.whose_move = BC.WHITE
    # new_s.__class__ = BC.BC_state
    print(new_s)
    # sys.exit()
    return new_s


def iterative_deepening(current_state, plyLimit, start, time_limit):
    """Finds the optimal route using IDDFS and minimax"""
    global EXPLORED_STATES, n_cutoffs, alpha, beta, PIECES_MOVES_DICT
    alpha = -100000
    beta = 100000
    depth = 0
    # available_moves = get_successors(current_state)
    best_move = None
    # print(PIECES_MOVES_DICT)
    if (time.time() - start) < time_limit:

        while plyLimit > depth:

            for m in PIECES_MOVES_DICT:

                # return [((1,2),(2,3)), current_state]
                # for m in PIECES_MOVES_DICT[p]:

                new_state = move(current_state, m)
                if get_win(new_state) != 'No win':
                    return [(m[1], m[2]), new_state]
                    # return [((1,2),(2,3)),current_state]
                newVal = minimax(new_state, depth, start, time_limit)
                # best_move = [[m[0], m[1]], new_state]

                if current_state.whose_move == BC.WHITE:
                    if newVal > alpha:
                        alpha = newVal
                        best_move = [(m[1], m[2]), new_state]
                        if pruning:
                            if alpha >= beta:
                                n_cutoffs = n_cutoffs + 1
                elif current_state.whose_move == BC.BLACK:
                    if newVal < beta:
                        beta = newVal
                        best_move = [(m[1], m[2]), new_state]
                        if pruning:
                            if alpha >= beta:
                                n_cutoffs = n_cutoffs + 1
                else:
                    best_move = [False, current_state]

            depth += 1
    return best_move


def makeMove(currentState, currentRemark, timelimit):
    # Compute the new state for a move.
    # You should implement an anytime algorithm based on IDDFS.

    # The following is a placeholder that just copies the current state.
    #    newState = BC.BC_state(currentState.board)
    #
    #    # Fix up whose turn it will be.
    #    newState.whose_move = 1 - currentState.whose_move
    #
    #    # Construct a representation of the move that goes from the
    #    # currentState to the newState.
    #    # Here is a placeholder in the right format but with made-up
    #    # numbers:
    #    move = ((6, 4), (3, 4))
    #
    #    # Make up a new remark
    #    newRemark = "I'll think harder in some future game. Here's my move"
    #
    #    return [[move, newState], newRemark]
    # global EXPLORED_STATES
    start = time.time()
    # current_state.__class__ = MY_TTS_State
    moveIndexs = [False, currentState]
    depth = 0
    stop_time = timelimit
    global PIECES_MOVES_DICT
    PIECES_MOVES_DICT = get_successors(currentState)
    while time.time() - start < 0.9 * timelimit:
        moveIndexs = iterative_deepening(currentState, depth + 1, start, stop_time)
    # PIECES_MOVES_DICT = get_successors(moveIndexs[1])
    if moveIndexs == None or moveIndexs[0] == False:
        return [[False, currentState], "Good game, it seems we are tied!"]

    return [[moveIndexs[0], moveIndexs[1]], 'good']


def zhash(board):
    global zobristnum
    b = [square for row in board for square in row]
    val = 0
    for i in range(len(b)):
        piece = None

        if b[i] in BC.INIT_TO_CODE:
            piece = BC.INIT_TO_CODE[b[i]]

        if piece is not None:
            val ^= zobristnum[i][piece]
    return val


def nickname():
    return "Newman"


def introduce():
    return "I'm Newman Cat, a newbie Baroque Chess agent."


def prepare(player2Nickname):
    """ Here the game master will give your agent the nickname of
    the opponent agent, in case your agent can use it in some of
    the dialog responses.  Other than that, this function can be
    used for initializing data structures, if needed."""
    global zobristnum
    S = 64
    P = 8
    zobristnum = [[0] * P for i in range(S)]
    for i in range(S):
        for j in range(P):
            zobristnum[i][j] = randint(0, 1000000000)

    return "OK"


def basicStaticEval(state):
    """Use the simple method for state evaluation described in the spec.
    This is typically used in parameterized_minimax calls to verify
    that minimax and alpha-beta pruning work correctly."""
    score = 0
    board = state.board
    letter_to_code_dict = BC.INIT_TO_CODE
    for i in range(BOARD_LENGTH):
        for j in range(BOARD_LENGTH):
            if board[i][j] == 'P':
                score += 1
            elif board[i][j] == 'p':
                score -= 1
            elif board[i][j] == 'K':
                score += 100
            elif board[i][j] == 'k':
                score -= 100
            elif board[i][j] != '-':
                if letter_to_code_dict[board[i][j]] % 2 == 1:
                    score += 2
                else:
                    score -= 2
    return score


def staticEval(state):
    """Compute a more thorough static evaluation of the given state.
    This is intended for normal competitive play.  How you design this
    function could have a significant impact on your player's ability
    to win games."""
    pass


class My_Priority_Queue:

    def __init__(self):
        self.q = []  # Actual data goes in a list.

    def __contains__(self, elt):
        """If there is a (state, priority) pair on the list
    where state==elt, then return True."""
        # print("In My_Priority_Queue.__contains__: elt= ", str(elt))
        for pair in self.q:
            if pair[0] == elt: return True
        return False

    def delete_min(self):
        """ Standard priority-queue dequeuing method."""
        if not self.q:
            return []  # Simpler than raising an exception.
        temp_min_pair = self.q[0]
        temp_min_value = temp_min_pair[1]
        temp_min_position = 0
        for j in range(1, len(self.q)):
            if self.q[j][1] < temp_min_value:
                temp_min_pair = self.q[j]
                temp_min_value = temp_min_pair[1]
                temp_min_position = j
        del self.q[temp_min_position]
        return temp_min_pair

    def insert(self, state, priority):
        """We do not keep the list sorted, in this implementation."""
        # print("calling insert with state, priority: ", state, priority)
        if self[state] != -1:
            print("Error: You're trying to insert an element into a My_Priority_Queue instance,")
            print(" but there is already such an element in the queue.")
            return
        self.q.append((state, priority))

    def __len__(self):
        """We define length of the priority queue to be the
  length of its list."""
        return len(self.q)

    def __getitem__(self, state):
        '''This method enables Pythons right-bracket syntax.
    Here, something like  priority_val = my_queue[state]
    becomes possible. Note that the syntax is actually used
    in the insert method above:  self[state] != -1  '''
        for (S, P) in self.q:
            if S == state: return P
        return -1  # This value means not found.

    def __delitem__(self, state):
        """This method enables Python's del operator to delete
    items from the queue."""
        # print("In MyPriorityQueue.__delitem__: state is: ", str(state))
        for count, (S, P) in enumerate(self.q):
            if S == state:
                del self.q[count]
                return

    def __str__(self):
        txt = "My_Priority_Queue: ["
        for (s, p) in self.q: txt += '(' + str(s) + ',' + str(p) + ') '
        txt += ']'
        return txt
