"""PlayerSkeletonA.py
The beginnings of an agent that might someday play Baroque Chess.

"""
import math
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
    CURRENT_PLAYER = currentState.whose_move
    global OTHER_PLAYER
    if CURRENT_PLAYER == 0:
        OTHER_PLAYER = 1
    else:
        OTHER_PLAYER = 0
    list_of_successors = get_successors(currentState)
    DATA = {'CURRENT_STATE_STATIC_VAL': -1000.0,
            'N_STATES_EXPANDED': 0,
            'N_STATIC_EVALS': 0,
            'N_CUTOFFS': 0}
    if alphaBeta:
        alpha_beta_minimax_helper(currentState, ply, list_of_successors, DATA)
    else:
        general_minimax_helper(currentState, ply, list_of_successors, DATA)

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
    # global PIECES_MOVES_DICT

    list_of_possible_moves = []

    # print('IN GET SUCCESSORS AFTER INITIALIZING DICT TO EMPTY')
    for j in range(BOARD_LENGTH):
        for i in range(BOARD_LENGTH - 1, -1, -1):
            piece = board[i][j]
            # print('PIECE AT GET SUC == ' + str(BC.CODE_TO_INIT[piece]))
            if CURRENT_PLAYER == BC.WHITE:
                if piece == BC.WHITE_FREEZER:  # white freezer
                    # print('GETTING F MOVES')
                    f_moves = get_F_moves(board, i, j)
                    if f_moves:
                        for k in range(len(f_moves)):
                            # print('F MOVES: ' + str(f_moves))
                            list_of_possible_moves.append([piece, (i, j), f_moves[k][1], f_moves[k][2]])

                elif piece == BC.WHITE_LEAPER:  # leaper
                    l_moves = get_L_moves(board, i, j)
                    if l_moves:
                        for k in range(len(l_moves)):
                            list_of_possible_moves.append([piece, l_moves[k][0], l_moves[k][1],
                                                           l_moves[k][2]])

                elif piece == BC.WHITE_KING:  # king
                    k_moves = get_K_moves(board, i, j)
                    if k_moves:
                        for k in range(len(k_moves)):
                            list_of_possible_moves.append([piece, k_moves[k][0], k_moves[k][1],
                                                           k_moves[k][2]])

                elif piece == BC.WHITE_WITHDRAWER:  # withdrawer
                    w_moves = get_W_moves(board, i, j)
                    if w_moves:
                        for k in range(len(w_moves)):
                            list_of_possible_moves.append([piece, w_moves[k][0], w_moves[k][1],
                                                           w_moves[k][2]])

                elif piece == BC.WHITE_COORDINATOR:  # Coordinator
                    c_moves = get_C_moves(board, i, j)
                    if c_moves:
                        for k in range(len(c_moves)):
                            list_of_possible_moves.append([piece, c_moves[k][0], c_moves[k][1],
                                                           c_moves[k][2]])

                elif piece == BC.WHITE_PINCER:  # Pincer
                    # print('getting upper case P at (' + str(i) + ', ' + str(j) + ')')
                    p_moves = get_P_moves(board, i, j)

                    if p_moves:
                        for k in range(len(p_moves)):
                            list_of_possible_moves.append([piece, p_moves[k][0], p_moves[k][1],
                                                           p_moves[k][2]])

            else:  # CURRENT_PLAYER == BC.BLACK
                if piece == BC.BLACK_FREEZER:  # BLACK freezer
                    f_moves = get_F_moves(board, i, j)
                    if f_moves:
                        for k in range(len(f_moves)):
                            list_of_possible_moves.append([piece, f_moves[k][0], f_moves[k][1],
                                                           f_moves[k][2]])

                elif piece == BC.BLACK_LEAPER:  # leaper
                    l_moves = get_L_moves(board, i, j)
                    if l_moves:
                        for k in range(len(l_moves)):
                            list_of_possible_moves.append([piece, l_moves[k][0], l_moves[k][1],
                                                           l_moves[k][2]])

                elif piece == BC.BLACK_KING:  # king
                    k_moves = get_K_moves(board, i, j)
                    if k_moves:
                        for k in range(len(k_moves)):
                            list_of_possible_moves.append([piece, k_moves[k][0], k_moves[k][1],
                                                           k_moves[k][2]])

                elif piece == BC.BLACK_WITHDRAWER:  # black withdrawer
                    w_moves = get_W_moves(board, i, j)
                    if w_moves:
                        for k in range(len(w_moves)):
                            list_of_possible_moves.append([piece, w_moves[k][0], w_moves[k][1],
                                                           w_moves[k][2]])

                elif piece == BC.BLACK_COORDINATOR:  # coordinator
                    c_moves = get_C_moves(board, i, j)
                    if c_moves:
                        for k in range(len(c_moves)):
                            list_of_possible_moves.append([piece, c_moves[k][0], c_moves[k][1],
                                                           c_moves[k][2]])

                elif piece == BC.BLACK_PINCER:  # pincer
                    # print('calling lower case p at (' + str(i) + ', ' + str(j) + ')')
                    p_moves = get_P_moves(board, i, j)
                    if p_moves:
                        for k in range(len(p_moves)):
                            list_of_possible_moves.append([piece, p_moves[k][0], p_moves[k][1],
                                                           p_moves[k][2]])
    # print('list after call on get suc ' + str(list_of_possible_moves))
    return list_of_possible_moves


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
            is_opponent = (board[i + 1][j + 1] % 2 != CURRENT_PLAYER and board[i + 1][j + 1] != 0)
            return [board[i + 1][j + 1] == 0, (i + 1, j + 1), is_opponent]

    if direction == 'S':
        # print('f')
        if i + 1 > max_index:
            return [False, (), False]
        else:
            is_opponent = (board[i + 1][j] % 2 != CURRENT_PLAYER and board[i + 1][j] != 0)
            return [board[i + 1][j] == 0, (i + 1, j), is_opponent]

    if direction == 'SW':

        if i + 1 > max_index or j - 1 < min_index:
            return [False, (), False]
        else:
            is_opponent = (board[i + 1][j - 1] % 2 != CURRENT_PLAYER and board[i + 1][j - 1] != 0)
            return [board[i + 1][j - 1] == 0, (i + 1, j - 1), is_opponent]

    if direction == 'W':

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
    # print('in get p moves')
    move_list = []
    # captive_list = []
    if frozen(board, i, j, CURRENT_PLAYER, 0):
        # move_list.append([(i,j),(i,j),captive_list])
        return move_list
    for direction_index in range(0, 8, 2):  # Explore N, E, S, W
        direction = DIRECTIONS[direction_index]
        #  #  print('PAWN DIR: ' + str(direction) + ' ====================================')
        target_tile = is_vacant(board, direction, i, j)
        #  # print(str(direction) + ' ' + str((i, j)) + ' target_tile')
        while target_tile[0]:
            target_tile_i = target_tile[1][0]
            target_tile_j = target_tile[1][1]
            # print('in while loop')
            captive_list = []  # list of captured pieces for move from i, j to target_i, target_j

            # Target_tile is vacant, now check for adjacent opponents and if allies behind them to see if we capture
            directions_to_check = [0, 2, 4, 6]  # Explore N, E, S, W except for direction we came from
            directions_to_check.remove(DIRECTION_TO_CODE_DICT[DIRECTION_OPPOSITES[direction]])
            for adjacent_of_target_tile in directions_to_check:
                adjacent_tile = is_vacant(board, DIRECTIONS[adjacent_of_target_tile], target_tile_i, target_tile_j)
                if adjacent_tile[2]:  # if tile adjacent to target_tile is occupied by enemy
                    adjacent_opponent_i = adjacent_tile[1][0]
                    adjacent_opponent_j = adjacent_tile[1][1]
                    tile_behind_opponent = is_vacant(board, DIRECTIONS[adjacent_of_target_tile], adjacent_opponent_i,
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
        my_king = BC.WHITE_KING
    else:
        my_king = BC.BLACK_KING
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
        Done = False
        while not Done:
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
                # break  # no more possible moves, get out of while loop
                Done = True
            # print('watt')
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
        return move_list
    for direction in DIRECTIONS:
        # print('direction: ' + str(direction) + ' ====================================')
        # print('at F directions')
        target_tile = is_vacant(board, direction, i, j)
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
def find_next_possible_move(board, current_player):
    global CURRENT_PLAYER
    CURRENT_PLAYER = current_player
    for j in range(BOARD_LENGTH):
        for i in range(BOARD_LENGTH - 1, -1, -1):
            if board[i][j] == BC.BLACK_PINCER + current_player:
                move_list = get_P_moves(board, i, j)
                if move_list:
                    return move_list[0]
            elif board[i][j] == BC.BLACK_FREEZER + current_player:
                move_list = get_F_moves(board, i, j)
                if move_list:
                    return move_list[0]
            elif board[i][j] == BC.BLACK_LEAPER + current_player:
                move_list = get_L_moves(board, i, j)
                if move_list:
                    return move_list[0]
            elif board[i][j] == BC.BLACK_KING + current_player:
                move_list = get_K_moves(board, i, j)
                if move_list:
                    return move_list[0]
            elif board[i][j] == BC.BLACK_WITHDRAWER + current_player:
                move_list = get_W_moves(board, i, j)
                if move_list:
                    return move_list[0]
            elif board[i][j] == BC.BLACK_COORDINATOR + current_player:
                move_list = get_C_moves(board, i, j)
                if move_list:
                    return move_list[0]
    return False


def alpha_beta_minimax_helper(current_state, depth, list_of_successors, DATA):
    action_value_dict = {}
    start_to_move_map = {}
    alpha = -math.inf
    if list_of_successors:
        best_action = list_of_successors[0]
        DATA["N_STATES_EXPANDED"] += 1
        for single_move_list in list_of_successors:
            # print('SINGLE MOVE LIST IN HELPER: ' + str(single_move_list) + ' 88888888888888888888888888888888888888888')
            new_state = move(current_state, single_move_list)
            # print(list_of_successors)
            # pass down alpha value?
            evaluation = alpha_beta_minimax(new_state, depth - 1, alpha, math.inf, False,
                                            get_successors(new_state), DATA)
            alpha = max(alpha, evaluation)
            action_key = (single_move_list[0], single_move_list[1], single_move_list[2], single_move_list[3])
            action_value_dict[single_move_list[1]] = evaluation
            start_to_move_map[single_move_list[1]] = action_key
            # print('in helper for loop')
            # print('action list: ' + str(single_move_list))
            if time.time() - START_TIME >= (0.80 * TIME_LIMIT):
                break
        # find best action among initial actions
        # print(' out of for loop 7777777777777777777777777777777777777777777777777777777777777777777777777777777777')
        for starting_index in action_value_dict:
            if action_value_dict[starting_index] > action_value_dict[best_action[1]]:
                best_action = start_to_move_map[starting_index]
    else:
        best_action = find_next_possible_move(current_state.board, current_state.whose_move)
        new_state = move(current_state, best_action)
        action_value_dict[best_action[1]] = get_static_eval(new_state)

    DATA["CURRENT_STATE_STATIC_VAL"] = action_value_dict[best_action[1]]
    return best_action, DATA["CURRENT_STATE_STATIC_VAL"]


def alpha_beta_minimax(state, depth, alpha, beta, is_maximizing_player, list_of_successors, DATA):
    if depth <= 0:
        # print('in ab mini if check')
        if time.time() - START_TIME >= (0.80 * TIME_LIMIT):
            # print(1)
            return DATA["CURRENT_STATE_STATIC_VAL"]
        else:
            # print(2)
            DATA['N_STATIC_EVALS'] += 1
            return get_static_eval(state)
    # print(' in ab mini')
    if is_maximizing_player:
        max_evaluation = -math.inf
        DATA["N_STATES_EXPANDED"] += 1
        for single_move_list in list_of_successors:
            # print(' ab maxer expansion')
            new_state = move(state, single_move_list)
            evaluation = alpha_beta_minimax(new_state, depth - 1, alpha, beta, False, get_successors(new_state),
                                            DATA)
            max_evaluation = max(max_evaluation, evaluation)
            alpha = max(alpha, evaluation)
            if beta <= alpha:
                DATA["N_CUTOFFS"] += 1
                break
            if time.time() - START_TIME >= (0.80 * TIME_LIMIT):
                break
        return max_evaluation
    else:  # is minimizing player's turn
        min_evaluation = math.inf
        DATA["N_STATES_EXPANDED"] += 1
        for single_move_list in list_of_successors:
            # print(' ab miner expansion')
            new_state = move(state, single_move_list)
            evaluation = alpha_beta_minimax(new_state, depth - 1, alpha, beta, True, get_successors(new_state),
                                            DATA)
            min_evaluation = min(min_evaluation, evaluation)
            beta = min(beta, evaluation)
            if beta <= alpha:
                DATA["N_CUTOFFS"] += 1
                break
            if time.time() - START_TIME >= (0.80 * TIME_LIMIT):
                break
        return min_evaluation


def alpha_beta_IDDFS(state, depth, list_of_successors, DATA):
    static_eval = get_static_eval(state)
    single_move_list = find_next_possible_move(state.board, state.whose_move)
    global START_TIME
    for i in range(depth + 1):
        # print('in ab iddfs')
        # print('depth of dfs: ' + str(i))
        if time.time() - START_TIME >= (0.80 * TIME_LIMIT):
            break
        else:
            single_move_list, static_eval = alpha_beta_minimax_helper(state, i, list_of_successors, DATA)
    return single_move_list, static_eval


def general_minimax_helper(current_state, depth, list_of_successors, DATA):
    action_value_dict = {}
    start_to_move_map = {}
    if list_of_successors:
        BEST_ACTION = list_of_successors[0]
        max_eval = -math.inf
        DATA["N_STATES_EXPANDED"] += 1
        for single_move_list in list_of_successors:
            new_state = move(current_state, single_move_list)
            evaluation = general_minimax(new_state, depth - 1, False, get_successors(new_state), DATA)
            action_key = (single_move_list[0], single_move_list[1], single_move_list[2], single_move_list[3])
            action_value_dict[single_move_list[1]] = evaluation
            start_to_move_map[single_move_list[1]] = action_key
            if action_value_dict[single_move_list[1]] > max_eval:
                BEST_ACTION = single_move_list
                max_eval = action_value_dict[single_move_list[1]]
            # action_value_dict[BEST_ACTION] = evaluation
            if time.time() - START_TIME >= (0.80 * TIME_LIMIT):
                break
    else:
        BEST_ACTION = find_next_possible_move(current_state.board, current_state.whose_move)

    DATA["CURRENT_STATE_STATIC_VAL"] = action_value_dict[BEST_ACTION]
    return BEST_ACTION, DATA["CURRENT_STATE_STATIC_VAL"]


def general_minimax(state, depth, is_maximizing_player, list_of_successors, DATA):
    if depth <= 0:
        if time.time() - START_TIME >= (0.80 * TIME_LIMIT):
            return DATA["CURRENT_STATE_STATIC_VAL"]
        else:
            DATA['N_STATIC_EVALS'] += 1
            return get_static_eval(state)

    if is_maximizing_player:
        max_evaluation = -math.inf
        DATA["N_STATES_EXPANDED"] += 1
        for single_move_list in list_of_successors:
            # single_move_list is of the following form:
            # index 0: piece being moved
            # index 1: original index of piece
            # index 2: destination index of piece
            # index 3: list of captured pieces for given move
            new_state = move(state, single_move_list)

            evaluation = general_minimax(new_state, depth - 1, False, get_successors(new_state),
                                         DATA)

            max_evaluation = max(max_evaluation, evaluation)
            if time.time() - START_TIME >= (0.80 * TIME_LIMIT):
                break
        return max_evaluation
    else:  # is minimizing player's turn
        min_evaluation = math.inf
        DATA["N_STATES_EXPANDED"] += 1
        for single_move_list in list_of_successors:
            new_state = move(state, single_move_list)
            evaluation = general_minimax(new_state, depth - 1, True, get_successors(new_state),
                                         DATA)
            min_evaluation = min(min_evaluation, evaluation)
            if time.time() - START_TIME >= (0.80 * TIME_LIMIT):
                break
        return min_evaluation


def general_minimax_IDDFS(state, depth, list_of_actions, DATA):
    static_eval = get_static_eval(state)
    single_move_list = find_next_possible_move(state.board, state.whose_move)
    global START_TIME
    for i in range(depth + 1):
        if time.time() - START_TIME >= (0.80 * TIME_LIMIT):
            break
        else:
            single_move_list, static_eval = general_minimax_helper(state, i, list_of_actions, DATA)
    return single_move_list, static_eval


def move(current_state, single_move_list):
    new_s = BC.BC_state()  # deeply copy state by using constructor
    new_s.__class__ = BC.BC_state
    new_s.board = current_state.board
    if CURRENT_PLAYER == BC.WHITE:
        new_s.whose_move = BC.BLACK
    else:
        new_s.whose_move = BC.WHITE

    original_index_i = single_move_list[1][0]
    original_index_j = single_move_list[1][1]

    destination_index_i = single_move_list[2][0]
    destination_index_j = single_move_list[2][1]
    capture_list = single_move_list[3]

    # assign  0 to indices of pieces captured i.e. '-'
    for captured_piece in capture_list:
        cpi = captured_piece[0]
        cpj = captured_piece[1]
        new_s.board[cpi][cpj] = 0

    new_s.board[destination_index_i][destination_index_j] = current_state.board[original_index_i][original_index_j]
    new_s.board[original_index_i][original_index_j] = 0

    # print(new_s)
    # sys.exit()
    return new_s


START_TIME = time.time()
TIME_LIMIT = 30.0


def makeMove(currentState, currentRemark, timelimit):
    global START_TIME
    START_TIME = time.time()
    global TIME_LIMIT
    TIME_LIMIT = timelimit
    current_player = currentState.whose_move
    new_player = BC.BLACK
    if current_player == BC.BLACK:
        new_player = BC.WHITE

    new_state = BC.BC_state()
    new_state.board = currentState.board
    new_state.whose_move = currentState.whose_move
    # new_state.__class__ = BC.BC_state
    DATA = {'CURRENT_STATE_STATIC_VAL': -1000.0,
            'N_STATES_EXPANDED': 0,
            'N_STATIC_EVALS': 0,
            'N_CUTOFFS': 0}
    chosen_move, static_eval = alpha_beta_IDDFS(new_state, 3, get_successors(currentState), DATA)
    # print('chosen_move: ' + str('chosen_move'))
    original_index = chosen_move[1]
    destination_index = chosen_move[2]
    new_state = move(new_state, chosen_move)
    # print('not here I guess')
    global retort_index
    new_utterance = list_of_retorts[retort_index % len(list_of_retorts)]
    retort_index += 1

    return [[(original_index, destination_index), new_state], new_utterance]

retort_index = 0
list_of_retorts = [
                   "One does not simply walk into Mordor",
                   "'Ave you lost yer mind?",
                   "Run you fools",
                   "You know nothing",
                   "Pointless, like tears in the rain" ,
                   "Que sera, sera",
                   "That was your move? really? you can do better than that, dear.",
                   "It's over, Anakin, I have the high ground",
                   "There is no hope",
                   "He's too dangerous to be left alive",
                   "Sorry I can't hear you, I'm listening to Nights Beat Switch",
                   "Be sure to listen to Frank Ocean and Childish Gambino",
                   "Is the game still going?",
                   "I don't want to be anymore",
                   "Omae wa mou shindeiru",
                   "We end our day up on the roof",
                   "Do they sew wings on tailored suits",
                   "Do or do not, there is no try",
                   "This is goodbye, old friend.",
                   "Pointless, like tears in the rain",
                   "Eat the ultra-wealthy",
                   "Nothing really matters"
                   "Nothing really matters, to meeeeeeeeeeeeee",
                   "anywhere the wind blows"
                                            ]


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


def get_static_eval(state):
    if USE_CUSTOM_STATIC_EVAL_FUNCTION:
        return staticEval(state)
    else:
        return basicStaticEval(state)


def basicStaticEval(state):
    """Use the simple method for state evaluation described in the spec.
    This is typically used in parameterized_minimax calls to verify
    that minimax and alpha-beta pruning work correctly."""
    score = 0
    board = state.board
    # letter_to_code_dict = BC.INIT_TO_CODE
    for i in range(BOARD_LENGTH):
        for j in range(BOARD_LENGTH):
            if board[i][j] == BC.WHITE_PINCER:
                score += 1
            elif board[i][j] == BC.BLACK_PINCER:
                score -= 1
            elif board[i][j] == BC.WHITE_KING:
                score += 100
            elif board[i][j] == BC.BLACK_KING:
                score -= 100
            elif board[i][j] != 0:
                if board[i][j] % 2 == BC.WHITE:
                    score += 2
                else:  # == BC.Black
                    score -= 2
    return score


def staticEval(state):
    """Compute a more thorough static evaluation of the given state.
    This is intended for normal competitive play.  How you design this
    function could have a significant impact on your player's ability
    to win games."""
    score = 0
    num_white_pieces = 0
    num_black_pieces = 0
    board = state.board
    for i in range(BOARD_LENGTH):
        for j in range(BOARD_LENGTH):
            piece = board[i][j]
            if piece % 2 == BC.WHITE:  # white pieces, odd
                num_white_pieces += 1
                if is_threatened(board, i, j, BC.WHITE):
                    if piece == BC.WHITE_KING:
                        score -= 100000
                    elif piece == BC.WHITE_WITHDRAWER:
                        score -= 1000
                    elif piece == BC.WHITE_LEAPER:
                        score -= 1200
                    elif piece == BC.WHITE_COORDINATOR:
                        score -= 900
                    elif piece == BC.WHITE_FREEZER:
                        score -= 1500
                    elif piece == BC.WHITE_PINCER:
                        score -= 300
            elif piece % 2 == 0 and piece is not 0:
                num_black_pieces += 1
                if is_threatened(board, i, j, BC.BLACK):
                    if piece == BC.BLACK_KING:
                        score += 100000
                    elif piece == BC.BLACK_WITHDRAWER:
                        score += 1000
                    elif piece == BC.BLACK_LEAPER:
                        score += 1200
                    elif piece == BC.BLACK_COORDINATOR:
                        score += 900
                    elif piece == BC.BLACK_FREEZER:
                        score += 1500
                    elif piece == BC.BLACK_PINCER:
                        score += 300
    return 3 * (num_white_pieces - num_black_pieces) + score


# if adj_enemy = dict[enemy, withdrawer]; dict[enemy, w]: code_value
ENEMY_PIECES_DICT = {
    (BC.WHITE, 'W'): BC.WHITE_WITHDRAWER,
    (BC.WHITE, 'K'): BC.WHITE_KING,
    (BC.WHITE, 'F'): BC.WHITE_FREEZER,
    (BC.WHITE, 'P'): BC.WHITE_PINCER,
    (BC.WHITE, 'C'): BC.WHITE_COORDINATOR,
    (BC.WHITE, 'L'): BC.WHITE_LEAPER,
    (BC.BLACK, 'W'): BC.BLACK_WITHDRAWER,
    (BC.BLACK, 'K'): BC.BLACK_KING,
    (BC.BLACK, 'F'): BC.BLACK_FREEZER,
    (BC.BLACK, 'P'): BC.BLACK_PINCER,
    (BC.BLACK, 'C'): BC.BLACK_COORDINATOR,
    (BC.BLACK, 'L'): BC.BLACK_LEAPER,
}


def is_threatened(board, i, j, player):
    """See if given piece is threatened by the enemy or not"""
    # find location of enemy king and who our enemy is
    if player is BC.WHITE:
        ENEMY = BC.BLACK
        enemy_king = BC.BLACK_KING
    else:
        ENEMY = BC.WHITE
        enemy_king = BC.WHITE_KING

    enemy_king_location = ()
    for i_k in range(BOARD_LENGTH):
        if not enemy_king_location:
            for j_k in range(BOARD_LENGTH):
                if BC.CODE_TO_INIT[board[i_k][j_k]] == enemy_king:
                    enemy_king_location = i_k, j_k

        # check if threatened by every kind of piece i guess
        directions_to_pursue = {}
        for direction in DIRECTIONS:
            adjacent_tile = is_vacant(board, direction, i, j)
            if adjacent_tile[2]:
                adjacent_enemy_i = adjacent_tile[1][0]
                adjacent_enemy_j = adjacent_tile[1][1]
                adjacent_enemy = board[adjacent_enemy_i][adjacent_enemy_j]
                if adjacent_enemy == ENEMY_PIECES_DICT[ENEMY, 'P'] and direction in ['N', 'E', 'S', 'W']:
                    return True
                elif adjacent_enemy == ENEMY_PIECES_DICT[ENEMY, 'W']:
                    tile_behind_withdrawer = is_vacant(board, direction, adjacent_enemy_i, adjacent_enemy_j)
                    if tile_behind_withdrawer[0]:
                        return True
                elif adjacent_enemy == ENEMY_PIECES_DICT[ENEMY, 'L']:
                    tile_behind_me = is_vacant(board, DIRECTION_OPPOSITES[direction], i, j)
                    if tile_behind_me[0]:
                        return True
                elif adjacent_enemy == ENEMY_PIECES_DICT[ENEMY, 'F']:
                    return True
                elif adjacent_enemy == ENEMY_PIECES_DICT[ENEMY, 'K']:
                    return True
                elif adjacent_enemy == ENEMY_PIECES_DICT[ENEMY, 'C']:
                    if enemy_king_location[1] == j and adjacent_enemy_i == i:
                        return True
                    elif enemy_king_location[1] == adjacent_enemy_j and adjacent_enemy_i == i:
                        return True
            if adjacent_tile[0]:
                directions_to_pursue[direction] = (adjacent_tile[1][0], adjacent_tile[1][1])
        while directions_to_pursue:
            still_pursuing = {}
            for direction in directions_to_pursue:
                tile_to_check = is_vacant(board, direction, directions_to_pursue[direction][0],
                                          directions_to_pursue[direction][1])
                if tile_to_check[2]:  # if there is an enemy
                    tile_to_check_enemy_i = tile_to_check[1][0]
                    tile_to_check_enemy_j = tile_to_check[1][1]
                    enemy_piece = board[tile_to_check_enemy_i][tile_to_check_enemy_j]
                    if enemy_piece == ENEMY_PIECES_DICT[ENEMY, 'L']:
                        tile_behind_me = is_vacant(board, DIRECTION_OPPOSITES[direction], i, j)
                        if tile_behind_me[0]:
                            return True
                    elif enemy_piece == ENEMY_PIECES_DICT[ENEMY, 'C']:
                        if enemy_king_location[1] == j and tile_to_check_enemy_i == i:
                            return True
                        elif enemy_king_location[1] == tile_to_check_enemy_j and tile_to_check_enemy_i == i:
                            return True
                elif tile_to_check[0]:  # if dir is still valid and empty
                    still_pursuing[direction] = (tile_to_check[1][0], tile_to_check[1][1])
            directions_to_pursue = still_pursuing
    return False