"""
agent that plays Toro-Tile Straight,
by Andrew Garwood
ID: 1835687


"""
import math
import time
from queue import Queue
from TTS_State import TTS_State

USE_CUSTOM_STATIC_EVAL_FUNCTION = True


class vector_2D:
    def __init__(self, i, j, name):
        self.i = i
        self.j = j
        self.name = name
        self.length = 0

    def __str__(self):
        return self.name


class MY_TTS_State(TTS_State):
    def static_eval(self):
        if USE_CUSTOM_STATIC_EVAL_FUNCTION:
            return self.custom_static_eval()
        else:
            return self.basic_static_eval()

    def basic_static_eval(self):
        TWF = 0
        TBF = 0
        board_length = len(self.board)
        for i in range(board_length):
            for j in range(board_length):
                current_tile = self.board[i][j]
                North = ((i - 1) % board_length, j)
                Northeast = ((i - 1) % board_length, (j + 1) % board_length)
                East = (i, (j + 1) % board_length)
                Southeast = ((i + 1) % board_length, (j + 1) % board_length)
                South = ((i + 1) % board_length, j)
                Southwest = ((i + 1) % board_length, (j - 1) % board_length)
                West = (i, (j - 1) % board_length)
                Northwest = ((i - 1) % board_length, (j - 1) % board_length)
                if self.is_vacant(North[0], North[1]):
                    if current_tile == 'W':
                        TWF += 1
                    elif current_tile == 'B':
                        TBF += 1
                if self.is_vacant(Northeast[0], Northeast[1]):
                    if current_tile == 'W':
                        TWF += 1
                    elif current_tile == 'B':
                        TBF += 1
                if self.is_vacant(East[0], East[1]):
                    if current_tile == 'W':
                        TWF += 1
                    elif current_tile == 'B':
                        TBF += 1
                if self.is_vacant(Southeast[0], Southeast[1]):
                    if current_tile == 'W':
                        TWF += 1
                    elif current_tile == 'B':
                        TBF += 1
                if self.is_vacant(South[0], South[1]):
                    if current_tile == 'W':
                        TWF += 1
                    elif current_tile == 'B':
                        TBF += 1
                if self.is_vacant(Southwest[0], Southwest[1]):
                    if current_tile == 'W':
                        TWF += 1
                    elif current_tile == 'B':
                        TBF += 1
                if self.is_vacant(West[0], West[1]):
                    if current_tile == 'W':
                        TWF += 1
                    elif current_tile == 'B':
                        TBF += 1
                if self.is_vacant(Northwest[0], Northwest[1]):
                    if current_tile == 'W':
                        TWF += 1
                    elif current_tile == 'B':
                        TBF += 1
        return TWF - TBF

    def is_vacant(self, i, j):
        return self.board[i][j] == ' '

    def custom_static_eval(self):
        board_length = len(self.board)
        score = 0
        for i in range(board_length):
            for j in range(board_length):
                current_tile = self.board[i][j]
                neighbors = self.get_neighbors(i, j)
                if current_tile == MY_CHARACTER:  # then find length of line and if it's blocked
                    neighbors = self.find_length_and_if_blocked(i, j, neighbors, MY_CHARACTER, [])
                    for vector in neighbors:
                        if vector.length == K:  # we found a winning line. dk if this works
                            return score + 100000000
                        elif vector.length >= 0:
                            score += vector.length * vector.length
                elif current_tile == OPPONENT:
                    neighbors = self.find_length_and_if_blocked(i, j, neighbors, OPPONENT, [])
                    for vector in neighbors:
                        if vector.length >= K - 2:  # don't want this to happen
                            score -= 1000000
                        # elif vector.length < 0:  # blocking is good
                        #     score += 10000
                        elif vector.length >= 0:
                            score -= vector.length * vector.length
        return score

    def find_length_and_if_blocked(self, i, j, neighbors, character, seen):
        board = self.board
        for direction in neighbors:
            if board[direction.i][direction.j] == character and vector_2D(i, j, direction.name) not in seen:
                seen.append(vector_2D(i, j, direction.name))
                direction.length += 1
                direction.length += self.pursue_direction(direction.i, direction.j, self.get_direction(direction),
                                                          character, seen)
        return neighbors

    def pursue_direction(self, i, j, direction, character, seen):  # pursue individual direction rather than all 8
        board = self.board
        # if character == 'W':  # was going to be used to test blocking
        #     other_character = 'B'
        # else:
        #     other_character = 'W'

        if board[direction.i][direction.j] == character and vector_2D(i, j, direction.name) not in seen:
            seen.append(vector_2D(i, j, direction.name))
            return 1 + self.pursue_direction(direction.i, direction.j, self.get_direction(direction), character, seen)
        # elif board[i][j] == other_character or board[i][j] == '-':
        #     return -1  # negative number or dash indicates direction is blocked?
        elif board[i][j] == ' ':
            return 0 + self.pursue_direction(direction.i, direction.j, self.get_direction(direction), character, seen)
        else:
            return 0

    def get_direction(self, vector):  # returns vector in same direction with new coords. 
        name = vector.name
        board_length = len(self.board)
        if name == 'N':
            return vector_2D((vector.i - 1) % board_length, vector.j, 'N')
        elif name == 'NE':
            return vector_2D((vector.i - 1) % board_length, (vector.j + 1) % board_length, 'NE')
        elif name == 'E':
            return vector_2D(vector.i, (vector.j + 1) % board_length, 'E')
        elif name == 'SE':
            return vector_2D((vector.i + 1) % board_length, (vector.j + 1) % board_length, 'SE')
        elif name == 'S':
            return vector_2D((vector.i + 1) % board_length, vector.j, 'S')
        elif name == 'SW':
            return vector_2D((vector.i + 1) % board_length, (vector.j - 1) % board_length, 'SW')
        elif name == 'W':
            return vector_2D(vector.i, (vector.j - 1) % board_length, 'W')
        elif name == 'NW':
            return vector_2D((vector.i - 1) % board_length, (vector.j - 1) % board_length, 'NW')

    def get_neighbors(self, i, j):
        board_length = len(self.board)
        N = vector_2D((i - 1) % board_length, j, 'N')
        NE = vector_2D((i - 1) % board_length, (j + 1) % board_length, 'NE')
        E = vector_2D(i, (j + 1) % board_length, 'E')
        SE = vector_2D((i + 1) % board_length, (j + 1) % board_length, 'SE')
        S = vector_2D((i + 1) % board_length, j, 'S')
        SW = vector_2D((i + 1) % board_length, (j - 1) % board_length, 'SW')
        W = vector_2D(i, (j - 1) % board_length, 'W')
        NW = vector_2D((i - 1) % board_length, (j - 1) % board_length, 'NW')
        neighbors = [N, NE, E, SE, S, SW, W, NW]
        return neighbors


MY_CHARACTER = ''  # letter that represents maximizing player aka me
OPPONENT = ''  # letter that represents minimizing player


def parameterized_minimax(current_state, max_ply=2, use_alpha_beta=True, use_basic_static_eval=True):
   
    DATA = {'CURRENT_STATE_STATIC_VAL': -1000.0,
            'N_STATES_EXPANDED': 0,
            'N_STATIC_EVALS': 0,
            'N_CUTOFFS': 0}

    list_of_actions = get_list_of_actions(current_state)

    global USE_CUSTOM_STATIC_EVAL_FUNCTION
    if use_basic_static_eval:
        USE_CUSTOM_STATIC_EVAL_FUNCTION = True
    else:
        USE_CUSTOM_STATIC_EVAL_FUNCTION = False

    # 
    if use_alpha_beta:
        alpha_beta_minimax_helper(current_state, max_ply, list_of_actions, DATA)
    else:
        general_minimax_helper(current_state, max_ply, list_of_actions, DATA)

    # Actually return all results...
    return DATA


def alpha_beta_minimax_helper(current_state, depth, list_of_actions, DATA):
    action_value_dict = {}
    alpha = -math.inf
    if list_of_actions:
        best_action = list_of_actions[0]
        DATA["N_STATES_EXPANDED"] += 1
        for action in list_of_actions:
            new_state = MY_TTS_State(current_state.board)
            new_state.board[action[0]][action[1]] = MY_CHARACTER

            # pass down alpha value?
            evaluation = alpha_beta_minimax(new_state, depth - 1, alpha, math.inf, False,
                                            get_list_of_actions(new_state), DATA)
            alpha = max(alpha, evaluation)
            action_value_dict[action] = evaluation
            if time.time() - START_TIME >= (0.80 * TIME_LIMIT):
                break
        # find best action among initial actions
        for action_key in action_value_dict:
            if action_value_dict[action_key] > action_value_dict[best_action]:
                best_action = action_key
    else:
        best_action = _find_next_vacancy(current_state.board)
        new_state = MY_TTS_State(current_state.board)
        new_state.board[best_action[0]][best_action[1]] = MY_CHARACTER
        action_value_dict[best_action] = new_state.static_eval()

    DATA["CURRENT_STATE_STATIC_VAL"] = action_value_dict[best_action]
    return best_action, DATA["CURRENT_STATE_STATIC_VAL"]


def alpha_beta_minimax(state, depth, alpha, beta, is_maximizing_player, list_of_actions, DATA):
    if depth <= 0:
        if time.time() - START_TIME >= (0.80 * TIME_LIMIT):
            return DATA["CURRENT_STATE_STATIC_VAL"]
        else:
            DATA['N_STATIC_EVALS'] += 1
            return state.static_eval()

    if is_maximizing_player:
        max_evaluation = -math.inf
        DATA["N_STATES_EXPANDED"] += 1
        for action in list_of_actions:
            new_state = MY_TTS_State(state.board, MY_CHARACTER)
            new_state.board[action[0]][action[1]] = MY_CHARACTER
            evaluation = alpha_beta_minimax(new_state, depth - 1, alpha, beta, False, get_list_of_actions(new_state),
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
        for action in list_of_actions:
            new_state = MY_TTS_State(state.board, OPPONENT)
            new_state.board[action[0]][action[1]] = OPPONENT
            evaluation = alpha_beta_minimax(new_state, depth - 1, alpha, beta, True, get_list_of_actions(new_state),
                                            DATA)
            min_evaluation = min(min_evaluation, evaluation)
            beta = min(beta, evaluation)
            if beta <= alpha:
                DATA["N_CUTOFFS"] += 1
                break
            if time.time() - START_TIME >= (0.80 * TIME_LIMIT):
                break
        return min_evaluation


def alpha_beta_minimax_caller(state, depth, list_of_actions, DATA):
    static_eval = state.static_eval()
    action = _find_next_vacancy(state.board)
    global START_TIME
    for i in range(depth + 1):
        if time.time() - START_TIME >= (0.80 * TIME_LIMIT):
            break
        else:
            action, static_eval = alpha_beta_minimax_helper(state, i, list_of_actions, DATA)
    return action, static_eval


def general_minimax_helper(current_state, depth, list_of_actions, DATA):
    action_value_dict = {}
    if list_of_actions:
        BEST_ACTION = list_of_actions[0]
        max_eval = -math.inf
        DATA["N_STATES_EXPANDED"] += 1
        for action in list_of_actions:
            new_state = MY_TTS_State(current_state.board)
            new_state.board[action[0]][action[1]] = MY_CHARACTER
            evaluation = general_minimax(new_state, depth - 1, False, get_list_of_actions(new_state), DATA)
            action_value_dict[action] = evaluation
            if action_value_dict[action] > max_eval:
                BEST_ACTION = action
                max_eval = action_value_dict[action]
            # action_value_dict[BEST_ACTION] = evaluation
            if time.time() - START_TIME >= (0.80 * TIME_LIMIT):
                break
    else:
        BEST_ACTION = _find_next_vacancy(current_state.board)

    DATA["CURRENT_STATE_STATIC_VAL"] = action_value_dict[BEST_ACTION]
    return BEST_ACTION, DATA["CURRENT_STATE_STATIC_VAL"]


def general_minimax(state, depth, is_maximizing_player, list_of_actions, DATA):
    if depth <= 0:
        if time.time() - START_TIME >= (0.80 * TIME_LIMIT):
            return DATA["CURRENT_STATE_STATIC_VAL"]
        else:
            DATA['N_STATIC_EVALS'] += 1
            return state.static_eval()

    if is_maximizing_player:
        max_evaluation = -math.inf
        DATA["N_STATES_EXPANDED"] += 1
        for action in list_of_actions:
            new_state = MY_TTS_State(state.board, MY_CHARACTER)
            new_state.board[action[0]][action[1]] = MY_CHARACTER
            evaluation = general_minimax(new_state, depth - 1, False, get_list_of_actions(new_state),
                                         DATA)

            max_evaluation = max(max_evaluation, evaluation)
            if time.time() - START_TIME >= (0.80 * TIME_LIMIT):
                break
        return max_evaluation
    else:  # is minimizing player's turn
        min_evaluation = math.inf
        DATA["N_STATES_EXPANDED"] += 1
        for action in list_of_actions:
            new_state = MY_TTS_State(state.board, OPPONENT)
            new_state.board[action[0]][action[1]] = OPPONENT
            evaluation = general_minimax(new_state, depth - 1, True, get_list_of_actions(new_state),
                                         DATA)
            min_evaluation = min(min_evaluation, evaluation)
            if time.time() - START_TIME >= (0.80 * TIME_LIMIT):
                break
        return min_evaluation


def general_minimax_caller(state, depth, list_of_actions, DATA):
    static_eval = state.static_eval()
    action = _find_next_vacancy(state.board)
    global START_TIME
    for i in range(depth + 1):
        if time.time() - START_TIME >= (0.80 * TIME_LIMIT):
            break
        else:
            action, static_eval = general_minimax_helper(state, i, list_of_actions, DATA)
    return action, static_eval


queue = Queue(10)
START_TIME = time.time()
TIME_LIMIT = 30.0


def take_turn(current_state, last_utterance, time_limit):
    # Compute the new state for a move.
    # Start by copying the current state.
    new_state = MY_TTS_State(current_state.board)
    # Fix up whose turn it will be.
    who = current_state.whose_turn
    new_who = 'B'
    if who == 'B':
        new_who = 'W'
    new_state.whose_turn = new_who

    global TIME_LIMIT
    TIME_LIMIT = time_limit

    DATA = {'CURRENT_STATE_STATIC_VAL': -1000.0,
            'N_STATES_EXPANDED': 0,
            'N_STATIC_EVALS': 0,
            'N_CUTOFFS': 0}

    # location, static_eval = iterative_deepening(current_state, 2, get_list_of_actions(current_state), DATA,
    # time_limit)

    # location, static_eval = alpha_beta_minimax_helper(current_state, 2, get_list_of_actions(current_state), DATA)
    global START_TIME
    START_TIME = time.time()
    # location, static_eval = general_minimax_caller(new_state, 2, get_list_of_actions(new_state), DATA)
    location, static_eval = alpha_beta_minimax_caller(new_state, 3, get_list_of_actions(new_state), DATA)
    # print("static eval: " + str(DATA["CURRENT_STATE_STATIC_VAL"]))
    # print("states expanded: " + str(DATA["N_STATES_EXPANDED"]))
    # print("number of static evals: " + str(DATA["N_STATIC_EVALS"]))
    # print("N_CUTOFFS: " + str(DATA["N_CUTOFFS"]))
    move = location
    new_state.board[location[0]][location[1]] = MY_CHARACTER

    # Make up a new remark
    global retort_index
    new_utterance = list_of_retorts[retort_index % len(list_of_retorts)]
    retort_index += 1

    return [[move, new_state], new_utterance]


retort_index = 0
list_of_retorts = [   
                   "It's over, Anakin, I have the high ground",
                   "I don't mean to be rude or anything, but AI agents. . . ",
                   "just yell at me",
                   "Shout out to the people that helped me and worked with me, you know who you are.",
                   "I can't wait",
                   "Sorry I can't hear you, I'm listening to Nights Beat Switch",
                   "Be sure to listen to Frank Ocean",
                   "Well this is almost my twentieth line, so if the game is still going I guesss I'm doing okay",
                   "Does anyone want to get together and review?",
                   "Omae wa mou shindeiru", 
                   "Can someone teach me how to cook.",
                   "This is goodbye, old friend."]


def _find_next_vacancy(b):
    for i in range(len(b)):
        for j in range(len(b[0])):
            if b[i][j] == ' ':
                return i, j
    return False


def get_list_of_actions(state):
    list_of_actions = []
    board = state.board
    board_size = len(board)
    for i in range(board_size):
        for j in range(len(board[0])):
            if board[i][j] == ' ':
                list_of_actions.append((i, j))
    return list_of_actions


def moniker():
    return "Skipper"  # Return your agent's short nickname here.


def who_am_i():
    return """My name is Skipper, created by Andrew Garwood. """


K = 0
OPPONENT_MONIKER = "toast"


def get_ready(initial_state, k, who_i_play, opponent_moniker):
    # do any prep, like eval pre-calculation, here.
    # k is number of things in a row needed to win

    """Preparation? I hardly know her"""
    global K
    K = k
    global MY_CHARACTER
    MY_CHARACTER = who_i_play
    global OPPONENT_MONIKER
    OPPONENT_MONIKER = opponent_moniker
    global OPPONENT
    if MY_CHARACTER == 'W':
        OPPONENT = "B"
    else:
        OPPONENT = "W"

    return "OK"
