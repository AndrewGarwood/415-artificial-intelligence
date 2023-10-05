'''
Agent done by Chi Hyun Song
'''
from TTS_State import TTS_State
import time

initial_state = None
kRow = None
who_i_play = ' '
player2Nickname = None
BOARD_ROW = None
BOARD_COLUMN = None
int_utter = 0
new_utterance = []
spots = 0
turns = 0
called = 0

USE_CUSTOM_STATIC_EVAL_FUNCTION = False


class MY_TTS_State(TTS_State):
    def static_eval(self):
        if USE_CUSTOM_STATIC_EVAL_FUNCTION:
            return self.custom_static_eval()
        else:
            return self.basic_static_eval()

    def basic_static_eval(self):
        TWF = 0
        TBF = 0
        for i in range(len(self.board)):
            for j in range(len(self.board[0])):
                if self.board[i][j] == 'W':
                    for k in range(-1, 2):
                        for q in range(-1, 2):
                            if k == 0 and q == 0:
                                continue
                            a = k + i
                            b = q + j
                            if a >= len(self.board):
                                a = 0
                            if b >= len(self.board[0]):
                                b = 0
                            if self.board[a][b] == ' ':
                                TWF += 1
                if self.board[i][j] == 'B':
                    for k in range(-1, 2):
                        for q in range(-1, 2):
                            if k == 0 and q == 0:
                                continue
                            a = k + i
                            b = q + j
                            if a >= len(self.board):
                                a = 0
                            if b >= len(self.board[0]):
                                b = 0
                            if self.board[a][b] == ' ':
                                TBF += 1
        return TWF - TBF

    def custom_static_eval(self):
        score = 0
        for i in range(len(self.board)):
            inARowW = 0
            inARowB = 0
            for j in range(-len(self.board[0]) + 1, len(self.board[0])):
                if self.board[i][j] == 'W':
                    score += -2 ** inARowB + 1
                    inARowB = 0
                    inARowW += 1
                if self.board[i][j] == 'B':
                    score += 2 ** inARowW - 1
                    inARowW = 0
                    inARowB += 1
                if self.board[i][j] == '-':
                    score += -2 ** inARowB + 1
                    score += 2 ** inARowW - 1
                    inARowW = 0
                    inARowB = 0
                if j == len(self.board[0]) - 1:
                    score += -2 ** inARowB + 1
                    score += 2 ** inARowW - 1
        for j in range(len(self.board[0])):
            inAColumnW = 0
            inAColumnB = 0
            for i in range(-len(self.board) + 1, len(self.board)):
                if self.board[i][j] == 'W':
                    score += -2 ** inAColumnB + 1
                    inAColumnB = 0
                    inAColumnW += 1
                if self.board[i][j] == 'B':
                    score += 2 ** inAColumnW - 1
                    inAColumnW = 0
                    inAColumnB += 1
                if self.board[i][j] == '-':
                    score += -2 ** inAColumnB + 1
                    score += 2 ** inAColumnW - 1
                    inAColumnW = 0
                    inAColumnB = 0
                if i == len(self.board) - 1:
                    score += -2 ** inAColumnB + 1
                    score += 2 ** inAColumnW - 1
        return score


# The following is a skeleton for the function called parameterized_minimax,
# which should be a top-level function in each agent file.
# A tester or an autograder may do something like
# import ABC_TTS_agent as player, call get_ready(),
# and then it will be able to call tryout using something like this:
# results = player.parameterized_minimax(**kwargs)

def parameterized_minimax(
        current_state=None,
        max_ply=2,
        use_alpha_beta=False,
        use_basic_static_eval=True):
    alpha = -10000000000.0
    beta = 10000000000.0

    DATA = _parameterized_minimax(current_state, max_ply, use_alpha_beta,
                                  use_basic_static_eval, alpha, beta)

    return DATA[0]


def _parameterized_minimax(current_state, max_ply, use_alpha_beta,
                           use_basic_static_eval, alpha, beta):
    board = current_state.board
    whose_turn = current_state.whose_turn

    DATA = []
    META_DATA = {}
    if (whose_turn == 'W'):
        META_DATA['CURRENT_STATE_STATIC_VAL'] = -10000000000.0
    else:
        META_DATA['CURRENT_STATE_STATIC_VAL'] = 10000000000.0
    META_DATA['N_STATES_EXPANDED'] = 0
    META_DATA['N_STATIC_EVALS'] = 0
    META_DATA['N_CUTOFFS'] = 0
    LOC_DATA = []
    DATA.insert(0, META_DATA)
    DATA.insert(1, LOC_DATA)

    if (max_ply == 0 or not _find_next_vacancy(board)):
        current_state.__class__ = MY_TTS_State
        if (use_basic_static_eval):
            DATA[0]['CURRENT_STATE_STATIC_VAL'] = current_state.basic_static_eval()
        else:
            DATA[0]['CURRENT_STATE_STATIC_VAL'] = current_state.custom_static_eval()
        DATA[0]['N_STATIC_EVALS'] += 1
        return DATA

    if whose_turn == 'W':
        for i in range(len(board)):
            for j in range(len(board[0])):
                if board[i][j] == ' ':
                    new_state = current_state.copy()
                    new_state.whose_turn = 'B'
                    new_state.board[i][j] = 'W'
                    NEW_DATA = _parameterized_minimax(new_state, max_ply - 1, use_alpha_beta,
                                                      use_basic_static_eval, alpha, beta)
                    DATA[0]['N_STATES_EXPANDED'] += NEW_DATA[0]['N_STATES_EXPANDED']
                    DATA[0]['N_STATIC_EVALS'] += NEW_DATA[0]['N_STATIC_EVALS']
                    DATA[0]['N_CUTOFFS'] += NEW_DATA[0]['N_CUTOFFS']
                    if (DATA[0]['CURRENT_STATE_STATIC_VAL'] <
                            NEW_DATA[0]['CURRENT_STATE_STATIC_VAL']):
                        DATA[1].clear()
                        DATA[1].insert(0, i)
                        DATA[1].insert(1, j)
                        DATA[0]['CURRENT_STATE_STATIC_VAL'] = NEW_DATA[0]['CURRENT_STATE_STATIC_VAL']
                    alpha = max(alpha, DATA[0]['CURRENT_STATE_STATIC_VAL'])
                    if (alpha >= beta and use_alpha_beta):
                        DATA[0]['N_CUTOFFS'] += 1
                        DATA[0]['N_STATES_EXPANDED'] += 1
                        return DATA
        DATA[0]['N_STATES_EXPANDED'] += 1
        return DATA
    else:
        for i in range(len(board)):
            for j in range(len(board[0])):
                if board[i][j] == ' ':
                    new_state = current_state.copy()
                    new_state.whose_turn = 'W'
                    new_state.board[i][j] = 'B'
                    NEW_DATA = _parameterized_minimax(new_state, max_ply - 1, use_alpha_beta,
                                                      use_basic_static_eval, alpha, beta)
                    DATA[0]['N_STATES_EXPANDED'] += NEW_DATA[0]['N_STATES_EXPANDED']
                    DATA[0]['N_STATIC_EVALS'] += NEW_DATA[0]['N_STATIC_EVALS']
                    DATA[0]['N_CUTOFFS'] += NEW_DATA[0]['N_CUTOFFS']
                    if (DATA[0]['CURRENT_STATE_STATIC_VAL'] >
                            NEW_DATA[0]['CURRENT_STATE_STATIC_VAL']):
                        DATA[1].clear()
                        DATA[1].insert(0, i)
                        DATA[1].insert(1, j)
                        DATA[0]['CURRENT_STATE_STATIC_VAL'] = NEW_DATA[0]['CURRENT_STATE_STATIC_VAL']
                    beta = min(beta, DATA[0]['CURRENT_STATE_STATIC_VAL'])
                    if (alpha >= beta and use_alpha_beta):
                        DATA[0]['N_CUTOFFS'] += 1
                        DATA[0]['N_STATES_EXPANDED'] += 1
                        return DATA
        DATA[0]['N_STATES_EXPANDED'] += 1
        return DATA


def take_turn(current_state, last_utterance, time_limit):
    global int_utter
    global turns
    global spots
    global called

    if called == 2:
        turns += 1
    if called == 1:
        turns += 2

    # Compute the new state for a move.
    # Start by copying the current state.
    new_state = MY_TTS_State(current_state.board)
    # Fix up whose turn it will be.
    who = current_state.whose_turn
    new_who = 'B'
    if who == 'B': new_who = 'W'
    new_state.whose_turn = new_who

    # Place a new tile

    left_time = time_limit
    expected_time = 0

    i = 1
    while (expected_time < left_time):
        duration_time = time.time()
        DATA = _parameterized_minimax(new_state, i, True,
                                      False, -10000000000.0, 10000000000.0)
        end_time = time.time() - duration_time
        expected_time = end_time * (BOARD_ROW * BOARD_COLUMN - 2 * i)
        left_time -= end_time
        i += 1
        if i > spots - turns:
            break

    location = DATA[1]
    if location == False: return [[False, current_state], "I don't have any moves!"]
    new_state.board[location[0]][location[1]] = who

    # Construct a representation of the move that goes from the
    # currentState to the newState.
    move = location

    # Make up a new remark

    if (int_utter <= 9):
        new_utterance_now = new_utterance[int_utter]
    else:
        int_utter = 0
        new_utterance_now = new_utterance[int_utter]
    int_utter += 1

    return [[move, new_state], new_utterance_now]


def _find_next_vacancy(b):
    for i in range(len(b)):
        for j in range(len(b[0])):
            if b[i][j] == ' ': return (i, j)
    return False


def moniker():
    return "thinker"


def who_am_i():
    return """My name is thinker, created by Chi Hyun Song.
Greatest thinker of all time"""


def get_ready(initial_state2, k2, who_i_play2, player2Nickname2):
    global initial_state
    global kRow
    global who_i_play
    global player2Nickname
    global BOARD_ROW
    global BOARD_COLUMN
    global new_utterance
    global spots
    global called

    called += 1

    new_utterance.append("Thinking...")
    new_utterance.append("Considering...")
    new_utterance.append("Expecting...")
    new_utterance.append("Guessing...")
    new_utterance.append("Realizing...")
    new_utterance.append("Understanding...")
    new_utterance.append("Concluding...")
    new_utterance.append("Foreseeing...")
    new_utterance.append("Regarding...")
    new_utterance.append("Visualizing...")

    initial_state = initial_state2
    for i in range(len(initial_state.board)):
        for j in range(len(initial_state.board[0])):
            if initial_state.board[i][j] == ' ' and called == 1:
                spots += 1
    kRow = k2
    who_i_play = who_i_play2
    player2Nickname = player2Nickname2
    BOARD_ROW = len(initial_state2.board[0])
    BOARD_COLUMN = len(initial_state2.board)
    print(called)
    return "OK"
