import BC_state_etc as BC
from random import randint
import time

player2_name = ''
zobristnum = [[0] * 16 for i in range(64)]
hashTable = {}
new_utterance = []
int_utter = 0
REAL_DATA = {}

# randomly assigns the value of each block in the chess board
for i in range(64):
    for j in range(16):
        zobristnum[i][j] = randint(0, 4294967296)

    # main algorithm


def parameterized_minimax(currentState, alphaBeta=False, ply=3,
                          useBasicStaticEval=True, useZobristHashing=False):
    alpha = -10000000000.0
    beta = 10000000000.0
    DATA = _parameterized_minimax(currentState, alphaBeta, ply, useBasicStaticEval, useZobristHashing, alpha, beta)

    REAL_DATA[0]['CURRENT_STATE_STATIC_VAL'] = DATA[0]['CURRENT_STATE_STATIC_VAL']
    REAL_DATA[0]['N_STATES_EXPANDED'] = DATA[0]['N_STATES_EXPANDED']
    REAL_DATA[0]['N_STATIC_EVALS'] = DATA[0]['N_STATIC_EVALS']
    REAL_DATA[0]['N_CUTOFFS'] = DATA[0]['N_CUTOFFS']
    REAL_DATA[0]['game_put_count'] += DATA[0]['move_put_count']
    REAL_DATA[0]['game_get_success_count'] += DATA[0]['move_get_success_count']
    REAL_DATA[0]['game_get_failure_count'] += DATA[0]['move_get_failure_count']
    REAL_DATA[0]['game_collision_count'] += DATA[0]['move_collision_count']
    REAL_DATA[0]['game_static_evals_saved_count'] += DATA[0]['move_static_evals_saved_count']
    REAL_DATA[0]['move_put_count'] = DATA[0]['move_put_count']
    REAL_DATA[0]['move_get_success_count'] = DATA[0]['move_get_success_count']
    REAL_DATA[0]['move_get_failure_count'] = DATA[0]['move_get_failure_count']
    REAL_DATA[0]['move_collision_count'] = DATA[0]['move_collision_count']
    REAL_DATA[0]['move_static_evals_saved_count'] = DATA[0]['move_static_evals_saved_count']
    return REAL_DATA[0]


# helper function of main algorithm
def _parameterized_minimax(currentState, alphaBeta, ply, useBasicStaticEval,
                           useZobristHashing, alpha, beta):
    board = currentState.board
    whose_turn = currentState.whose_move
    DATA = _readyData(whose_turn)

    hashKey = _zhash(board)

    # checks if we have already visited this before
    if hashKey in hashTable and useZobristHashing:
        hashValue = hashTable.get(hashKey)
        if hashValue[3].__eq__(currentState):
            DATA[0]['move_collision_count'] += 1
        if hashValue[0] == whose_turn and hashValue[1] >= ply:
            DATA[0]['move_get_success_count'] += 1
            DATA[0]['CURRENT_STATE_STATIC_VAL'] = hashValue[2]
            return DATA

    DATA[0]['move_get_failure_count'] += 1

    # king is dead or we are at the end of iteration
    if ply == 0 or _kingDead(board):
        if useBasicStaticEval:
            DATA[0]['CURRENT_STATE_STATIC_VAL'] = basicStaticEval(currentState)
        else:
            DATA[0]['CURRENT_STATE_STATIC_VAL'] = staticEval(currentState)
        DATA[0]['N_STATIC_EVALS'] += 1
        return DATA

    # main part of main algorithm
    # looks at all possible place that all possible piece can move to
    for i in range(len(board)):
        for j in range(len(board[0])):
            if board[i][j] != 0 and BC.who(board[i][j]) == whose_turn and _freezerNotNear([i, j], whose_turn, board):
                for k in range(1, len(board)):
                    if _validateMove(board, [i, j], _upRight([i - k, j]), board[i][j], whose_turn):
                        AB = _oneStep(board, alphaBeta, ply, useBasicStaticEval, useZobristHashing, alpha, beta,
                                      whose_turn, [i, j], _upRight([i - k, j]), DATA)
                        alpha = AB[0]
                        beta = AB[1]
                        if AB[2] == 13:
                            _zhashComputation(DATA, whose_turn, ply)
                            return DATA
                for k in range(1, len(board)):
                    if _validateMove(board, [i, j], _upRight([i, j - k]), board[i][j], whose_turn):
                        AB = _oneStep(board, alphaBeta, ply, useBasicStaticEval, useZobristHashing, alpha, beta,
                                      whose_turn, [i, j], _upRight([i, j - k]), DATA)
                        alpha = AB[0]
                        beta = AB[1]
                        if AB[2] == 13:
                            _zhashComputation(DATA, whose_turn, ply)
                            return DATA
                if board[i][j] != 2 + whose_turn:
                    for k in range(0, 8 - abs(i - j)):
                        min_val = min(i, j)
                        if i + k - min_val != i and j + k - min_val != j:
                            if _validateMove(board, [i, j], [i + k - min_val, j + k - min_val], board[i][j],
                                             whose_turn):
                                AB = _oneStep(board, alphaBeta, ply, useBasicStaticEval, useZobristHashing, alpha, beta,
                                              whose_turn, [i, j], [i + k - min_val, j + k - min_val], DATA)
                                alpha = AB[0]
                                beta = AB[1]
                                if AB[2] == 13:
                                    _zhashComputation(DATA, whose_turn, ply)
                                    return DATA
                    for k in range(0, 8 - abs(7 - (i + j))):
                        if i + j <= 7:
                            if i + j - k != i and k != j:
                                if _validateMove(board, [i, j], [i + j - k, k], board[i][j], whose_turn):
                                    AB = _oneStep(board, alphaBeta, ply, useBasicStaticEval, useZobristHashing, alpha,
                                                  beta, whose_turn, [i, j], [i + j - k, k], DATA)
                                    alpha = AB[0]
                                    beta = AB[1]
                                    if AB[2] == 13:
                                        _zhashComputation(DATA, whose_turn, ply)
                                        return DATA
                        else:
                            if 7 - k != i and j - (7 - i) + k != j:
                                if _validateMove(board, [i, j], [7 - k, j - (7 - i) + k], board[i][j], whose_turn):
                                    AB = _oneStep(board, alphaBeta, ply, useBasicStaticEval, useZobristHashing, alpha,
                                                  beta, whose_turn, [i, j], [7 - k, j - (7 - i) + k], DATA)
                                    alpha = AB[0]
                                    beta = AB[1]
                                    if AB[2] == 13:
                                        _zhashComputation(DATA, whose_turn, ply)
                                        return DATA
    DATA[0]['N_STATES_EXPANDED'] += 1
    _zhashComputation(DATA, whose_turn, ply)
    return DATA


# do the move and show it
def makeMove(currentState, currentRemark, timelimit=10):
    global int_utter

    alpha = -10000000000.0
    beta = 10000000000.0

    left_time = timelimit
    expected_time = 0
    i = 1

    while (expected_time < left_time):
        duration_time = time.time()
        DATA = _parameterized_minimax(currentState, True, i, False, True, alpha, beta)
        end_time = time.time() - duration_time
        expected_time = end_time * 131072
        left_time -= end_time
        i += 1

    newState = BC.BC_state(DATA[3][0], 1 - currentState.whose_move)

    if (int_utter <= 9):
        new_utterance_now = new_utterance[int_utter]
    else:
        int_utter = 0
        new_utterance_now = new_utterance[int_utter]
    int_utter += 1

    move = ((DATA[1][0], DATA[1][1]), (DATA[2][0], DATA[2][1]))

    print('N_STATES_EXPANDED: ' + str(DATA[0]['N_STATES_EXPANDED']))
    print('N_STATIC_EVALS: ' + str(DATA[0]['N_STATIC_EVALS']))
    print('N_CUTOFFS: ' + str(DATA[0]['N_CUTOFFS']))
    print('move_put_count: ' + str(DATA[0]['move_put_count']))
    print('move_get_success_count: ' + str(DATA[0]['move_get_success_count']))
    print('move_get_failure_count: ' + str(DATA[0]['move_get_failure_count']))
    print('move_collision_count: ' + str(DATA[0]['move_collision_count']))
    print('move_static_evals_saved_count: ' + str(DATA[0]['move_static_evals_saved_count']))

    return [[move, newState], new_utterance_now]


def nickname():
    return "Gary"


def introduce():
    return "I'm Gary 23 from Vault 108"


def prepare(player2Nickname):
    global new_utterance
    global player2_name
    global REAL_DATA

    REAL_DATA = _readyData(1)

    player2_name = player2Nickname

    new_utterance.append("Gray!")
    new_utterance.append("Gray?")
    new_utterance.append("Haha! Gary")
    new_utterance.append("Gary...")
    new_utterance.append("Ah... Gary")
    new_utterance.append("Uh... Gary?")
    new_utterance.append("Gary!?!?")
    new_utterance.append("Um... Gary!")
    new_utterance.append("GARY~~~")
    new_utterance.append("ok Gary")


def basicStaticEval(state):
    board = state.board
    score = 0
    for i in range(len(board)):
        for j in range(len(board[0])):
            if board[i][j] == 2:
                score += -1
            if board[i][j] == 3:
                score += 1
            if board[i][j] in (4, 6, 8, 10, 14):
                score += -2
            if board[i][j] in (5, 7, 9, 11, 15):
                score += 2
            if board[i][j] == 12:
                score += -100
            if board[i][j] == 13:
                score += 100
    return score


def staticEval(state):
    board = state.board
    score = 0

    for i in range(len(board)):
        for j in range(len(board[0])):
            if board[i][j] != 0 and BC.who(board[i][j]) == 1:
                for k in range(-1, 2):
                    for q in range(-1, 2):
                        if _possibleSlot([i + k, j + q]):
                            if board[i + k][j + q] == 0 or (
                                    board[i + k][j + q] != 0 and BC.who(board[i + k][j + q]) == 0):
                                score += 1
            if board[i][j] != 0 and BC.who(board[i][j]) == 0:
                for k in range(-1, 2):
                    for q in range(-1, 2):
                        if _possibleSlot([i + k, j + q]):
                            if board[i + k][j + q] == 0 or (
                                    board[i + k][j + q] != 0 and BC.who(board[i + k][j + q]) == 1):
                                score -= 1
            if board[i][j] == 3:
                score += 40
            if board[i][j] in (5, 7, 9, 11, 15):
                score += 80
            if board[i][j] == 13:
                score += 1000
            if board[i][j] == 2:
                score -= 40
            if board[i][j] in (4, 6, 8, 10, 14):
                score -= 80
            if board[i][j] == 12:
                score -= 1000
    return score


# given a specific pattern on the board, give a specific number
def _zhash(board):
    global zobristnum
    val = 0
    for i in range(len(board)):
        for j in range(len(board[0])):
            board_key = i * 8 + j
            piece_key = board[i][j]
            if board[i][j] != 0:
                val ^= zobristnum[board_key][piece_key]
    return val


# ready the data to be filled in
def _readyData(whose_turn):
    DATA = []
    META_DATA = {}
    if (whose_turn == 1):
        META_DATA['CURRENT_STATE_STATIC_VAL'] = -10000000000.0
    else:
        META_DATA['CURRENT_STATE_STATIC_VAL'] = 10000000000.0
    META_DATA['N_STATES_EXPANDED'] = 0
    META_DATA['N_STATIC_EVALS'] = 0
    META_DATA['N_CUTOFFS'] = 0
    META_DATA['game_put_count'] = 0
    META_DATA['game_get_success_count'] = 0
    META_DATA['game_get_failure_count'] = 0
    META_DATA['game_collision_count'] = 0
    META_DATA['game_static_evals_saved_count'] = 0
    META_DATA['move_put_count'] = 0
    META_DATA['move_get_success_count'] = 0
    META_DATA['move_get_failure_count'] = 0
    META_DATA['move_collision_count'] = 0
    META_DATA['move_static_evals_saved_count'] = 0
    LOC_DATA = []
    MOVE_DATA = []
    BOARD_DATA = []
    DATA.insert(0, META_DATA)
    DATA.insert(1, LOC_DATA)
    DATA.insert(2, MOVE_DATA)
    DATA.insert(3, BOARD_DATA)
    return DATA


# if the list index goes negative upRight it so that it is no longer negative
def _upRight(move):
    if move[0] < 0:
        move[0] = 8 + move[0]
    if move[1] < 0:
        move[1] = 8 + move[1]
    return move


# Is the move valid? If not, return False
def _validateMove(board, starting, move, piece, whose_turn):
    starting_i = starting[0]
    starting_j = starting[1]
    direction = _findDirection([starting_i, starting_j], move)
    starting_i += direction[0]
    starting_j += direction[1]
    if piece in (6, 7):
        if board[move[0]][move[1]] != 0:
            return False
        while starting_i != move[0] or starting_j != move[1]:
            if board[starting_i][starting_j] != 0:
                if BC.who(board[starting_i][starting_j]) == whose_turn:
                    return False
                if BC.who(board[starting_i][starting_j]) != whose_turn:
                    if starting_i + direction[0] != move[0] or starting_j + direction[1] != move[1]:
                        return False
            starting_i += direction[0]
            starting_j += direction[1]
    elif piece == 12 + whose_turn:
        if move[0] != starting_i or move[1] != starting_j:
            return False
    else:
        if board[move[0]][move[1]] != 0:
            return False
        while starting_i != move[0] or starting_j != move[1]:
            if board[starting_i][starting_j] != 0:
                return False
            starting_i += direction[0]
            starting_j += direction[1]
    return True


# Find the direction of movement of the piece in one increment
def _findDirection(starting, move):
    i = 0
    j = 0
    direction = [move[0] - starting[0], move[1] - starting[1]]
    if direction[0] < 0:
        i = -1
    elif direction[0] > 0:
        i = 1
    else:
        i = 0
    if direction[1] < 0:
        j = -1
    elif direction[1] > 0:
        j = 1
    else:
        j = 0
    return [i, j]


# is king dead?
def _kingDead(b):
    white_king_alive = False
    black_king_alive = False
    for i in range(len(b)):
        for j in range(len(b[0])):
            if b[i][j] == 12:
                black_king_alive = True
            if b[i][j] == 13:
                white_king_alive = True
    if white_king_alive and black_king_alive:
        return False
    return True


# Is freezer near so that this piece may not be able to move?
def _freezerNotNear(starting, whose_turn, board):
    for i in range(-1, 2):
        for j in range(-1, 2):
            if _possibleSlot([starting[0] + i, starting[1] + j]):
                if board[starting[0] + i][starting[1] + j] == 15 - whose_turn:
                    return False
    return True


# Is this spot a possible spot on the board?
def _possibleSlot(starting):
    if starting[0] < 0 or starting[0] > 7 or starting[1] < 0 or starting[1] > 7:
        return False
    return True


# Where is king?
def _kingLocation(board, whose_turn):
    for i in range(len(board)):
        for j in range(len(board[0])):
            if board[i][j] == 12 + whose_turn:
                return [i, j]
    return None


# Given this valid turn, how would it change the board?
def _oneTurn(board, piece, starting, move, whose_turn):
    if piece == 2 + whose_turn:
        if _possibleSlot([move[0] - 2, move[1]]):
            if BC.who(board[move[0] - 1][move[1]]) != whose_turn and BC.who(
                    board[move[0] - 2][move[1]]) == whose_turn and board[move[0] - 2][move[1]] != 0:
                board[move[0] - 1][move[1]] = 0
        if _possibleSlot([move[0] + 2, move[1]]):
            if BC.who(board[move[0] + 1][move[1]]) != whose_turn and BC.who(
                    board[move[0] + 2][move[1]]) == whose_turn and board[move[0] + 2][move[1]] != 0:
                board[move[0] + 1][move[1]] = 0
        if _possibleSlot([move[0], move[1] - 2]):
            if BC.who(board[move[0]][move[1] - 1]) != whose_turn and BC.who(
                    board[move[0]][move[1] - 2]) == whose_turn and board[move[0]][move[1] - 2] != 0:
                board[move[0]][move[1] - 1] = 0
        if _possibleSlot([move[0], move[1] + 2]):
            if BC.who(board[move[0]][move[1] + 1]) != whose_turn and BC.who(
                    board[move[0]][move[1] + 2]) == whose_turn and board[move[0]][move[1] + 2] != 0:
                board[move[0]][move[1] + 1] = 0
        board[starting[0]][starting[1]] = 0
        board[move[0]][move[1]] = 2 + whose_turn
    if piece == 4 + whose_turn:
        kingLoc = _kingLocation(board, whose_turn)
        difference = [move[0] - kingLoc[0], move[1] - kingLoc[1]]
        if difference[0] != 0 and difference[1] != 0:
            if BC.who(board[move[0]][kingLoc[1]]) != whose_turn:
                board[move[0]][kingLoc[1]] = 0
            if BC.who(board[kingLoc[0]][move[1]]) != whose_turn:
                board[kingLoc[0]][move[1]] = 0
        board[starting[0]][starting[1]] = 0
        board[move[0]][move[1]] = 4 + whose_turn
    if piece == 6 + whose_turn:
        direction = _findDirection(starting, move)
        board[starting[0]][starting[1]] = 0
        board[move[0] - direction[0]][move[1] - direction[1]] = 0
        board[move[0]][move[1]] = 6 + whose_turn
    # if piece == 8 + whose_turn:  
    if piece == 10 + whose_turn:
        board[starting[0]][starting[1]] = 0
        direction = _findDirection(starting, move)
        if _possibleSlot([starting[0] - direction[0], starting[1] - direction[1]]):
            if BC.who(board[starting[0] - direction[0]][starting[1] - direction[1]]) != whose_turn and BC.who(
                    board[starting[0] - direction[0]][starting[1] - direction[1]]) != 0:
                board[starting[0] - direction[0]][starting[1] - direction[1]] = 0
        board[move[0]][move[1]] = 10 + whose_turn
    if piece == 12 + whose_turn:
        board[starting[0]][starting[1]] = 0
        board[move[0]][move[1]] = 12 + whose_turn
    if piece == 14 + whose_turn:
        board[starting[0]][starting[1]] = 0
        board[move[0]][move[1]] = 14 + whose_turn


# One move of all possible moves available
def _oneStep(board, alphaBeta, ply, useBasicStaticEval, useZobristHashing, alpha, beta, whose_turn, starting, move,
             DATA):
    newState_board = [r[:] for r in board]
    _oneTurn(newState_board, newState_board[starting[0]][starting[1]], starting, move, whose_turn)
    newState = BC.BC_state(newState_board, 1 - whose_turn)
    NEW_DATA = _parameterized_minimax(newState, alphaBeta, ply - 1, useBasicStaticEval, useZobristHashing, alpha, beta)
    DATA[0]['N_STATES_EXPANDED'] += NEW_DATA[0]['N_STATES_EXPANDED']
    DATA[0]['N_STATIC_EVALS'] += NEW_DATA[0]['N_STATIC_EVALS']
    DATA[0]['N_CUTOFFS'] += NEW_DATA[0]['N_CUTOFFS']
    DATA[0]['move_put_count'] += NEW_DATA[0]['move_put_count']
    DATA[0]['move_get_success_count'] += NEW_DATA[0]['move_get_success_count']
    DATA[0]['move_get_failure_count'] += NEW_DATA[0]['move_get_failure_count']
    DATA[0]['move_collision_count'] += NEW_DATA[0]['move_collision_count']
    DATA[0]['move_static_evals_saved_count'] += NEW_DATA[0]['move_static_evals_saved_count']
    if whose_turn == 1:
        if (DATA[0]['CURRENT_STATE_STATIC_VAL'] < NEW_DATA[0]['CURRENT_STATE_STATIC_VAL']):
            DATA[1].clear()
            DATA[2].clear()
            DATA[3].clear()
            DATA[1].insert(0, starting[0])
            DATA[1].insert(1, starting[1])
            DATA[2].insert(0, move[0])
            DATA[2].insert(1, move[1])
            DATA[3].insert(0, newState_board)
            DATA[0]['CURRENT_STATE_STATIC_VAL'] = NEW_DATA[0]['CURRENT_STATE_STATIC_VAL']
        alpha = max(alpha, DATA[0]['CURRENT_STATE_STATIC_VAL'])
        if (alpha >= beta and alphaBeta):
            DATA[0]['N_CUTOFFS'] += 1
            DATA[0]['N_STATES_EXPANDED'] += 1
            return [alpha, beta, 13]
    else:
        if (DATA[0]['CURRENT_STATE_STATIC_VAL'] > NEW_DATA[0]['CURRENT_STATE_STATIC_VAL']):
            DATA[1].clear()
            DATA[2].clear()
            DATA[3].clear()
            DATA[1].insert(0, starting[0])
            DATA[1].insert(1, starting[1])
            DATA[2].insert(0, move[0])
            DATA[2].insert(1, move[1])
            DATA[3].insert(0, newState_board)
            DATA[0]['CURRENT_STATE_STATIC_VAL'] = NEW_DATA[0]['CURRENT_STATE_STATIC_VAL']
        beta = min(beta, DATA[0]['CURRENT_STATE_STATIC_VAL'])
        if (alpha >= beta and alphaBeta):
            DATA[0]['N_CUTOFFS'] += 1
            DATA[0]['N_STATES_EXPANDED'] += 1
            return [alpha, beta, 13]
    return [alpha, beta, 12]


# make the hashtable of the board
def _zhashComputation(DATA, whose_turn, ply):
    DATA[0]['move_put_count'] += 1
    DATA[0]['move_static_evals_saved_count'] += 1
    hashKey = _zhash(DATA[3][0])
    hashTable[hashKey] = [whose_turn, ply, DATA[0]['CURRENT_STATE_STATIC_VAL'], BC.BC_state(DATA[3][0], whose_turn)]
