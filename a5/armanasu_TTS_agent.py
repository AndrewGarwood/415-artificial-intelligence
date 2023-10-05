'''armanasu_TTS_agent.py
A Toro-Tile-Straight playing agent, emulating the character of
someone who always says what's on their mind. Implements alpha-beta pruning
and iterative deepening search.
'''

from TTS_State import TTS_State, BLACK, WHITE
import time

USE_CUSTOM_STATIC_EVAL_FUNCTION = True
ARMANASU_PLAYER_SIDE = None
ARMANASU_INITIAL_SIDE = None
k_val = None
armanasu_lines = []
armanasu_last_state = None
armanasu_fail_states = []
armanasu_win_states = []


class MY_TTS_State(TTS_State):
    def static_eval(self):
        if USE_CUSTOM_STATIC_EVAL_FUNCTION:
            return self.custom_static_eval()
        else:
            return self.basic_static_eval()

    def basic_static_eval(self):  # calculates how many 'freedoms' white has over
        twf = 0  # black, using the formula defined in the
        tbf = 0  # assignment spec.
        n = len(self.board)
        m = len(self.board[0])
        for i, row in enumerate(self.board):
            for j, tile in enumerate(row):  # iterate over every tile
                if self.board[i][j] is BLACK or self.board[i][j] is WHITE:
                    freedoms = 0
                    for ny in [-1, 0, 1]:
                        for nx in [-1, 0, 1]:  # iterate over its neighbors
                            if self.board[(i + nx) % n][(j + ny) % m] == ' ':
                                freedoms += 1
                    if self.board[i][j] is BLACK:
                        tbf += freedoms
                    else:
                        twf += freedoms
        return twf - tbf

    def custom_static_eval(self):
        global armanasu_lines, k_val, ARMANASU_PLAYER_SIDE
        tww = 0
        tbw = 0
        for line in armanasu_lines:
            (white, black) = valid_line(self, line)
            if white:
                count = 0
                for k, (i, j) in enumerate(line):
                    if self.board[i][j] == 'W':
                        count += 1
                    elif self.board[i][j] == 'B':
                        (l, m) = line[(k - 1) % len(line)]
                        (n, o) = line[(k + 1) % len(line)]
                        if self.board[l][m] == 'W' or self.board[n][o] == 'W':
                            count -= .5
                if count + 2 >= k_val and ARMANASU_PLAYER_SIDE == 'B':
                    count += 2
                tww += 10 ** count
                if side_win(self, line, WHITE):
                    tww += 20000000
            if black:
                count = 0
                for k, (i, j) in enumerate(line):
                    if self.board[i][j] == 'B':
                        count += 1
                    elif self.board[i][j] == 'W':
                        (l, m) = line[(k - 1) % len(line)]
                        (n, o) = line[(k + 1) % len(line)]
                        if self.board[l][m] == 'B' or self.board[n][o] == 'B':
                            count -= .5
                if count + 2 >= k_val and ARMANASU_PLAYER_SIDE == 'W':
                    count += 2
                tbw += 10 ** count
                if side_win(self, line, BLACK):
                    tww += -20000000
        if ARMANASU_PLAYER_SIDE == 'W':
            return tww - 2 * tbw
        else:
            return 2 * tww - tbw


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
    # All students, add code to replace these default
    # values with correct values from your agent (either here or below).
    DATA = {}
    DATA['CURRENT_STATE_STATIC_VAL'] = -1000.0
    DATA['N_STATES_EXPANDED'] = 0
    DATA['N_STATIC_EVALS'] = 0
    DATA['N_CUTOFFS'] = 0
    maximize = ARMANASU_PLAYER_SIDE == 'W'

    # STUDENTS: You may create the rest of the body of this function here.
    (DATA['CURRENT_STATE_STATIC_VAL'],
     DATA['N_STATES_EXPANDED'],
     DATA['N_STATIC_EVALS'],
     DATA['N_CUTOFFS'],
     timeout) = recursive_minimax(current_state, max_ply, use_alpha_beta,
                                  basic=use_basic_static_eval,
                                  maximize=maximize)  # use recursive function to do
    # the heavy lifting
    # Actually return all results...
    return (DATA)


def recursive_minimax(state, ply, ab, alpha=float('-inf'), beta=float('inf'),
                      basic=True, maximize=True, timed=False, init_time=None,
                      timer=None, use_fail_states=False, depth=0,
                      geq=True):  # recursive version of parameterized minimax
    # with extra arguments to handle recursion
    global armanasu_fail_states, armanasu_win_states, ARMANASU_PLAYER_SIDE

    if ply == 0:  # base case, we've gone as deep as we can.
        val = None
        if not basic:
            val = state.custom_static_eval()

        else:
            val = state.basic_static_eval()
        return (val, 0, 1, 0, False)

    # recursive case, we can go a level deeper
    child_vals = []  # initialize our search values for this level of the tree
    exp_layer = 0
    evals_layer = 0
    cutoffs_layer = 0
    cutoff = False

    for i, row in enumerate(state.board):
        for j, tile in enumerate(row):  # iterate over the board
            if state.board[i][j] == ' ' and not cutoff:  # only go deeper if it's a valid position
                new_state = MY_TTS_State(state.board)  # note that 'cutoff' will only ever be true
                if maximize:  # when alpha-beta pruning is enabled
                    new_state.board[i][j] = 'W'
                else:
                    new_state.board[i][j] = 'B'
                if use_fail_states:  # determine whether it's worth expanding or the player can't win
                    for ((k, l), win_depth) in armanasu_win_states:
                        if win_depth == depth:
                            if ARMANASU_PLAYER_SIDE == 'W':
                                if new_state.board[k][l] == ' ':
                                    test_state = new_state
                                    test_state.board[k][l] = 'W'
                                    if did_win(test_state, 'W'):
                                        return (test_state.static_eval(), 1, 0, 1, False)
                            else:
                                if new_state.board[k][l] == ' ':
                                    test_state = new_state
                                    test_state.board[k][l] = 'B'
                                    if did_win(test_state, 'B'):
                                        return (test_state.static_eval(), 1, 0, 1, False)
                    for ((k, l), fail_depth) in armanasu_fail_states:
                        if fail_depth == depth:
                            if ARMANASU_PLAYER_SIDE == 'W':
                                if new_state.board[k][l] == ' ':
                                    test_state = new_state
                                    test_state.board[k][l] = 'B'
                                    if did_win(test_state, 'B'):
                                        return (test_state.static_eval(), 1, 0, 1, False)
                            else:
                                if new_state.board[k][l] == ' ':
                                    test_state = new_state
                                    test_state.board[k][l] = 'W'
                                    if did_win(test_state, 'W'):
                                        return (test_state.static_eval(), 1, 0, 1, False)

                (val, exp, evals, cutoffs, timeout) = recursive_minimax(new_state, ply - 1, ab,
                                                                        alpha, beta, basic,
                                                                        not maximize, timed,
                                                                        # alternate maximization and minimization layers
                                                                        init_time, timer,
                                                                        use_fail_states, depth + 1, geq)

                child_vals.append(val)  # update search values
                exp_layer += exp
                evals_layer += evals
                cutoffs_layer += cutoffs

                if use_fail_states:  # add to the list fail states that must be prevented
                    if ARMANASU_PLAYER_SIDE == 'W':
                        if val < -10000000:
                            armanasu_fail_states.append(((i, j), depth))
                        elif val > 10000000:
                            armanasu_win_states.append(((i, j), depth))
                    else:
                        if val > 10000000:
                            armanasu_fail_states.append(((i, j), depth))
                        elif val < -10000000:
                            armanasu_win_states.append(((i, j), depth))
                if timed:
                    if time.time() - init_time + .4 > timer:  # ran out of time
                        child_vals.append(float('NaN'))  # make sure we always have some value to return
                        if maximize:
                            return (max(child_vals), exp_layer + 1, evals_layer, cutoffs_layer, True)
                        else:
                            return (min(child_vals), exp_layer + 1, evals_layer, cutoffs_layer, True)
                if ab:  # apply alpha-beta pruning
                    if maximize:
                        alpha = max(alpha, val)
                    else:
                        beta = min(beta, val)
                    if (alpha > beta):
                        cutoff = True
                        cutoffs_layer += 1
                    elif (geq and alpha == beta):
                        cutoff = True
                        cutoffs_layer += 1

    if maximize:
        child_vals.append(float('-inf'))  # make sure we always have some value to return
        return (max(child_vals), exp_layer + 1, evals_layer, cutoffs_layer, False)  # return best move for max
    else:
        child_vals.append(float('inf'))  # make sure we always have some value to return
        return (min(child_vals), exp_layer + 1, evals_layer, cutoffs_layer, False)  # return best move for min


def take_turn(current_state, last_utterance, time_limit):
    global ARMANASU_PLAYER_SIDE, armanasu_last_state, armanasu_lines
    global armanasu_fail_states, armanasu_win_states
    t0 = time.time()
    # Compute the new state for a move.
    # Start by copying the current state.
    new_state = MY_TTS_State(current_state.board)
    # Fix up whose turn it will be.
    who = current_state.whose_turn
    ARMANASU_PLAYER_SIDE = who
    new_who = 'B'
    if who == 'B': new_who = 'W'
    new_state.whose_turn = new_who

    if new_state.board is not armanasu_last_state:  # check if opponent blocked lines
        diff = None
        for i, row in enumerate(new_state.board):
            for j, tile in enumerate(row):
                if new_state.board[i][j] != armanasu_last_state[i][j]:
                    diff = (i, j)
        for line in armanasu_lines:
            if diff in line:
                if valid_line(new_state, line) == (False, False):
                    armanasu_lines.remove(line)

    for (state, depth) in armanasu_fail_states:  # move failure states down
        armanasu_fail_states.remove((state, depth))
        if depth > 2:
            armanasu_fail_states.append((state, depth - 2))
    for (state, depth) in armanasu_win_states:  # move win states down
        armanasu_win_states.remove((state, depth))
        if depth > 2:
            armanasu_win_states.append((state, depth - 2))
    # Place a new tile
    last_move = False
    move = False
    timeout_search = False
    ply = 0
    last_optim = None
    optim = 0
    win_state = False
    while not timeout_search:
        if who == 'B':
            optim = float('inf')
        if who == 'W':
            optim = float('-inf')
        for i, row in enumerate(new_state.board):
            for j, tile in enumerate(row):
                if new_state.board[i][j] == ' ':  # only search valid locations
                    search_state = MY_TTS_State(new_state.board)
                    search_state.board[i][j] = who
                    (val, x, y, z, timeout) = recursive_minimax(search_state, ply, True, basic=False,
                                                                maximize=new_who == 'W', timed=True,
                                                                init_time=t0, timer=time_limit,
                                                                use_fail_states=True, geq=False)

                    timeout_search = timeout
                    if val > 10000000 or val < -10000000:  # someone wins at this depth, don't look deeper
                        win_state = True
                    if who == 'B' and val < optim:  # keep track of best move and its value
                        optim = val
                        move = (i, j)
                    elif who == 'W' and val > optim:
                        optim = val
                        move = (i, j)

                    if timeout_search:
                        break
            if timeout_search:
                break
        if optim == float('inf') or optim == float('-inf'):  # went too deep, game ended
            move = last_move
            optim = last_optim
            break
        if not timeout_search:  # we need to keep these values for if we time out
            last_move = move
            last_optim = optim
        if win_state:  # game ending can be forced by some side at this depth
            break
        ply += 1
    if timeout_search:
        move = last_move
        optim = last_optim
    if move:
        new_state.board[move[0]][move[1]] = who  # Construct a representation of the move that goes from the
        # currentState to the newState.

    for line in armanasu_lines:  # check if I blocked lines
        if move in line:
            if valid_line(new_state, line) == (False, False):
                armanasu_lines.remove(line)

    # Make up a new remark
    current_val = new_state.static_eval()
    if who == 'B':
        optim = -optim
        current_val = -current_val
    if move:
        new_utterance = chatter(optim, current_val, who)
    else:
        new_utterance = "A tie? I really thought I had you going there!"
    armanasu_last_state = new_state.board  # track the board state so I can know what move
    # the opponent made
    return [[move, new_state], new_utterance]


def chatter(static_val, current_val, who):  # actually decides how to respond, based
    # off of the current board state, and predicted
    # future board state
    global ARMANASU_INITIAL_SIDE, punt_count
    string = punt()

    if current_val > 9999999:
        string = "And the game is done! Better luck next time."
    elif static_val < -9999999:
        opts = ["Awww, I lost? Maybe I shoulda listened to that advice " + \
                "'bout blabbing on, then.",
                "Oh, no. No no no. I don't like this."]
        string = opts[punt_count % len(opts)]
    elif static_val > 9999999:
        opts = ["Ohohoh, this game is about to be FINISHED!"]
        string = opts[punt_count % len(opts)]
    elif static_val - current_val > 100:
        opts = ["Things are looking up for ol' Guff now, aren't they?",
                "Haha, take that!",
                "I hope ya don't mind me building my lead, not that there's much ya can do about it.",
                "Another step closer to victory."]
        string = opts[punt_count % len(opts)]
    elif static_val - current_val < -100:
        opts = ["Can I take that move back? I think I may have made a mistake here.",
                "Aww, no, this ain't looking good.",
                "This is about to get a lot worse, ain't it?"]
        string = opts[punt_count % len(opts)]
    elif static_val == 0:
        opts = ["Seems like we're reaching a tie, eh?",
                "And we'll end up back to where we started.",
                "Who's in the lead now? I honestly can't say."]
        string = opts[punt_count % len(opts)]
    elif current_val == 0:
        opts = ["Let's make this FUN, shall we?",
                "Evenly matched, it seems. Lets see if this pulls me ahead.",
                "I'd almost call the board pretty. Pretty BORING! So I'll spice it up a little!"]
        string = opts[punt_count % len(opts)]
    elif static_val > 500:
        opts = ["I seem to be doing well for myself, wouldn't ya say?",
                "Careful! Ya never know which move might be yer last.",
                "Stay focused Guff, ya got this.",
                "Look for the victory, yer in the lead already."]
        string = opts[punt_count % len(opts)]
    elif static_val < -500:
        opts = ["Okay, okay, you're hot, but ya haven't won yet!",
                "Sheesh, where'd I go so wrong?",
                "I gotta think long 'n hard about this one.",
                "Can't let 'em win so easy, Guff!"]
        string = opts[punt_count % len(opts)]
    elif static_val * current_val > 0:
        opts = ["The lead remains, I suppose.",
                "Staying ahead, but parlaying that into a victory's something else altogether.",
                "Once ya get ahead, ya tend ta stay ahead."]
        string = opts[punt_count % len(opts)]
    elif who is not ARMANASU_INITIAL_SIDE and punt_count % 6 == 0:
        opts = ["Wait a quick minute, am I playing against myself?",
                "I think I really am!",
                "Ohh, this grog might have been a bit too strong for me.",
                "It's either the grog, or someone's pullin' me leg.",
                "Who cares, at least I can see if I can beat me!"]
        string = opts[punt_count % len(opts)]
        ARMANASU_INITIAL_SIDE = who

    return string + suff()


PUNTS = ["Well, ya stumped me. I got nothing to say!",
         "Now this is much more interesting than Tic-Tac-Toe, I'll give it that",
         "What do you think about when you play? I like forks, and not the eating kind!",
         "I'll give ya a hint, free of charge: give up, while you can! Hehehe.",
         "Ya think you got me? Cuz ya don't. Just saying."]
punt_count = 0


def punt():  # helper function to iterate over various standard responses.
    global punt_count
    punt_count += 1
    return PUNTS[punt_count % 5]


SUFFS = [" Anyways, could ya pass the grog?",
         " Ba daaa, da da daaaa, ba daaa, da da daaa...",
         " Ah, dropped me pen. One second.",
         " How experienced are ya with this, by the by?",
         " Are you worried? I ain't worried.",
         "",
         " What's going on in that head a' yers, I wonder."]
suff_count = 0


def suff():  # helper function to iterate over various standard response suffixes.
    global suff_count
    suff_count += 1
    return SUFFS[suff_count % 7]


def moniker():
    return "Guff"  # Returns this agent's short nickname.


def who_am_i():
    return """My name's Guff Grenn, created by yours truly, Matei Armanasu. (NetID: armanasu)
People like to say I blab on too much, but what's the fun in a game without
small talk? It's madness I tell ya! Who cares if ya know what I think about
the game? You can see it too, can't you?"""


def get_ready(initial_state, k, who_i_play, player2Nickname):
    global ARMANASU_PLAYER_SIDE, ARMANASU_INITIAL_SIDE, k_val, armanasu_lines
    global armanasu_last_state, armanasu_fail_states, armanasu_win_states
    ARMANASU_PLAYER_SIDE = who_i_play
    ARMANASU_INITIAL_SIDE = who_i_play  # purely for extra dialogue when bot plays itself
    armanasu_last_state = initial_state.board  # for checking which move the opponent made
    armanasu_fail_states = []  # array of tiles opponent could place to win
    armanasu_win_states = []  # array of tiles we could place to win
    k_val = k
    n = len(initial_state.board)
    armanasu_lines = []  # which lines of play can at least one player win on
    for i in range(n):
        row = [(i, j) for j in range(n)]
        col = [(j, i) for j in range(n)]
        diag = [((j + i) % n, j) for j in range(n)]
        diag2 = [((i - j) % n, j) for j in range(n)]
        if valid_line(initial_state, row) > (False, False):  # at least one of the tuple returned is true
            armanasu_lines.append(row)
        if valid_line(initial_state, col) > (False, False):  # at least one of the tuple returned is true
            armanasu_lines.append(col)
        if valid_line(initial_state, diag) > (False, False):  # at least one of the tuple returned is true
            armanasu_lines.append(diag)
        if valid_line(initial_state, diag2) > (False, False):  # at least one of the tuple returned is true
            armanasu_lines.append(diag2)
    return "OK"


def valid_line(state, line):  # checks if a given line of victory can still result
    global k_val  # in a win, returns a tuple with two booleans, whether
    bd = state.board  # white can win the row and whether black can win the row
    blocks = chunk(state, line)
    n = len(bd)

    index = 0
    for k in range(len(blocks)):  # add the blank blocks to their adjacent line
        n = len(blocks)  # types, making sure not to double add and loop
        block = blocks[index]  # around properly, while removing the blank blocks
        size = block[1]  # from the list
        owner = block[0]
        if owner == ' ':
            if size >= k_val:  # case where there's a blank block big enough to allow
                return (True, True)  # a line of either color ignoring its boundaries
            blocks[(index + 1) % n][1] += size
            if blocks[(index + 1) % n][0] != blocks[(index - 1) % n][0]:
                blocks[(index - 1) % n][1] += size
            blocks.pop(index)
        else:
            index += 1  # either an element is removed from the list or the index increments

    index = 0
    for k in range(len(blocks)):  # add together the adjacent sections of line types
        n = len(blocks)  # while making sure to loop around properly and remove
        block = blocks[index]  # added blocks from the list
        if len(blocks) >= 2 and block[0] == blocks[(index + 1) % n][0]:
            blocks[(index + 1) % n][1] += block[1]
            blocks.pop(index)
        else:
            index += 1  # either an element is removed from the list or the index increments

    white = False
    black = False
    for block in blocks:  # check if there exists a possible line by which 'B' or 'W'
        size = block[1]  # can win.
        owner = block[0]
        if size >= k_val:
            if owner == 'B':
                black = True
            elif owner == 'W':
                white = True
    return (white, black)


def side_win(state, line, who):  # checks if a side has won a line across the board
    global k_val
    blocks = chunk(state, line)
    for block in blocks:
        if block[1] >= k_val and block[0] == who:
            return True
    return False


def did_win(state, who):  # checks if a side has won the board
    global k_val, armanasu_lines
    for line in armanasu_lines:
        if side_win(state, line, who):
            return True
    return False


def chunk(state, line):  # group up all the elements along the line into contiguous
    bd = state.board  # blocks of one type of tile, ' ','W','B', or '-'
    blocks = []
    curr = None
    for i, j in line:  # group up all the elements along the line into contiguous
        val = bd[i][j]  # blocks of one type of tile, ' ','W','B', or '-'
        if val != curr:
            curr = val
            blocks.append([curr, 1])
        else:
            blocks[-1][1] += 1

    if len(blocks) >= 2 and blocks[-1][0] == blocks[0][0]:  # simplify looparound
        blocks[0][1] += blocks[-1][1]
        blocks.pop()
    return blocks
