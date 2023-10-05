'''BFS.py
by Andrew William Garwood
UWNetID: awg1024
Student number: 1835687

Assignment 2, in CSE 415, Autumn 2019.

This file contains my implementation for BFS
'''

import sys

if sys.argv == [''] or len(sys.argv) < 2:
    import Farmer_Fox as Problem
else:
    import importlib

    Problem = importlib.import_module(sys.argv[1])

print("\nWelcome to BFS")
COUNT = None
BACKLINKS = {}
MAX_OPEN_LENGTH = 0


def runBFS():
    initial_state = Problem.CREATE_INITIAL_STATE()
    print("Initial State:")
    print(initial_state)
    global COUNT, BACKLINKS, MAX_OPEN_LENGTH
    COUNT = 0
    BACNKLINKS = {}
    MAX_OPEN_LENGTH = 0
    BFS(initial_state)
    print(str(COUNT) + " states expanded.")
    print("MAX_OPEN_LENGTH = " + str(MAX_OPEN_LENGTH))


def BFS(initial_state):
    global COUNT, BACKLINKS, MAX_OPEN_LENGTH

    # Step 1: Put that start state on a list OPEN

    OPEN = [initial_state]
    CLOSED = []  # closed is empty as we have not processed anything yet?
    BACKLINKS[initial_state] = None  # For the backlinks dictionary, add None for initial state

    # Step 2: If OPEN is empty, output "DONE" and stop

    while OPEN:
        report(OPEN, CLOSED, COUNT)

        # Update max open length
        if len(OPEN) > MAX_OPEN_LENGTH:
            MAX_OPEN_LENGTH = len(OPEN)

        # Step 3: Select the first state on OPEN and call it S.
        #         Delete S from OPEN.
        #         Put S on CLOSED.
        #         If S is a goal state, output its description

        state = OPEN.pop(0)
        CLOSED.append(state)

        if Problem.GOAL_TEST(state):
            print(Problem.GOAL_MESSAGE_FUNCTION(state))
            path = backtrace(state)
            print("Length of solution path found: " + str(len(path) - 1) + " edges.")
            return

        COUNT += 1

        # Step 4: Generate the list of L successors and delete
        #         from L those states already appearing on CLOSED.

        successors = []
        for operator in Problem.OPERATORS:
            if operator.precond(state):
                new_state = operator.state_transf(state)
                if not new_state in CLOSED:
                    successors.append(new_state)
                    # BACKLINKS[new_state] = state

        # Step 5: Delete from successors any members of OPEN that occur on successors
        #         Insert all members of L at the END of OPEN

        for successor in successors:
            for i in range(len(OPEN)):
                if successor.__eq__(OPEN[i]):  # if successor = OPEN[i]
                    successors.remove(successor)
                    break

        # for item in OPEN:
        #     for successor in range(len(successors)):
        #         if item.__eq__(successors[successor]):
        #             del successors[successor]
        #             break

        for item in successors:
            BACKLINKS[item] = state

        OPEN = OPEN + successors
        print_state_list("OPEN", OPEN)


def print_state_list(name, open_list):
    print(name + " is now: ", end='')
    for state in open_list[:-1]:
        print(str(state), end=', ')
    print(str(open_list[-1]))


def backtrace(state):
    global BACKLINKS
    path = []

    while state:
        path.append(state)
        state = BACKLINKS[state]

    path.reverse()
    print("Solution path: ")
    for state in path:
        print(state)

    return path


def report(open_list, closed_list, count):
    print("len(OPEN) = " + str(len(open_list)), end='; ')
    print("len(CLOSED) = " + str(len(closed_list)), end='; ')
    print("COUNT = " + str(count))


if __name__ == '__main__':
    runBFS()
