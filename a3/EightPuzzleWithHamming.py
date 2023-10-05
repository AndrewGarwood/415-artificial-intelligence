"""EightPuzzleWithHamming.pu
This file augments EightPuzzle.py with heuristic information to be
used in an A* implementation. The particular heuristic is the summation of
the hamming distance for each tile

By: Andrew Garwood
"""

from EightPuzzle import *


# Map numbers 0, 1, 2, 3, 4, 5, 6, 7, 8 to their goal column index
# Note to Self: x_coordinate = number % 3

def h(state):
    """Return the sum of the horizontal displacement of each tile"""
    hamming_sum = 0

    # concise implementation
    for number in range(9):
        number_coordinate = find_num_location(state, number)
        x_coord = number_coordinate[1]
        hamming_sum += abs(x_coord - (number % 3))

    return hamming_sum


# find coordinate of number; essentially same as find_void_location
def find_num_location(state, num):
    for i in range(3):
        for j in range(3):
            if state.b[i][j] == num:
                return i, j
    raise Exception("Given number not found in state " + str(state))
