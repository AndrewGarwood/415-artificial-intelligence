"""EightPuzzleWithHamming.pu
This file augments EightPuzzle.py with heuristic information to be
used in an A* implementation. The particular heuristic is the summation of
the Manhattan distance for each tile

By: Andrew Garwood
"""

from EightPuzzle import *

# Map numbers 0, 1, 2, 3, 4, 5, 6, 7, 8 to their goal coordinate
# Note to Self: x_coordinate = number % 3
desired_location = {0: (0, 0),
                    1: (0, 1),
                    2: (0, 2),
                    3: (1, 0),
                    4: (1, 1),
                    5: (1, 2),
                    6: (2, 0),
                    7: (2, 1),
                    8: (2, 2)}


def h(state):
    """Return the sum of the horizontal and vertical displacement of each tile"""

    # Note: Could use fact that desired_x_coord = number % 3 to save space
    manhattan_sum = 0
    for number in range(9):
        number_coordinate = find_num_location(state, number)
        x_coord = number_coordinate[1]
        y_coord = number_coordinate[0]

        manhattan_sum += abs(x_coord - desired_location[number][1])  # add x displacement for number
        manhattan_sum += abs(y_coord - desired_location[number][0])  # add y displacement for number

    return manhattan_sum


def find_num_location(state, num):
    for i in range(3):
        for j in range(3):
            if state.b[i][j] == num:
                return i, j
    raise Exception("Given number not found in state " + str(state))
