"""Farmer_Fox.py
by Andrew William Garwood
UWNetID: awg1024
Student number: 1835687

Assignment 2, in CSE 415, Autumn 2019.

This file contains my problem formulation for the problem of
the Farmer, Fox, Chicken, and Grain.
"""

# Put your formulation of the Farmer-Fox-Chicken-and-Grain problem here.
# Be sure your name, uwnetid, and 7-digit student number are given above in 
# the format shown.
farmer_index = 0
fox_index = 1
chicken_index = 2
grain_index = 3
left_side = 0
right_side = 1


class State:

    def __init__(self, dictionary=None):
        if dictionary is None:
            dictionary = {'entities': [[0, 0, 0, 0], [0, 0, 0, 0]],
                          'boat_location': left_side}
            # dictionary formulation:
            # dictionary['entities'] returns list of two lists
            # dictionary['entities'][left_side] returns list of counts of entities on the left side; left_side = 0
            # dictionary['entities'][right_side] returns list of counts of entities on the right side; right_side = 1
            # dictionary['entities'][left_side][farmer_index] returns count of farmers on left side;
            # the pattern on the previous lines persist for all indexes initialized above State

        self.dictionary = dictionary

    def __eq__(self, other):
        for item in ['entities', 'boat_location']:
            if self.dictionary[item] != other.dictionary[item]:
                return False

        return True

    def __str__(self):
        # Returns textual description of the state
        entity_lists = self.dictionary['entities']
        # text = "\n Farmer(s) on left: " + str(entity_lists[left_side][farmer_index]) + "\n"
        # text += " Foxes"
        text = "\n"
        if entity_lists[left_side][farmer_index] == 1:
            text += " Farmer is on the left side. \n"
        else:
            text += " Farmer is on the right side \n"

        if entity_lists[left_side][fox_index] == 1:
            text += " Fox is on the left side. \n"
        else:
            text += " Fox is on the right side. \n"

        if entity_lists[left_side][chicken_index] == 1:
            text += " Chicken is on the left side. \n"
        else:
            text += " Chicken is on the right side. \n"

        if entity_lists[left_side][grain_index] == 1:
            text += " Grain is on the left side. \n"
        else:
            text += " Grain is on the right side. \n"

        current_side = "left"

        if self.dictionary['boat_location'] == right_side:
            current_side = "right"
        text += " Boat is on the " + current_side + " side.\n"

        return text

    def __hash__(self):
        return (self.__str__()).__hash__()

    def copy(self):
        # Performs appropriately deep copy of state
        # used by operators to create new states
        new_state = State({})  # pass in an empty dictionary

        # initialize the copy's dictionary with self's dictionary using a nifty for loop
        new_state.dictionary['entities'] = [self.dictionary['entities'][side][:] for side in [left_side, right_side]]
        new_state.dictionary['boat_location'] = self.dictionary['boat_location']

        return new_state

    def can_move(self, farmer_count, fox_count, chicken_count, grain_count):
        # returns whether or not one can move the boat across
        current_side = self.dictionary['boat_location']  # current location of boat
        entity_list = self.dictionary['entities']

        if farmer_count < 1:  # need farmer to steer boat
            return False

        if (fox_count + chicken_count + grain_count) > 1:  # if user tries to take more than one entity with Farmer
            return False

        farmers_available = entity_list[current_side][farmer_index]
        if farmers_available < farmer_count:  # user tries to take more farmers than available
            return False

        foxes_available = entity_list[current_side][fox_index]
        if foxes_available < fox_count:  # user tries to take more foxes than available
            return False

        chickens_available = entity_list[current_side][chicken_index]
        if chickens_available < chicken_count:  # user tries to take more chickens than available
            return False

        grain_available = entity_list[current_side][grain_index]
        if grain_available < grain_count:  # user tries to take more grain than available
            return False

        farmers_remaining = farmers_available - farmer_count
        foxes_remaining = foxes_available - fox_count
        chickens_remaining = chickens_available - chicken_count
        grain_remaining = grain_available - grain_count

        if farmers_remaining == 0 and foxes_remaining == 1 and chickens_remaining == 1:
            # if user leaves fox and chicken alone
            return False

        if farmers_remaining == 0 and chickens_remaining == 1 and grain_remaining == 1:
            # if user leaves chicken and grain alone
            return False

        # Don't think we have to do the "at arrival" checks as the farmer will always be there at arrival to ensure no
        # entity is eaten by another.
        # farmers_at_arrival = entity_list[1 - current_side] + farmer_count
        # foxes_at_arrival
        return True

    def move(self, farmer_count, fox_count, chicken_count, grain_count):
        # If a move is legal, create the new state that results after moving the boat and entities
        new_state = self.copy()  # copy previous state
        current_side = self.dictionary['boat_location']  # gets current location of boat
        entity_list = new_state.dictionary['entities']

        # remove entities from current_side
        entity_list[current_side][farmer_index] = entity_list[current_side][farmer_index] - farmer_count
        entity_list[current_side][fox_index] = entity_list[current_side][fox_index] - fox_count
        entity_list[current_side][chicken_index] = entity_list[current_side][chicken_index] - chicken_count
        entity_list[current_side][grain_index] = entity_list[current_side][grain_index] - grain_count

        # Add entities being moved to opposite side
        other_side = 1 - current_side
        entity_list[other_side][farmer_index] = entity_list[other_side][farmer_index] + farmer_count
        entity_list[other_side][fox_index] = entity_list[other_side][fox_index] + fox_count
        entity_list[other_side][chicken_index] = entity_list[other_side][chicken_index] + chicken_count
        entity_list[other_side][grain_index] = entity_list[other_side][grain_index] + grain_count

        # Move the Boat
        new_state.dictionary['boat_location'] = other_side

        return new_state


def goal_test(state):
    # if all entities are on right side, then param 'state' is a goal state
    entity_list = state.dictionary['entities']
    for i in range(0, 4):  # i represents the indices of each entity, we check each entity is on the right
        if entity_list[right_side][i] != 1:  # if an entity is missing, it's not a goal state, return false
            return False
    return True


def goal_message(state):
    return "Nice job, you successfully guided the farmer, the animals, and the grain across the river!"


class Operator:
    def __init__(self, name, precond, state_transf):
        self.name = name
        self.precond = precond
        self.state_transf = state_transf

    def is_applicable(self, state):
        return self.precond(state)

    def apply(self, state):
        return self.state_transf(state)


# <INITIAL_STATE>
CREATE_INITIAL_STATE = lambda: State(dictionary={'entities': [[1, 1, 1, 1], [0, 0, 0, 0]], 'boat_location': left_side})
# </INITIAL_STATE>


# <OPERATORS>
entity_combinations = [(0, 1, 0, 1), (1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1), (1, 1, 0, 0), (1, 0, 1, 0),
                       (1, 0, 0, 1), (1, 1, 1, 1)]

OPERATORS = [Operator(
    "Cross the river with " + str(farmer_count) + " farmer(s), " + str(fox_count) + " fox(es), " + str(chicken_count) +
    " chicken(s), and " + str(grain_count) + " grain. ",
    lambda state, farmer_param = farmer_count, fox_param = fox_count, chicken_param = chicken_count, grain_param =
    grain_count: state.can_move(farmer_param, fox_param, chicken_param, grain_param),
    lambda state, farmer_param = farmer_count, fox_param = fox_count, chicken_param = chicken_count,
    grain_param = grain_count: state.move(farmer_param, fox_param, chicken_param, grain_param))
    for (farmer_count, fox_count, chicken_count, grain_count) in entity_combinations]

# <GOAL_TEST>
GOAL_TEST = lambda state: goal_test(state)
# </GOAL_TEST>

# <GOAL_MESSAGE_FUNCTION>
GOAL_MESSAGE_FUNCTION = lambda state: goal_message(state)
# </GOAL_MESSAGE_FUNCTION>

