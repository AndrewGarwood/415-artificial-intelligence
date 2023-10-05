"""
by  Andrew William Garwood
Email: awg1024@uw.edu
Student Number: 1835687

PUT YOUR ADDITIONAL COMMENTS HERE, SUCH AS EMAIL ADDRESS, VERSION INFO, ETC>

This file Includes a priority queue implementation by
 S. Tanimoto, Univ. of Washington.
Paul G. Allen School of Computer Science and Engineering

Intended USAGE:
 python3 AStar.py FranceWithCosts


"""

VERBOSE = True  # Set to True to see progress; but it slows the search.

import sys

if sys.argv == [''] or len(sys.argv) < 2:
    try:
        import FranceWithDXHeuristic as Problem
    except:
        print("Note that the EightPuzzle formulation will be used in Assignment 3, not Assignment 2")
        print("Try python3 AStar.py FranceWithCosts")

else:
    import importlib

    Problem = importlib.import_module(sys.argv[1])

print("\nWelcome to AStar, by Andrew Garwood!")

h = Problem.h  # import the dx heuristic

COUNT = None  # Number of nodes expanded.
MAX_OPEN_LENGTH = None  # How long OPEN ever gets.
SOLUTION_PATH = None  # List of states from initial to goal, along lowest-cost path.
TOTAL_COST = None  # Sum of edge costs along the lowest-cost path.
BACKLINKS = {}  # Predecessor links, used to recover the path.

# The value g(s) represents the cost along the best path found so far
# from the initial state to state s.
g = {}  # We will use a global hash table to associate g values with states.


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


def runAStar():
    """This is an encapsulation of some setup before running
  AStar, plus running it and then printing some stats."""
    initial_state = Problem.CREATE_INITIAL_STATE()
    print("Initial State:")
    print(initial_state)
    global COUNT, BACKLINKS, MAX_OPEN_LENGTH, SOLUTION_PATH
    COUNT = 0
    BACKLINKS = {}
    MAX_OPEN_LENGTH = 0
    SOLUTION_PATH = AStar(initial_state)
    print(str(COUNT) + " states expanded.")
    print('MAX_OPEN_LENGTH = ' + str(MAX_OPEN_LENGTH))


def AStar(initial_state):
    """A Star Search. This is the actual algorithm."""
    global g, COUNT, BACKLINKS, MAX_OPEN_LENGTH, CLOSED, TOTAL_COST
    CLOSED = []
    BACKLINKS[initial_state] = None
    # The "Step" comments below help relate AStar's implementation to
    # those of Depth-First Search and Breadth-First Search.

    # STEP 1a. Put the start state on a priority queue called OPEN
    OPEN = My_Priority_Queue()
    OPEN.insert(initial_state, h(initial_state))
    # STEP 1b. Assign g=0 to the start state.
    g[initial_state] = 0.0

    # STEP 2. If OPEN is empty, output “DONE” and stop.
    while len(OPEN) != 0:
        # LEAVE THE FOLLOWING CODE IN PLACE TO INSTRUMENT AND/OR DEBUG YOUR IMPLEMENTATION
        if VERBOSE:
            report(OPEN, CLOSED, COUNT)

        if len(OPEN) > MAX_OPEN_LENGTH:  # if len(OPEN) has exceeded the max length, update the max length
            MAX_OPEN_LENGTH = len(OPEN)

        # STEP 3. Select the state on OPEN having lowest priority value and call it S.
        #         Delete S from OPEN.
        #         Put S on CLOSED.
        #         If S is a goal state, output its description
        (S, P) = OPEN.delete_min()
        # print("In Step 3, returned from OPEN.delete_min with results (S,P)= ", (str(S), P))
        CLOSED.append((S, P))

        if Problem.GOAL_TEST(S):
            print(Problem.GOAL_MESSAGE_FUNCTION(S))
            path = backtrace(S)
            TOTAL_COST = g[S]
            # print("Solution path found: " +  path)
            # print("Length of solution path found " + str(len(path)) + " edges")
            print("Total cost of path: " + str(TOTAL_COST))
            return path

        COUNT += 1

        # STEP 4. Generate each successor of S
        #         and if it is already on CLOSED, delete the new instance.

        # f(s) = g(s) + h(s)
        # f(s) -> represents cost (distance) of a shortest path that starts at the start node,
        #         goes through s, and ends at a goal node
        # g(s) represents the cost of a shortest path from the start node to s
        # h(s) represents the cost of a shortest path from s to a goal node
        # h(s) is a heuristic function i.e. estimate ?
        # f(s) for current state S

        L = []
        # Generate list L of [successor, f(successor)] pairs

        for operator in Problem.OPERATORS:
            if operator.precond(S):
                successor = operator.state_transf(S)
                # Compute f(successor)
                successor_g = S.edge_distance(successor) + g[S]  # calculate current g for s'
                successor_fs = successor_g + h(successor)  # calculate current f(s) for s'

                if not OPEN.__contains__(successor):
                    g[successor] = successor_g
                elif successor_g < g[successor]:
                    g[successor] = successor_g
                # else do not update g. i.e. only update g value if it is better than the original

                # add successor to L
                L.append((successor, successor_fs))

                # following while loop is to determine if successor is in closed. having closed be a hashmap would be
                # more efficient; I can come back to this later and change it to a hashmap
                successor_in_CLOSED = False  # boolean for if successor is in closed
                successor_index_in_CLOSED = -1  # index for successor in closed, default value is -1 for not found
                i = 0
                while i < len(CLOSED) and not successor_in_CLOSED:
                    if CLOSED[i][0] == successor:  # if we find the s' is in closed
                        successor_in_CLOSED = True  # s' is indeed in closed
                        successor_index_in_CLOSED = i  # record index of s' in closed for future reference
                    i += 1  # Either we found s' in closed or we didn't, increment i and keep searching in necessary

                if successor_in_CLOSED:
                    q = CLOSED[successor_index_in_CLOSED][1]  # successors old f(s)
                    # if f(successor) >= f(successor in closed), remove successor from L
                    if successor_fs >= q:
                        L.remove((successor, successor_fs))
                    elif successor_fs < q:  # if f(s') < f(s' in closed) remove one in closed
                        CLOSED.remove((successor, q))
                elif OPEN.__contains__(successor):  # if successor is in open
                    q = OPEN.__getitem__(successor)  # q = priority of successor in open;
                    if successor_fs >= q:  # if successor's current fs >= q, remove successor from OPEN; it's not optimal
                        L.remove((successor, successor_fs))
                    elif successor_fs < q:  # if successor's current fs < q
                        OPEN.__delitem__(successor)  # del successor from OPEN; we found a better route

        for pair in L:  # for each pair of (state, priority) values in L
            BACKLINKS[pair[0]] = S  # Assign backlinks[successor] to S
            OPEN.insert(pair[0], pair[1])  # add all pairs to OPEN to be examined

        # STEP 6. Go to Step 2.

    return None  # No more states on OPEN, and no goal reached.


def print_state_queue(name, q):
    print(name + " is now: ", end='')
    print(str(q))


def backtrace(S):
    global BACKLINKS
    path = []
    while S:
        path.append(S)
        S = BACKLINKS[S]
    path.reverse()
    print("Solution path: ")
    for s in path:
        print(s)
    return path


def report(open, closed, count):
    print("len(OPEN)=" + str(len(open)), end='; ')
    print("len(CLOSED)=" + str(len(closed)), end='; ')
    print("COUNT = " + str(count))


# TODO: comment this; says so in spec
if __name__ == '__main__':
    runAStar()
