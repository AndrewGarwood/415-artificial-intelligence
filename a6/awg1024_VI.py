'''awg1024_VI.p

Value Iteration for Markov Decision Processes.
'''


# Edit the returned name to ensure you get credit for the assignment.
def student_name():
    return "Garwood, Andrew"

Vkplus1 = {}
Q_Values_Dict = {}


# compute 1 iteration of VI from the given MDP information plus the
# current state values from the dictionary Vk
# @param S: set of states
# @param A: set of actions
# T (s, a, s'): probability of taking action a from s to get to s'
# R (s, a, s'): reward for taking action a to get from s to s'
def one_step_of_VI(S, A, T, R, gamma, Vk):
    """S is list of all the states defined for this MDP.
    A is a list of all the possible actions.
    T is a function representing the MDP's transition model.
    R is a function representing the MDP's reward function.
    gamma is the discount factor.
    The current value of each state s is accessible as Vk[s].


    Your code should fill the dictionaries Vkplus1 and Q_Values_dict
    with a new value for each state, and each q-state, and assign them
    to the state's and q-state's entries in the dictionaries, as in
    Vkplus1[s] = new_value
    Q_Values_Dict[(s, a)] = new_q_value

    Also determine delta_max, which we define to be the maximum
    amount that the absolute value of any state's value is changed
    during this iteration.
    """

    global Vkplus1
    Vkplus1 = {}
    delta_max = 0.0

    global Q_Values_Dict
    for state in S:  # s_to_a_to_sPrimes_dict:
        max_q_sum = Vk[state]
        for action in A:  # s_to_a_to_sPrimes_dict[state]:
            q_sum = 0.0
            for s_prime in S:  # s_to_a_to_sPrimes_dict[state][action]:
                q_sum += T(state, action, s_prime) * (R(state, action, s_prime) + gamma * Vk[s_prime])
            Q_Values_Dict[(state, action)] = q_sum
            if q_sum > max_q_sum:
                max_q_sum = q_sum
        Vkplus1[state] = max_q_sum
        delta = abs(Vk[state] - Vkplus1[state])
        if delta > delta_max:
            delta_max = delta

    return Vkplus1, delta_max  # return updated Vk and delta_max


def return_Q_values(S, A):
    """Return the dictionary whose keys are (state, action) tuples,
    and whose values are floats representing the Q values from the
    most recent call to one_step_of_VI. This is the normal case, and
    the values of S and A passed in here can be ignored.
    However, if no such call has been made yet, use S and A to
    create the answer dictionary, and use 0.0 for all the values.
    """
    # S is list of States
    # A is list of actions
    global Q_Values_Dict
    if Q_Values_Dict:
        return Q_Values_Dict  # placeholder
    else:  # if Q_Values_Dict is empty
        for state in S:
            for action in A:
                Q_Values_Dict[(state, action)] = 0.0
    return Q_Values_Dict


Policy = {}


def extract_policy(S, A):
    """Return a dictionary mapping states to actions. Obtain the policy
    using the q-values most recently computed.  If none have yet been
    computed, call return_Q_values to initialize q-values, and then
    extract a policy.  Ties between actions having the same (s, a) value
    can be broken arbitrarily.
    """
    global Policy
    Policy = {}
    q_val_dict = return_Q_values(S, A)
    # policy extraction
    # Perform single step of expectimax
    # pi*(s) = arg max summation of [T(s, a, s')[R(s, a, s') + gamma * V*(s')]]
    # For every s in S, extract the best action based on the Q values for s, a

    # 
    # For each s
    # For every possible action from s, choose the action with the highest q value
    # Think I have to use an external storage structure as one cannot easily access all possible actions from s
    # wait that's Policy
    for state, action in q_val_dict:
        if Policy:
            if state in Policy:
                if q_val_dict[(state, action)] > q_val_dict[state, Policy[state]]:
                    Policy[state] = action
            else:
                Policy[state] = action
        else:  # we have yet to choose a policy for state, so assign it to the current s, a pair
            Policy[state] = action

    return Policy


def apply_policy(s):
    """Return the action that your current best policy implies for state s."""
    global Policy
    return Policy[s]
