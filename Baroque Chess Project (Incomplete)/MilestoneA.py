#!/usr/bin/python3
"""MilestoneA.py
This runnable file will provide a representation of
answers to key questions about your project in CSE 415.

"""

# DO NOT EDIT THE BOILERPLATE PART OF THIS FILE HERE:

CATEGORIES = ['Baroque Chess Agent', 'Wicked Problem Formulation and A* Search',
              'Backgammon Agent that Learns', 'Hidden Markov Models: Algorithms and Applications']


class Partner:
    def __init__(self, lastname, firstname, uwnetid):
        self.uwnetid = uwnetid
        self.lastname = lastname
        self.firstname = firstname

    def __lt__(self, other):
        return (self.lastname + "," + self.firstname).__lt__(other.lastname + "," + other.firstname)

    def __str__(self):
        return self.lastname + ", " + self.firstname + " (" + self.uwnetid + ")"


class Who_and_what():
    def __init__(self, team, option, title, approach, workload_distribution, references):
        self.team = team
        self.option = option
        self.title = title
        self.approach = approach
        self.workload_distribution = workload_distribution
        self.references = references

    def report(self):
        rpt = 80 * "#" + "\n"
        rpt += '''The Who and What for This Submission

Project in CSE 415, University of Washington, Autumn, 2019
Milestone A

Team: 
'''
        team_sorted = sorted(self.team)
        # Note that the partner whose name comes first alphabetically
        # must do the turn-in.
        # The other partner(s) should NOT turn anything in.
        rpt += "    " + str(team_sorted[0]) + " Andrew Garwood\n"
        for p in team_sorted[1:]:
            rpt += "    " + str(p) + " Changyu Li\n\n"

        rpt += "Option: " + str(self.option) + "\n\n"
        rpt += "Title: " + self.title + "\n\n"
        rpt += "Approach: " + self.approach + "\n\n"
        rpt += "Workload Distribution: " + self.workload_distribution + "\n\n"
        rpt += "References: \n"
        for i in range(len(self.references)):
            rpt += "  Ref. " + str(i + 1) + ": " + self.references[i] + "\n"

        rpt += "\n\nThe information here indicates that the following file will need\n" + \
               "to be submitted (in addition to code and possible data files):\n"
        rpt += "    " + \
               {'1': "Baroque_Chess_Agent_Report", '2': "Wicked_Problem_Forulation_Report", \
                '3': "Backgammon_Agent_That_Learns_Report", '4': "Hidden_Markov_Models_Report"} \
                   [self.option] + ".pdf\n"

        rpt += "\n" + 80 * "#" + "\n"
        return rpt


# END OF BOILERPLATE.

# Change the following to represent your own information:

andrew = Partner("Garwood", "Andrew", "awg1024")
changyu = Partner("Li", "Changyu", "changl28")
team = [andrew, changyu]

OPTION = '1'
# Legal options are 1, 2, 3, and 4.

title = "A Blustering Baroque Chess Player"

approach = '''Our approach will be to first understand the rules,
then code our move generator, develop a static evaluation function,
a personality for the agent, and then optimize using alpha-beta
pruning, Zobrist hashing, and comparison of alternative static
evaluation functions.

For me (Andrew) specifically, I will familiarize myself with the rules of the game.
This will allow my partner and I to develop our move generator. For the static evaluation,
I intend to have us develop our own implementations separately, then compare
and see which one is better (we can combine features too). Although if we are pressed
on time, perhaps we will only develop one. We will improve upon our previous
implementation of alpha-beta, and work together to properly implement Zobrist
hashing.
'''

workload_distribution = '''Andrew and Changyu will hopefully have an even distribution of work between each other
We will do our best to pair program most of the features except for the aforementioned method to develop a
static evaluation. Alpha-beta should not be a huge hurdle to deal with as we will improve our previous work.
We will also work together on Zobrist hashing and the move generator. We plan to check in
with one another often enough so that we feel we are making meaningful and correct progress;
we will likely keep a line of communication open with each other and the TA to ensure everything
is going okay.

Name of agent is subject to change as we decide its personality
'''

reference1 = '''Wikipedia article on Baroque Chess;
    URL: https://en.wikipedia.org/wiki/Baroque_chess (accessed Nov. 18, 2019)'''

reference2 = '''"What\'s Wrong with Ultima," by Robert Abbott,
    available online at: http://www.logicmazes.com/games/wgr.html'''

our_submission = Who_and_what(team, OPTION, title, approach, workload_distribution, [reference1, reference2])

# You can run this file from the command line by typing:
# python3 who_and_what.py

# Running this file by itself should produce a report that seems correct to you.
if __name__ == '__main__':
    print(our_submission.report())
