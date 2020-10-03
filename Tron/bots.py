#!/usr/bin/python

import numpy as np
from tronproblem import *
from trontypes import CellType, PowerupType
import random, math
import sys
from queue import Queue
import collections
import time
# Throughout this file, ASP means adversarial search problem.


class StudentBot:
    """ Write your student bot here"""
    def __init__(self):
        self.BOT_NAME = "Sir Williams"
    def decide(self, asp):
        """
        Input: asp, a TronProblem
        Output: A direction in {'U','D','L','R'}

        To get started, you can get the current
        state by calling asp.get_start_state()
        """
        state = asp.get_start_state()
        locs = state.player_locs
        board = state.board
        player_to_max = state.ptm
        # if ptm == 1:
        #     player_to_max = ptm
        # else:
        #     player_to_max
        loc = locs[state.ptm]
        possibilities = TronProblem.get_safe_actions(board, loc)

        cutoff_ply = 3
        depths = {}
        depths[state] = 0
        actions_dict = {}
        a = -sys.maxsize + 1
        b = sys.maxsize
        if not possibilities:
            return "U"
        # start_time = time.time()
        # period = .25
        for p in possibilities:
            next_state = asp.transition(state, p)
            depths[next_state] = depths[state] + 1
            score = self.cutoff_min(asp,next_state,player_to_max,a, b,depths, cutoff_ply, self.heuristic)
            a = max(score,a)
            actions_dict[score] = p
        return actions_dict[max(actions_dict.keys())]

    def heuristic(self, state, player_to_max):
        return self.bds(state, player_to_max)


    def cutoff_max(self, asp, state, player_to_max, a, b, depths, cutoff_ply, eval_func):
        if asp.is_terminal_state(state):
            return 10000*(asp.evaluate_state(state)[player_to_max])
        if self.cutoff_reached(depths[state], cutoff_ply):
            return eval_func(state,player_to_max)


        #actions = asp.get_available_actions(state)
        locs = state.player_locs
        actions = TronProblem.get_safe_actions(state.board, locs[state.ptm])
        highest = -sys.maxsize + 1
        for act in actions:
            next_state = asp.transition(state, act)
            depths[next_state] = depths[state] + 1
            highest = max(highest, self.cutoff_min(asp, next_state, player_to_max, a, b, depths, cutoff_ply, eval_func))
            if highest >= b:
                return highest
            a = max(a, highest)
        if highest == -sys.maxsize + 1:
            highest = sys.maxsize
        return highest

    def cutoff_min(self, asp, state, player_to_max, a, b, depths, cutoff_ply, eval_func):
        if asp.is_terminal_state(state):
            return 10000*(asp.evaluate_state(state)[player_to_max])
        if self.cutoff_reached(depths[state], cutoff_ply):
            return eval_func(state,player_to_max)
        #actions = asp.get_available_actions(state)
        locs = state.player_locs
        actions = TronProblem.get_safe_actions(state.board, locs[state.ptm])
        lowest = sys.maxsize
        for act in actions:
            next_state = asp.transition(state, act)
            depths[next_state] = depths[state] + 1
            lowest = min(lowest, self.cutoff_max(asp, next_state, player_to_max, a, b, depths, cutoff_ply, eval_func))
            if lowest <= a:
                return lowest
            b = min(b, lowest)
        if lowest == sys.maxsize:
            lowest = -sys.maxsize +1
        return lowest

    def cutoff_reached(self, depth, cutoff_ply):
        if depth > cutoff_ply:
            return True
        else:
            return False

    def bds(self,state,player_to_max):
        visited = set()
        locs = state.player_locs
        board = state.board
        #use deque instead of a queue
        frontierOne = collections.deque()
        frontierTwo = collections.deque()
        frontierOne.append(locs[0])
        frontierTwo.append(locs[1])
        scoreOne = 0
        scoreTwo = 0
        visited.add(locs[0])
        visited.add(locs[1])

        while frontierOne or frontierTwo:
            if frontierOne:
                curr_loc = frontierOne.popleft()
                #no conversion to list
                #use time library to tell it to stop
                possibilities = TronProblem.get_safe_actions(board, curr_loc)
                scoreOne+=len(possibilities)+self.extra(board,curr_loc)

                for p in possibilities:
                    next_loc = TronProblem.move(curr_loc, p)
                    if next_loc not in visited:
                        visited.add(next_loc)
                        frontierOne.append(next_loc)

            if frontierTwo:
                curr_loc = frontierTwo.popleft()
                possibilities = TronProblem.get_safe_actions(board, curr_loc)
                scoreTwo+=len(possibilities) + self.extra(board,curr_loc)

                for p in possibilities:
                    next_loc = TronProblem.move(curr_loc, p)
                    if next_loc not in visited:
                        visited.add(next_loc)
                        frontierTwo.append(next_loc)
        if player_to_max == 0:
            return scoreOne-scoreTwo
        else:
            return scoreTwo-scoreOne

    def extra(self,board, loc):
        r, c = loc
        if board[r][c]==CellType.ARMOR:
            return 3
        if board[r][c]==CellType.BOMB:
            return 4
        if board[r][c]==CellType.SPEED:
            return -2
        if board[r][c]==CellType.TRAP:
            return 2
        return 0
    def cleanup(self):
        pass


class RandBot:
    """Moves in a random (safe) direction"""

    def decide(self, asp):
        """
        Input: asp, a TronProblem
        Output: A direction in {'U','D','L','R'}
        """
        state = asp.get_start_state()
        locs = state.player_locs
        board = state.board
        ptm = state.ptm
        loc = locs[ptm]
        possibilities = list(TronProblem.get_safe_actions(board, loc))
        if possibilities:
            return random.choice(possibilities)
        return "U"

    def cleanup(self):
        pass


class WallBot:
    """Hugs the wall"""

    def __init__(self):
        order = ["U", "D", "L", "R"]
        random.shuffle(order)
        self.order = order

    def cleanup(self):
        order = ["U", "D", "L", "R"]
        random.shuffle(order)
        self.order = order

    def decide(self, asp):
        """
        Input: asp, a TronProblem
        Output: A direction in {'U','D','L','R'}
        """
        state = asp.get_start_state()
        locs = state.player_locs
        board = state.board
        ptm = state.ptm
        loc = locs[ptm]
        possibilities = list(TronProblem.get_safe_actions(board, loc))
        if not possibilities:
            return "U"

        decision = possibilities[0]
        for move in self.order:
            if move not in possibilities:
                continue
            next_loc = TronProblem.move(loc, move)
            if len(TronProblem.get_safe_actions(board, next_loc)) < 3:
                decision = move
                break
        return decision
