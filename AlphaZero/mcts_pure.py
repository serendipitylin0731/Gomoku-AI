# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 21:19:11 2018

@author: initial-h
"""

import numpy as np
from operator import itemgetter
from collections import defaultdict


def policy_value_fn(board):
    '''
    a function that takes in a state and outputs a list of (action, probability)
    tuples and a score for the state
    '''
    # return uniform probabilities and 0 score for pure MCTS
    action_probs = np.ones(len(board.availables)) / len(board.availables)
    return zip(board.availables, action_probs), 0


class TreeNode(object):
    '''
    A node in the MCTS tree. Each node keeps track of its own value Q,
    prior probability P, and its visit-count-adjusted prior score u.
    '''

    def __init__(self, parent, prior_p):
        self._parent = parent
        self._children = {}  # a map from action to TreeNode
        self._n_visits = 0
        self._Q = 0
        self._u = 0
        self._P = prior_p  # its the prior probability that action's taken to get this node

    def expand(self, action_priors):
        '''扩展子节点'''
        for action, prob in action_priors:
            if action not in self._children:
                self._children[action] = TreeNode(self, prob)

    def select(self, c_puct):
        '''选择 UCB 值最大的子节点'''
        return max(self._children.items(),
                   key=lambda act_node: act_node[1].get_value(c_puct))

    def update(self, leaf_value):
        '''根据叶子值更新节点统计'''
        # 访问次数加一
        self._n_visits += 1
        # 增量更新 Q 值
        self._Q += 1.0 * (leaf_value - self._Q) / self._n_visits

    def update_recursive(self, leaf_value):
        '''递归更新路径上的节点'''
        # 非根节点先更新父节点
        if self._parent:
            self._parent.update_recursive(-leaf_value)
        self.update(leaf_value)

    def get_value(self, c_puct):
        '''计算并返回 UCB 值'''
        self._u = (c_puct * self._P *
                   np.sqrt(self._parent._n_visits) / (1 + self._n_visits))
        return self._Q + self._u

    def is_leaf(self):
        '''判断是否为叶子节点'''
        return self._children == {}

    def is_root(self):
        return self._parent is None


class MCTS(object):
    '''
    A simple implementation of Monte Carlo Tree Search.
    '''

    def __init__(self, policy_value_fn, c_puct=5, n_playout=400):
        '''
        policy_value_fn: a function that takes in a board state and outputs
            a list of (action, probability) tuples and also a score in [-1, 1]
            (i.e. the expected value of the end game score from the current
            player's perspective) for the current player.
        c_puct: a number in (0, inf) that controls how quickly exploration
            converges to the maximum-value policy. A higher value means
            relying on the prior more.
        '''
        self._root = TreeNode(parent=None, prior_p=1.0)
        # root node do not have parent ,and sure with prior probability 1
        self._policy = policy_value_fn
        self._c_puct = c_puct
        self._n_playout = n_playout  # times of tree search

    def _playout(self, state):
        '''执行一次模拟（选择→扩展→评估→回传）'''
        node = self._root
        while True:
            if node.is_leaf():
                break
            # UCB 选择
            action, node = node.select(self._c_puct)
            state.do_move(action)

        # 检查终局
        end, winner = state.game_end()
        if not end:
            # 未终局则扩展叶子
            action_probs, _ = self._policy(state)
            node.expand(action_probs)
            # 随机 rollout 评估
            leaf_value = self._evaluate_rollout(state)
        else:
            # 已终局，按当前玩家视角赋值
            if winner == -1:  # 平局
                leaf_value = 0.0
            else:
                leaf_value = (
                    1.0 if winner == state.get_current_player() else -1.0
                )
        # 回传更新路径节点
        node.update_recursive(-leaf_value)

    def _evaluate_rollout(self, state, limit=1000):
        '''随机 rollout 至终局'''
        player = state.get_current_player()
        for i in range(limit):
            end, winner = state.game_end()
            if end:
                break
            action_probs, _ = self._policy(state)
            action_probs = list(action_probs)
            # 按概率分布随机选动作
            action = np.random.choice(
                [a for a, _ in action_probs],
                p=[p for _, p in action_probs]
            )
            state.do_move(action)
        else:
            # 达到 rollout 上限
            print("WARNING: rollout reached move limit")

        if winner == -1:  # 平局
            return 0
        else:
            return 1 if winner == player else -1

    def get_move_probs(self, state, temp=1e-3):
        '''多次模拟后返回动作概率'''
        for n in range(self._n_playout):
            state_copy = state.copy()
            self._playout(state_copy)

        # 按根节点访问次数计算概率
        act_visits = [(act, node._n_visits)
                      for act, node in self._root._children.items()]
        acts, visits = zip(*act_visits)
        act_probs = softmax(1.0 / temp * np.log(np.array(visits) + 1e-10))

        return acts, act_probs

    def update_with_move(self, last_move):
        '''沿 last_move 前进并复用子树'''
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.0)

    def __str__(self):
        return "MCTS"


class MCTSPlayer(object):
    '''
    AI player based on MCTS
    '''

    def __init__(self, c_puct=5, n_playout=400):
        '''
        init a mcts class
        '''
        self.mcts = MCTS(policy_value_fn, c_puct, n_playout)

    def set_player_ind(self, p):
        '''
        set player index
        '''
        self.player = p

    def reset_player(self):
        '''
        reset player
        '''
        self.mcts.update_with_move(-1)  # reset the node

    def get_action(self, board, is_selfplay=False, print_probs_value=0):
        '''
        get an action by mcts
        do not discard all the tree and retain the useful part
        '''
        sensible_moves = board.availables
        if len(sensible_moves) == 0:
            print("WARNING: the board is full")
            return None, None

        # 获取 MCTS 策略分布
        acts, probs = self.mcts.get_move_probs(board, temp=1e-3)
        move_probs = np.zeros(board.width * board.height)
        move_probs[list(acts)] = probs

        if is_selfplay:
            # 自弈加入 Dirichlet 噪声探索
            move = np.random.choice(
                acts,
                p=0.75 * probs + 0.25 * np.random.dirichlet(0.3 * np.ones(len(probs)))
            )
            # 复用搜索子树
            self.mcts.update_with_move(move)
        else:
            # temp 趋于 0，近似选访问最多动作
            move = np.random.choice(acts, p=probs)
            # 重置根节点
            self.mcts.update_with_move(-1)

        if print_probs_value:
            act_probs, value = policy_value_fn(board)
            print('-' * 10)
            print('value', value)
            probs = np.array(move_probs).reshape((board.width, board.height)).round(3)[::-1, :]
            for p in probs:
                for x in p:
                    print("{0:6}".format(x), end='')
                print('\r')

        return move, move_probs

    def __str__(self):
        return "MCTS {}".format(self.player)


def softmax(x):
    '''softmax'''
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs
