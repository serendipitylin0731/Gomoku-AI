# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 22:05:17 2018

@author: initial
"""


import numpy as np


def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs


class TreeNode(object):
    '''
    A node in the MCTS tree.
    Each node keeps track of its own value Q, prior probability P, and
    its visit-count-adjusted prior score u.
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
    An implementation of Monte Carlo Tree Search.
    '''
    # action_fc and evaluation_fc are not used in Pytorch version, just placeholders. see details in policy_value_net_pytorch.py
    def __init__(self, policy_value_fn, action_fc, evaluation_fc, is_selfplay, c_puct=5, n_playout=400):
        '''
        policy_value_fn: a function that takes in a board state and outputs
            a list of (action, probability) tuples and also a score in [-1, 1]
            (i.e. the expected value of the end game score from the current
            player's perspective) for the current player.
        c_puct: a number in (0, inf) that controls how quickly exploration
            converges to the maximum-value policy. A higher value means
            relying on the prior more.
        '''
        self._root = TreeNode(None, 1.0)
        # root node do not have parent ,and sure with prior probability 1

        self._policy_value_fn = policy_value_fn
        self._action_fc = action_fc
        self._evaluation_fc = evaluation_fc

        self._c_puct = c_puct
        # it's 5 in paper and don't change here,but maybe a better number exists in gomoku domain
        self._n_playout = n_playout  # times of tree search
        self._is_selfplay = is_selfplay

    def _playout(self, state):
        '''执行一次神经网络引导的模拟'''
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
            # 未终局则用神经网络扩展
            action_probs, leaf_value = self._policy_value_fn(
                state, self._action_fc, self._evaluation_fc
            )
            node.expand(action_probs)
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
    def __init__(self, policy_value_function, action_fc, evaluation_fc, c_puct=5, n_playout=400, is_selfplay=0):
        '''
        init some parameters
        '''
        self._is_selfplay = is_selfplay
        self.policy_value_function = policy_value_function
        self.action_fc = action_fc
        self.evaluation_fc = evaluation_fc
        # self.first_n_moves = 12
        self.first_n_moves = 4
        # For the first n moves of each game, the temperature is set to τ = 1,
        # For the remainder of the game, an infinitesimal temperature is used, τ→ 0.
        # in paper n=30, here i choose 12 for 11x11, entirely by feel
        self.mcts = MCTS(policy_value_fn=policy_value_function,
                         action_fc=action_fc,
                         evaluation_fc=evaluation_fc,
                         is_selfplay=self._is_selfplay,
                         c_puct=c_puct,
                         n_playout=n_playout)

    def set_player_ind(self, p):
        '''
        set player index
        '''
        self.player = p

    def reset_player(self):
        '''
        reset player
        '''
        self.mcts.update_with_move(-1)

    def get_action(self, board, is_selfplay, print_probs_value):
        '''
        get an action by mcts
        do not discard all the tree and retain the useful part
        '''
        sensible_moves = board.availables
        # the pi vector returned by MCTS as in the alphaGo Zero paper
        move, move_probs = None, np.zeros(board.width * board.height)
        if len(sensible_moves) > 0:
            # 根据步数选择温度
            temp = 1.0 if is_selfplay and len(board.states) < self.first_n_moves else 1e-3
            acts, probs = self.mcts.get_move_probs(board, temp)
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
                # temp 趋于 0，按 Handout 要求选择访问次数最多的动作
                move = acts[np.argmax(probs)]
                # 重置根节点
                self.mcts.update_with_move(-1)

            if print_probs_value and move_probs is not None:
                act_probs, value = self.policy_value_function(board, self.action_fc, self.evaluation_fc)
                print('-' * 10)
                print('value', value)
                # print the probability of each move
                probs = np.array(move_probs).reshape((board.width, board.height)).round(3)[::-1, :]
                for p in probs:
                    for x in p:
                        print("{0:6}".format(x), end='')
                    print('\r')

            return move, move_probs

        else:
            print("WARNING: the board is full")

    def __str__(self):
        return "Alpha {}".format(self.player)
