# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 13:51:53 2018

@author: initial-h
"""

from __future__ import print_function
import argparse
from Board.game_board import Board, Game
from AlphaZero.mcts_pure import MCTSPlayer as MCTS_pure
from AlphaZero.mcts_alphaZero import MCTSPlayer
# from policy_value_net_tensorlayer import PolicyValueNet
from AlphaZero.policy_value_net_torch import PolicyValueNet
import time
from os import path
import os
from collections import defaultdict
from Minimax.search_wrapper import MinimaxPlayer

class Human(object):
    """
    human player
    """
    def __init__(self):
        self.player = None

    def set_player_ind(self, p):
        self.player = p

    def get_action(self, board,is_selfplay=False,print_probs_value=0):
        # no use params in the func : is_selfplay,print_probs_value
        # just to stay the same with AI's API
        try:
            location = input("Your move: ")
            if isinstance(location, str):  # for python3
                location = [int(n, 10) for n in location.split(",")]
            move = board.location_to_move(location)
        except Exception as e:
            move = -1
        if move == -1 or move not in board.availables:
            print("invalid move")
            move,_ = self.get_action(board)
        return move,None

    def __str__(self):
        return "Human {}".format(self.player)

def run(start_player=0, is_shown=1, width=3, height=3, n_in_row=3,
        model_file='./model/3_3_3.model', block=1, n_playout=400, cuda=True):
    # run a gomoku game with AI
    # you can set
    # human vs AI or AI vs AI
    p = os.getcwd()
    model_file = path.join(p, model_file)

    board = Board(width=width, height=height, n_in_row=n_in_row)
    game = Game(board)

    best_policy = PolicyValueNet(board_width=width, board_height=height,
                                 block=block, init_model=model_file, cuda=cuda)

    alpha_zero_player = MCTSPlayer(policy_value_function=best_policy.policy_value_fn_random,
                                   action_fc=best_policy.action_fc_test,
                                   evaluation_fc=best_policy.evaluation_fc2_test,
                                   c_puct=5,
                                   n_playout=n_playout,
                                   is_selfplay=False)

    # play in GUI
    game.start_play_with_UI(alpha_zero_player, start_player=start_player)


def parse_args():
    parser = argparse.ArgumentParser(description='与 AlphaZero 人机对战')
    parser.add_argument('--width', type=int, default=3,
                        help='棋盘宽度 (默认 3)')
    parser.add_argument('--height', type=int, default=3,
                        help='棋盘高度 (默认 3)')
    parser.add_argument('--n-in-row', type=int, default=3,
                        help='获胜连子数 (默认 3)')
    parser.add_argument('--model', type=str, default='./model/3_3_3.model',
                        help='模型路径')
    parser.add_argument('--block', type=int, default=1,
                        help='ResNet 块数 (默认 1)')
    parser.add_argument('--n-playout', type=int, default=400,
                        help='MCTS 模拟次数 (默认 400)')
    parser.add_argument('--start-player', type=int, default=0,
                        help='先手玩家 0=人类, 1=AI (默认 0)')
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='使用 GPU（默认开启）')
    parser.add_argument('--no-cuda', dest='cuda', action='store_false',
                        help='禁用 GPU')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run(start_player=args.start_player, is_shown=True,
        width=args.width, height=args.height, n_in_row=args.n_in_row,
        model_file=args.model, block=args.block, n_playout=args.n_playout,
        cuda=args.cuda)