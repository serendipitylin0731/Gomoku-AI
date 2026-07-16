# -*- coding: utf-8 -*-
"""
5x5 四子棋 AlphaZero 模型对战评估脚本（Option 2）

用法示例：
    python evaluate_5_5_4.py --model model/5_5_4_baseline/best_policy.model --block 2 --n-playout 400 --opponent-playouts 50 100 200
"""

from __future__ import print_function
import argparse
import numpy as np
from collections import defaultdict
from Board.game_board import Board, Game
from AlphaZero.mcts_pure import MCTSPlayer as MCTS_Pure
from AlphaZero.mcts_alphaZero import MCTSPlayer
from AlphaZero.policy_value_net_torch import PolicyValueNet


def evaluate(model_path, block, n_playout, opponent_playouts, n_games, cuda):
    board = Board(width=5, height=5, n_in_row=4)
    game = Game(board)

    policy = PolicyValueNet(5, 5, block=block, init_model=model_path, cuda=cuda)
    alpha_zero_player = MCTSPlayer(
        policy_value_function=policy.policy_value_fn_random,
        action_fc=policy.action_fc_test,
        evaluation_fc=policy.evaluation_fc2_test,
        c_puct=5,
        n_playout=n_playout,
        is_selfplay=False
    )

    results = {}
    for opp_playout in opponent_playouts:
        pure = MCTS_Pure(c_puct=5, n_playout=opp_playout)
        win_cnt = defaultdict(int)
        for i in range(n_games):
            winner = game.start_play(
                player1=alpha_zero_player,
                player2=pure,
                start_player=i % 2,
                is_shown=0,
                print_prob=False
            )
            win_cnt[winner] += 1
        win_ratio = 1.0 * (win_cnt[1] + 0.5 * win_cnt[-1]) / n_games
        results[opp_playout] = {
            'win': win_cnt[1],
            'lose': win_cnt[2],
            'tie': win_cnt[-1],
            'win_ratio': win_ratio
        }
        print("opponent_playout={}: win={}, lose={}, tie={}, win_ratio={:.3f}".format(
            opp_playout, win_cnt[1], win_cnt[2], win_cnt[-1], win_ratio))

    return results


def parse_args():
    parser = argparse.ArgumentParser(description='5x5 四子棋模型评估')
    parser.add_argument('--model', type=str, required=True,
                        help='待评估模型路径')
    parser.add_argument('--block', type=int, default=2,
                        help='模型 ResNet 块数')
    parser.add_argument('--n-playout', type=int, default=400,
                        help='AlphaZero 评估时 MCTS 模拟次数 (默认 400)')
    parser.add_argument('--opponent-playouts', type=int, nargs='+',
                        default=[50, 100, 200, 400],
                        help='纯 MCTS 对手模拟次数列表 (默认 50 100 200 400)')
    parser.add_argument('--n-games', type=int, default=20,
                        help='每个对手对战局数 (默认 20)')
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='使用 GPU（默认开启）')
    parser.add_argument('--no-cuda', dest='cuda', action='store_false',
                        help='禁用 GPU')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    evaluate(args.model, args.block, args.n_playout,
             args.opponent_playouts, args.n_games, args.cuda)
