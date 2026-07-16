# -*- coding: utf-8 -*-
"""
5x5 四子棋 AlphaZero 训练入口（Option 2）

用法示例：
    # 推荐优化训练（从当前 baseline 继续）
    python train_5_5_4.py \
        --block 2 \
        --n-playout 200 \
        --self-play-n-playout 200 \
        --eval-n-playout 400 \
        --eval-n-games 40 \
        --lr 1e-3 \
        --lr-milestones 2000 4000 6000 8000 \
        --lr-decay 0.5 \
        --batch-size 256 \
        --buffer-size 50000 \
        --game-batch-num 10000 \
        --check-freq 100 \
        --pure-mcts-playout-num 50 \
        --opponent-threshold 0.9 \
        --opponent-step 100 \
        --opponent-cap 2000 \
        --save-dir model/5_5_4_optimized \
        --init-model model/5_5_4_baseline/best_policy.model \
        --cuda

    # 也可以直接运行脚本：
    bash scripts/train_5_5_4_optimized.sh

优化要点（对比 baseline）：
1. 自弈 MCTS 模拟次数从 50 提升到 200，生成更高质量训练数据。
2. 评估时 AZ 使用 400 playouts，对手从 50 开始、按 0.9 胜率阈值逐步提升到 2000。
3. 训练 batch size 128 -> 256，buffer 10000 -> 50000，总 batch 5000 -> 10000。
4. 学习率阶梯衰减（默认 2000/4000/6000/8000 处乘以 0.5）。
5. 评估/对战时选择访问次数最多的动作（temp -> 0 的严格 argmax），减少随机失误。
"""

from __future__ import print_function
import argparse
import builtins
import contextlib
import io
import multiprocessing as mp
import random
import numpy as np
import os
import time
import csv
from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor
from Board.game_board import Board, Game
from AlphaZero.mcts_pure import MCTSPlayer as MCTS_Pure
from AlphaZero.mcts_alphaZero import MCTSPlayer
from AlphaZero.policy_value_net_torch import PolicyValueNet


# 给所有 print 加上从程序启动到当前的累计耗时，和原来的 now time 一致
_original_print = builtins.print
_start_time = time.time()

def ts_print(*args, **kwargs):
    elapsed_min = (time.time() - _start_time) / 60.0
    sep = kwargs.pop('sep', ' ')
    msg = sep.join(str(a) for a in args)
    msg = '{} [elapsed {:.2f}min]'.format(msg, elapsed_min)
    _original_print(msg, **kwargs)

builtins.print = ts_print


def selfplay_worker(config):
    '''多进程自弈工作进程：加载模型、玩若干局、返回原始数据'''
    import torch
    os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
    torch.set_num_threads(1)

    from Board.game_board import Board, Game
    from AlphaZero.mcts_alphaZero import MCTSPlayer
    from AlphaZero.policy_value_net_torch import PolicyValueNet

    board = Board(width=config['width'],
                  height=config['height'],
                  n_in_row=config['n_in_row'])
    game = Game(board)

    # 抑制子进程里的初始化打印
    with contextlib.redirect_stdout(io.StringIO()):
        policy = PolicyValueNet(
            config['width'], config['height'],
            block=config['block'],
            init_model=config['model_path'],
            cuda=config['cuda'],
            compile_model=False,
            use_amp=config.get('amp', False)
        )
        player = MCTSPlayer(
            policy_value_function=policy.policy_value_fn_random,
            action_fc=policy.action_fc_test,
            evaluation_fc=policy.evaluation_fc2_test,
            c_puct=config['c_puct'],
            n_playout=config['n_playout'],
            is_selfplay=True
        )
        player.first_n_moves = config['first_n_moves']

    data = []
    episode_lens = []
    for _ in range(config['n_games']):
        winner, play_data = game.start_self_play(player, is_shown=False)
        play_data = list(play_data)[:]
        episode_lens.append(len(play_data))
        for state, mcts_prob, winner_val in play_data:
            data.append((state, mcts_prob, winner_val))
    return data, episode_lens


class TrainPipeline():
    def __init__(self, args):
        # 5x5 四子棋棋盘参数
        self.resnet_block = args.block
        self.board_width = 5
        self.board_height = 5
        self.n_in_row = 4
        self.board = Board(width=self.board_width,
                           height=self.board_height,
                           n_in_row=self.n_in_row)
        self.game = Game(self.board)

        # 训练超参
        self.learn_rate = args.lr
        self.lr_decay = args.lr_decay
        self.lr_milestones = set(args.lr_milestones)
        self.n_playout = args.n_playout
        self.self_play_n_playout = args.self_play_n_playout or args.n_playout
        self.eval_n_playout = args.eval_n_playout
        self.eval_n_games = args.eval_n_games
        self.c_puct = args.c_puct
        self.skip_tie = args.skip_tie
        self.buffer_size = args.buffer_size
        self.batch_size = args.batch_size
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.self_play_workers = args.self_play_workers
        self.games_per_worker = args.games_per_worker
        self.play_batch_size = (self.self_play_workers * self.games_per_worker
                                if self.self_play_workers > 1 else 1)
        self.check_freq = args.check_freq
        self.eval_freq = args.eval_freq if args.eval_freq is not None else args.check_freq * 2
        self.game_batch_num = args.game_batch_num
        self.best_win_ratio = 0.0
        self.pure_mcts_playout_num = args.pure_mcts_playout_num
        self.opponent_threshold = args.opponent_threshold
        self.opponent_step = args.opponent_step
        self.opponent_cap = args.opponent_cap
        self.first_n_moves = args.first_n_moves

        # 保存路径
        self.save_dir = args.save_dir
        self.tmp_dir = os.path.join(self.save_dir, 'tmp')
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.tmp_dir, exist_ok=True)

        # CSV 日志路径
        self.log_csv = os.path.join(self.save_dir, 'training_log.csv')
        self._init_csv()

        # 初始化网络
        init_model = args.init_model
        if init_model is not None and not os.path.exists(init_model):
            print(f'Warning: init model {init_model} not found, train from scratch.')
            init_model = None

        self.policy_value_net = PolicyValueNet(
            self.board_width, self.board_height,
            block=self.resnet_block,
            init_model=init_model,
            cuda=args.cuda,
            compile_model=args.compile,
            use_amp=args.amp
        )

        # 多进程自弈池（如启用）
        self._pool = None
        if self.self_play_workers > 1:
            try:
                mp.set_start_method('spawn', force=True)
            except RuntimeError:
                pass
            self._pool = ProcessPoolExecutor(max_workers=self.self_play_workers)
            print('Using {} self-play workers, {} games per worker per batch'.format(
                self.self_play_workers, self.games_per_worker))

        self.mcts_player = MCTSPlayer(
            policy_value_function=self.policy_value_net.policy_value_fn_random,
            action_fc=self.policy_value_net.action_fc_test,
            evaluation_fc=self.policy_value_net.evaluation_fc2_test,
            c_puct=self.c_puct,
            n_playout=self.self_play_n_playout,
            is_selfplay=True
        )
        self.mcts_player.first_n_moves = self.first_n_moves

    def _init_csv(self):
        """初始化 CSV 日志文件"""
        if not os.path.exists(self.log_csv):
            with open(self.log_csv, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'batch', 'episode_len', 'loss', 'entropy', 'kl',
                    'explained_var_old', 'explained_var_new',
                    'win_ratio', 'pure_mcts_playout_num', 'learn_rate',
                    'collect_data_time_s', 'train_data_time_s', 'evaluate_time_s'
                ])

    def _log_csv(self, row):
        """追加一行训练指标到 CSV"""
        with open(self.log_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)

    def get_equi_data(self, play_data):
        '''通过旋转和翻转增强数据'''
        extend_data = []
        for state, mcts_porb, winner in play_data:
            for i in [1, 2, 3, 4]:
                equi_state = np.array([np.rot90(s, i) for s in state])
                equi_mcts_prob = np.rot90(np.flipud(
                    mcts_porb.reshape(self.board_height, self.board_width)), i)
                extend_data.append((equi_state,
                                    np.flipud(equi_mcts_prob).flatten(),
                                    winner))
                equi_state = np.array([np.fliplr(s) for s in equi_state])
                equi_mcts_prob = np.fliplr(equi_mcts_prob)
                extend_data.append((equi_state,
                                    np.flipud(equi_mcts_prob).flatten(),
                                    winner))
        return extend_data

    def collect_selfplay_data(self, n_games=1):
        '''收集自弈数据（支持多进程并行）'''
        if self.self_play_workers <= 1:
            # 单进程模式
            for i in range(n_games):
                winner, play_data = self.game.start_self_play(self.mcts_player, is_shown=False)
                play_data = list(play_data)[:]
                self.episode_len = len(play_data)
                # 若开启 skip_tie，跳过平局对局
                if self.skip_tie and winner == -1:
                    print('skip tie game, episode_len: {}'.format(self.episode_len))
                    continue
                play_data = self.get_equi_data(play_data)
                self.data_buffer.extend(play_data)
        else:
            # 多进程模式：每个 worker 加载当前模型，并行玩若干局
            worker_model_path = os.path.join(self.tmp_dir, 'worker_policy.model')
            self.policy_value_net.save_model(worker_model_path)

            base = n_games // self.self_play_workers
            rem = n_games % self.self_play_workers
            cuda = self.policy_value_net.device.type == 'cuda'
            configs = []
            for i in range(self.self_play_workers):
                configs.append({
                    'width': self.board_width,
                    'height': self.board_height,
                    'n_in_row': self.n_in_row,
                    'block': self.resnet_block,
                    'model_path': worker_model_path,
                    'cuda': cuda,
                    'amp': getattr(self.policy_value_net, 'use_amp', False),
                    'c_puct': self.c_puct,
                    'n_playout': self.self_play_n_playout,
                    'first_n_moves': self.first_n_moves,
                    'n_games': base + (1 if i < rem else 0),
                })

            futures = [self._pool.submit(selfplay_worker, cfg) for cfg in configs]
            raw_data = []
            episode_lens = []
            for f in futures:
                d, lens = f.result()
                raw_data.extend(d)
                episode_lens.extend(lens)

            self.episode_len = int(np.mean(episode_lens)) if episode_lens else 0
            print('collected {} games, avg episode_len: {}'.format(
                len(episode_lens), self.episode_len))

            if self.skip_tie:
                # 多进程模式下不保留平局：worker 已返回全部对局，这里过滤
                raw_data = [(s, p, z) for s, p, z in raw_data if z != 0]

            extended_data = self.get_equi_data(raw_data)
            self.data_buffer.extend(extended_data)

    def policy_update(self):
        '''更新策略价值网络'''
        tmp_buffer = list(self.data_buffer)
        random.shuffle(tmp_buffer)
        steps = len(tmp_buffer) // self.batch_size
        print('tmp buffer: {}, steps: {}'.format(len(tmp_buffer), steps))

        last_metrics = None
        for i in range(steps):
            mini_batch = tmp_buffer[i * self.batch_size:(i + 1) * self.batch_size]
            state_batch = [data[0] for data in mini_batch]
            mcts_probs_batch = [data[1] for data in mini_batch]
            winner_batch = [data[2] for data in mini_batch]

            old_probs, old_v = self.policy_value_net.policy_value(
                state_batch=state_batch,
                actin_fc=self.policy_value_net.action_fc_test,
                evaluation_fc=self.policy_value_net.evaluation_fc2_test
            )
            loss, entropy = self.policy_value_net.train_step(
                state_batch, mcts_probs_batch, winner_batch, self.learn_rate
            )
            new_probs, new_v = self.policy_value_net.policy_value(
                state_batch=state_batch,
                actin_fc=self.policy_value_net.action_fc_test,
                evaluation_fc=self.policy_value_net.evaluation_fc2_test
            )
            kl = np.mean(np.sum(old_probs * (
                    np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)),
                    axis=1)
            )

            explained_var_old = (1 -
                                 np.var(np.array(winner_batch) - old_v.flatten()) /
                                 np.var(np.array(winner_batch)))
            explained_var_new = (1 -
                                 np.var(np.array(winner_batch) - new_v.flatten()) /
                                 np.var(np.array(winner_batch)))

            if steps < 10 or (i % (steps // 10) == 0):
                print('batch: {}, length: {} '
                      'kl:{:.5f}, '
                      'loss:{}, '
                      'entropy:{}, '
                      'explained_var_old:{:.3f}, '
                      'explained_var_new:{:.3f}, '
                      'lr:{:.6f}'.format(
                          i, len(mini_batch), kl, loss, entropy,
                          explained_var_old, explained_var_new, self.learn_rate))

            last_metrics = (loss, entropy, kl, explained_var_old, explained_var_new)

        return last_metrics

    def policy_evaluate(self, n_games=10):
        '''与纯 MCTS 对战评估当前策略'''
        current_mcts_player = MCTSPlayer(
            policy_value_function=self.policy_value_net.policy_value_fn_random,
            action_fc=self.policy_value_net.action_fc_test,
            evaluation_fc=self.policy_value_net.evaluation_fc2_test,
            c_puct=self.c_puct,
            n_playout=self.eval_n_playout,
            is_selfplay=False
        )

        test_player = MCTS_Pure(c_puct=self.c_puct,
                                n_playout=self.pure_mcts_playout_num)

        win_cnt = defaultdict(int)
        for i in range(n_games):
            winner = self.game.start_play(
                player1=current_mcts_player,
                player2=test_player,
                start_player=i % 2,
                is_shown=0,
                print_prob=False
            )
            win_cnt[winner] += 1
        win_ratio = 1.0 * (win_cnt[1] + 0.5 * win_cnt[-1]) / n_games
        print("num_playouts:{}, win: {}, lose: {}, tie:{}".format(
            self.pure_mcts_playout_num,
            win_cnt[1], win_cnt[2], win_cnt[-1]))
        return win_ratio

    def run(self):
        '''运行训练流程'''
        start_time = time.time()
        collect_data_time = 0
        train_data_time = 0
        evaluate_time = 0

        try:
            for i in range(self.game_batch_num):
                # 学习率衰减
                if (i + 1) in self.lr_milestones:
                    self.learn_rate *= self.lr_decay
                    print(f"LR decay at batch {i + 1}: learn_rate = {self.learn_rate:.6f}")

                # 收集自弈数据
                collect_data_start_time = time.time()
                self.collect_selfplay_data(self.play_batch_size)
                collect_data_time += time.time() - collect_data_start_time
                print("batch i:{}, episode_len:{}".format(
                    i + 1, self.episode_len))

                metrics = (None, None, None, None, None)
                if len(self.data_buffer) > self.batch_size * 2:
                    # 训练网络
                    train_data_start_time = time.time()
                    metrics = self.policy_update()
                    train_data_time += time.time() - train_data_start_time

                    print('now time : {}'.format((time.time() - start_time) / 3600))
                    print('collect_data_time : {}, train_data_time : {}, evaluate_time : {}'.format(
                        collect_data_time / 3600, train_data_time / 3600, evaluate_time / 3600))

                if (i + 1) % self.check_freq == 0:
                    # 保存当前模型
                    current_path = os.path.join(self.tmp_dir, 'current_policy.model')
                    self.policy_value_net.save_model(current_path)

                if (i + 1) % self.eval_freq == 0:
                    print("current self-play batch: {}".format(i + 1))
                    evaluate_start_time = time.time()
                    win_ratio = self.policy_evaluate(n_games=self.eval_n_games)
                    evaluate_time += time.time() - evaluate_start_time

                    # 记录 CSV
                    self._log_csv([
                        i + 1, self.episode_len,
                        metrics[0] if metrics[0] is not None else '',
                        metrics[1] if metrics[1] is not None else '',
                        metrics[2] if metrics[2] is not None else '',
                        metrics[3] if metrics[3] is not None else '',
                        metrics[4] if metrics[4] is not None else '',
                        win_ratio, self.pure_mcts_playout_num, self.learn_rate,
                        collect_data_time, train_data_time, evaluate_time
                    ])

                    if win_ratio > self.best_win_ratio:
                        print("New best policy!!!!!!!!")
                        self.best_win_ratio = win_ratio
                        best_path = os.path.join(self.save_dir, 'best_policy.model')
                        self.policy_value_net.save_model(best_path)

                    # 当胜率超过阈值时提升对手强度，推动模型继续进化
                    if (win_ratio >= self.opponent_threshold and
                            self.pure_mcts_playout_num < self.opponent_cap):
                        self.pure_mcts_playout_num += self.opponent_step
                        self.best_win_ratio = 0.0
                        print(f"Increase pure_mcts_playout_num to {self.pure_mcts_playout_num}")

        except KeyboardInterrupt:
            print('\n\rquit')
        finally:
            # 退出前保存当前模型，防止中断丢失进度
            current_path = os.path.join(self.tmp_dir, 'current_policy.model')
            self.policy_value_net.save_model(current_path)
            print(f"Saved current model to {current_path}")
            # 关闭多进程池
            if self._pool is not None:
                self._pool.shutdown(wait=True)
                print('Self-play worker pool shutdown')


def parse_args():
    parser = argparse.ArgumentParser(description='5x5 四子棋 AlphaZero 训练')
    parser.add_argument('--block', type=int, default=2,
                        help='ResNet 残差块数量 (默认 2)')
    parser.add_argument('--n-playout', type=int, default=200,
                        help='默认 MCTS 模拟次数 (默认 200)')
    parser.add_argument('--self-play-n-playout', type=int, default=None,
                        help='自弈时 MCTS 模拟次数 (默认与 --n-playout 相同)')
    parser.add_argument('--eval-n-playout', type=int, default=400,
                        help='评估时 AlphaZero MCTS 模拟次数 (默认 400)')
    parser.add_argument('--eval-n-games', type=int, default=40,
                        help='每次评估与纯 MCTS 对战局数 (默认 40)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='学习率 (默认 1e-3)')
    parser.add_argument('--lr-milestones', type=int, nargs='+',
                        default=[2000, 4000, 6000, 8000],
                        help='学习率衰减里程碑 (默认 2000 4000 6000 8000)')
    parser.add_argument('--lr-decay', type=float, default=0.5,
                        help='学习率衰减系数 (默认 0.5)')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='训练 batch size (默认 256)')
    parser.add_argument('--buffer-size', type=int, default=50000,
                        help='经验回放缓冲区大小 (默认 50000)')
    parser.add_argument('--game-batch-num', type=int, default=10000,
                        help='总训练局数 (默认 10000)')
    parser.add_argument('--check-freq', type=int, default=100,
                        help='每隔多少局保存一次模型 (默认 100)')
    parser.add_argument('--eval-freq', type=int, default=None,
                        help='每隔多少局评估一次（默认是 --check-freq 的两倍）')
    parser.add_argument('--pure-mcts-playout-num', type=int, default=50,
                        help='评估对手纯 MCTS 起始模拟次数 (默认 50)')
    parser.add_argument('--opponent-threshold', type=float, default=0.9,
                        help='胜率达到该阈值后提升对手模拟次数 (默认 0.9)')
    parser.add_argument('--opponent-step', type=int, default=100,
                        help='每次提升对手模拟次数的步长 (默认 100)')
    parser.add_argument('--opponent-cap', type=int, default=2000,
                        help='对手模拟次数上限 (默认 2000)')
    parser.add_argument('--first-n-moves', type=int, default=4,
                        help='前 n 步使用 temp=1 探索 (默认 4)')
    parser.add_argument('--c-puct', type=float, default=5.0,
                        help='MCTS UCT 探索常数 (默认 5.0)')
    parser.add_argument('--compile', action='store_true', default=False,
                        help='使用 torch.compile 加速推理（需要 PyTorch 2.0+，默认关闭）')
    parser.add_argument('--amp', action='store_true', default=False,
                        help='使用自动混合精度 AMP 加速训练和推理（默认关闭）')
    parser.add_argument('--skip-tie', action='store_true', default=False,
                        help='自弈时跳过平局对局（默认不跳过）')
    parser.add_argument('--self-play-workers', type=int, default=1,
                        help='自弈并行进程数（默认 1，>1 时启用多进程自弈）')
    parser.add_argument('--games-per-worker', type=int, default=5,
                        help='每个 worker 每轮自弈局数（默认 5，仅多进程时生效）')
    parser.add_argument('--save-dir', type=str, default='model/5_5_4_optimized',
                        help='模型与日志保存目录')
    parser.add_argument('--init-model', type=str, default=None,
                        help='从已有模型继续训练（可选）')
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='使用 GPU（默认开启）')
    parser.add_argument('--no-cuda', dest='cuda', action='store_false',
                        help='禁用 GPU')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    training_pipeline = TrainPipeline(args)
    training_pipeline.run()
