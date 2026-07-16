"""
Created on Sat Dec  8 13:02:14 2018

@author: initial-h

Modified on Tue May 13 22:06:14 2025

@author: RPChe_
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# This is the policy value network model for AlphaZero.
class PolicyValueNetModel(nn.Module):
    def __init__(self, board_width, board_height, block, planes_num):
        super().__init__()
        self.planes_num = planes_num
        self.nb_block = block
        self.board_width = board_width
        self.board_height = board_height

        # 公共网络层
        self.zeropad2d = nn.ZeroPad2d(2)
        self.conv2d_1 = nn.Conv2d(planes_num, 64, kernel_size=1, stride=1, padding=0)

        # 残差块
        self.resnet = nn.ModuleList()
        for i in range(block):
            layer = nn.Sequential()
            layer.add_module('0', nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1))
            layer.add_module('1', nn.BatchNorm2d(64))
            layer.add_module('2', nn.ReLU())
            layer.add_module('3', nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1))
            layer.add_module('4', nn.BatchNorm2d(64))
            self.resnet.append(layer)

        # 策略头
        self.conv2d_2 = nn.Conv2d(64, 2, kernel_size=1, stride=1, padding=0)
        self.bn_1 = nn.BatchNorm2d(2)
        self.dense_layer_1 = nn.Linear(
            (board_width + 4) * (board_height + 4) * 2,
            board_width * board_height
        )

        # 价值头
        self.conv2d_3 = nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0)
        self.bn_2 = nn.BatchNorm2d(1)
        self.dense_layer_2 = nn.Linear(
            (board_width + 4) * (board_height + 4),
            256
        )
        self.flatten_layer_3 = nn.Linear(256, 1)

    def forward(self, x):
        '''
        The output of the network is log probabilities for actions and a value for the state.
        x: [B, planes_num, board_height, board_width]
        '''
        # 公共网络
        out = self.zeropad2d(x)
        out = self.conv2d_1(out)

        # 残差块
        for block in self.resnet:
            identity = out
            tmp = block(out)
            out = F.relu(tmp + identity)

        # 策略头
        act = self.conv2d_2(out)
        act = F.relu(self.bn_1(act))
        act = act.permute(0, 2, 3, 1).contiguous()
        act = act.view(act.size(0), -1)
        act = F.log_softmax(self.dense_layer_1(act), dim=1)

        # 价值头
        val = self.conv2d_3(out)
        val = F.relu(self.bn_2(val))
        val = val.view(val.size(0), -1)
        val = F.relu(self.dense_layer_2(val))
        val = torch.tanh(self.flatten_layer_3(val))

        return act, val


# This is a wrapper class for the PolicyValueNetModel.
class PolicyValueNet():
    def __init__(self, board_width, board_height, block, init_model=None, transfer_model=None, cuda=False, compile_model=False, use_amp=False):
        print()
        print('building network ...')
        print()

        # Save hyperparameters
        self.planes_num = 9  # feature planes
        self.nb_block = block # resnet blocks
        self.device = torch.device("cuda" if cuda and torch.cuda.is_available() else "cpu")
        self.board_width = board_width
        self.board_height = board_height
        # Used as macros
        # This part is set to fit the api in tensorflow version.
        # but is not used in this Pytorch version.
        # Actually, only action_fc_test and evaluation_fc2_test are used.
        self.action_fc_train = 0
        self.evaluation_fc2_train = 1
        self.action_fc_test = 2
        self.evaluation_fc2_test = 3

        # Initialize model
        self.model = PolicyValueNetModel(self.board_width, self.board_height, self.nb_block, self.planes_num)
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        # lr is set during training before, but Pytorch does not support this. So if you want to configure the learning rate, you have to do it here.
        self.oppo = PolicyValueNetModel(board_width, board_height, block, self.planes_num)
        self.oppo.to(self.device)

        # Load model if specified
        if init_model is not None:
            self.restore_model(init_model)
            print('model loaded!')
        elif transfer_model is not None:
            self.restore_model(transfer_model)
            print('transfer model loaded!')
        else:
            print('can not find saved model, learn from scratch !')

        # 可选：使用 torch.compile 加速推理（需要 PyTorch 2.0+）
        if compile_model:
            try:
                self.model = torch.compile(self.model)
                print('torch.compile enabled for PolicyValueNetModel')
            except Exception as e:
                print(f'Warning: torch.compile failed: {e}')

        # 可选：使用自动混合精度（AMP）加速训练，仅在 CUDA 上有效
        self.use_amp = use_amp and self.device.type == 'cuda'
        self.scaler = torch.amp.GradScaler('cuda') if self.use_amp else None
        if self.use_amp:
            print('Automatic Mixed Precision (AMP) enabled for training')

    # def save_numpy(self, params):
    # def load_numpy(self, params, path='tmp/model.npy'):

    # def print_params(self, params):
    def print_params(self):
        for name, param in self.model.named_parameters():
            print(f"name: {name}, shape: {param.data.shape}")
        for name, param in self.model.named_buffers():
            print(f"buffer name: {name}, shape: {param.data.shape}")

    def policy_value(self, state_batch, actin_fc, evaluation_fc):
        '''
        input: a batch of states, actin_fc, evaluation_fc
        output: a batch of action probabilities and state values
        '''
        assert actin_fc == self.action_fc_test, "policy_value: action_fc not match"
        assert evaluation_fc == self.evaluation_fc2_test, "policy_value: evaluation_fc not match"
        # As I said, actin_fc and evaluation_fc are not used in this Pytorch version.

        self.model.eval()
        state_batch = np.asarray(state_batch)
        state_batch = torch.from_numpy(state_batch).to(self.device).float()
        with torch.no_grad(), torch.autocast(device_type=self.device.type, enabled=self.use_amp):
            log_act_probs, value = self.model(state_batch)
        act_probs = np.exp(log_act_probs.cpu().numpy())
        return act_probs, value.cpu().numpy()

    def policy_value_fn(self, board, actin_fc, evaluation_fc):
        '''
        input: board,actin_fc,evaluation_fc
        output: a list of (action, probability) tuples for each available
        action and the score of the board state
        '''
        # the accurate policy value fn,
        # i prefer to use one that has some randomness even when test,
        # so that each game can play some different moves, all are ok here
        legal_positions = board.availables
        current_state = np.ascontiguousarray(board.current_state().reshape(-1, self.planes_num, self.board_width, self.board_height))
        act_probs, value = self.policy_value(current_state, actin_fc, evaluation_fc)
        act_probs = zip(legal_positions, act_probs[0][legal_positions])
        return act_probs, value

    def policy_value_fn_random(self, board, actin_fc, evaluation_fc):
        '''
        input: board,actin_fc,evaluation_fc
        output: a list of (action, probability) tuples for each available
        action and the score of the board state
        '''
        # like paper said,
        # The leaf node sL is added to a queue for neural network
        # evaluation, (di(p), v) = fθ(di(sL)),
        # where di is a dihedral reflection or rotation
        # selected uniformly at random from i in [1..8]

        legal_positions = board.availables
        current_state = np.ascontiguousarray(board.current_state().reshape(-1, self.planes_num, self.board_width, self.board_height))

        # print('current state shape',current_state.shape)

        #add dihedral reflection or rotation
        rotate_angle = np.random.randint(1, 5)
        flip = np.random.randint(0, 2)
        equi_state = np.array([np.rot90(s, rotate_angle) for s in current_state[0]])
        if flip:
            equi_state = np.array([np.fliplr(s) for s in equi_state])
        # print(equi_state.shape)

        # put equi_state to network
        act_probs, value = self.policy_value(np.array([equi_state]), actin_fc, evaluation_fc)

        # get dihedral reflection or rotation back
        equi_mcts_prob = np.flipud(act_probs[0].reshape(self.board_height, self.board_width))
        if flip:
            equi_mcts_prob = np.fliplr(equi_mcts_prob)
        equi_mcts_prob = np.rot90(equi_mcts_prob, 4 - rotate_angle)
        act_probs = np.flipud(equi_mcts_prob).flatten()

        act_probs = zip(legal_positions, act_probs[legal_positions])
        return act_probs, value

    def train_step(self, state_batch, mcts_probs, winner_batch, lr):
        '''
        perform a training step
        state_batch: NCHW, mcts_probs: NHW, winner_batch: N
        '''
        self.model.train()

        state_batch = np.asarray(state_batch)
        mcts_probs = np.asarray(mcts_probs)
        winner_batch = np.asarray(winner_batch)

        state_batch = torch.from_numpy(state_batch).to(self.device).float()
        mcts_probs = torch.from_numpy(mcts_probs).to(self.device).float()
        winner_batch = torch.from_numpy(winner_batch).to(self.device).float()

        # 清空梯度
        self.optimizer.zero_grad()
        # 设置学习率
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        # 前向传播
        log_act_probs, value = self.model(state_batch)

        # 价值损失
        value_loss = F.mse_loss(value.view(-1), winner_batch)
        # 策略损失
        policy_loss = -torch.mean(torch.sum(mcts_probs * log_act_probs, dim=1))
        # L2 正则
        l2_penalty = 0
        for param in self.model.parameters():
            if param.dim() > 1:  # 偏置不参与
                l2_penalty += torch.sum(param ** 2)
        l2_penalty = 1e-4 / 2.0 * l2_penalty

        loss = value_loss + policy_loss + l2_penalty
        loss.backward()
        self.optimizer.step()

        # 计算策略熵用于监控
        entropy = -torch.mean(torch.sum(torch.exp(log_act_probs) * log_act_probs, dim=1))

        return loss.item(), entropy.item()

    def _unwrap_model(self):
        '''如果模型被 torch.compile 包装，返回原始模块'''
        return self.model._orig_mod if hasattr(self.model, '_orig_mod') else self.model

    def save_model(self, model_path):
        '''
        save model to model_path
        '''
        torch.save(self._unwrap_model().state_dict(), model_path)

    def restore_model(self, model_path):
        '''
        restore model from model_path
        '''
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path {model_path} does not exist.")
        self._unwrap_model().load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
