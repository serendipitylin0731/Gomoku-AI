#!/usr/bin/env bash
# 5x5 四子棋 block=2 继续训练脚本
# - 从 model/5_5_4_optimized/current_policy.model 恢复
# - 总共 600 个 batch
# - 每 30 个 batch 评估一次
# - 4 进程并行自弈，每轮共 20 局

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

CONDA_ENV="gomoku"
PYTHON="conda run -n ${CONDA_ENV} --no-capture-output python"

# 优先从之前 block=2 的 current_policy 恢复
INIT_MODEL=""
if [ -f "model/5_5_4_optimized/current_policy.model" ]; then
    INIT_MODEL="model/5_5_4_optimized/current_policy.model"
    echo "Resume from ${INIT_MODEL}"
elif [ -f "model/5_5_4_optimized/best_policy.model" ]; then
    INIT_MODEL="model/5_5_4_optimized/best_policy.model"
    echo "Resume from ${INIT_MODEL}"
else
    echo "No block=2 checkpoint found, train from scratch"
fi

INIT_ARG=""
if [ -n "${INIT_MODEL}" ]; then
    INIT_ARG="--init-model ${INIT_MODEL}"
fi

${PYTHON} train_5_5_4.py \
    --block 2 \
    --n-playout 400 \
    --self-play-n-playout 400 \
    --eval-n-playout 400 \
    --eval-n-games 40 \
    --lr 2e-4 \
    --batch-size 2048 \
    --buffer-size 50000 \
    --game-batch-num 600 \
    --check-freq 30 \
    --eval-freq 30 \
    --pure-mcts-playout-num 150 \
    --opponent-threshold 0.85 \
    --opponent-step 50 \
    --opponent-cap 2000 \
    --first-n-moves 10 \
    --c-puct 5.0 \
    --compile \
    --amp \
    --self-play-workers 4 \
    --games-per-worker 5 \
    --save-dir model/5_5_4_block2_600 \
    ${INIT_ARG} \
    --cuda
