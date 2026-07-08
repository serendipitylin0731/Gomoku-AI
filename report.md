# 五子棋 AI 课程项目报告

> 林函锋 | 525021910285 | serendipity_lin@sjtu.edu.cn

---

## 1. 本地运行环境

- **运行操作系统**：WSL2（Ubuntu 22.04）
- **CPU**：AMD Ryzen 9 9955HX （16 核 32 线程）
- **GPU**：NVIDIA RTX 4070 Laptop
- **Python 环境**：Conda，Python 3.10.20
- **C++ 编译器**：g++（GCC 11+）

---

## 2. Minimax Search AI

### 2.1 实现思路

Minimax Search AI 的初始实现效果较差，分析后发现，主要问题在于棋盘状态维护、候选集排序和搜索框架的效率与正确性不足。因此后续实现没有继续从头硬写，而是借鉴了 baseline 的基础代码，保留其核心设计，并针对开局搜索、置换表、时间控制和随机性等方面进行改进。

### 2.2 代码结构

当前 `AIController.cpp` 的主要模块如下：
```
AIController.cpp
├── Board + Status：增量棋盘与威胁状态维护
├── blank set：按威胁值排序的空点候选集
├── calc / Status::w：单点威胁评估
├── solve：Negamax + Alpha-Beta + TT 查询/存储
├── first_two_black / second / is_change：开局三手处理
└── turn：中盘迭代加深搜索
```

### 2.3 继承的 baseline 核心设计与改进

#### 以下设计直接沿用 baseline 的实现：

- **`Status` 增量状态维护**：对每个坐标、每个方向、每种颜色维护连续子数与两端是否为空。落子或提子时仅更新受影响区域，使任意空点的威胁值可在 `O(1)` 内查询。
- **`std::set<Blank>` 候选集**：空点按 `max(我方威胁, 对方威胁)` 降序排列，为搜索提供高质量的优先顺序。
- **Minimax + Alpha-Beta 搜索框架**：采用 `solve(color, depth, width, low, high)` 的结构与标准窗口传递。
- **评估权重表**：成五 / 活四 / 冲四 / 活三 / 眠三的层级权重设计。

#### 主要改进点：

#### （1）加深开局搜索

baseline 的 `first_two_black()` 与 `second()` 仅搜索 6 层，且遍历全部 225 个空点，搜索质量有限。改进后：

- 仅评估威胁最大的前 40 个候选（`top_empty(40)`）；
- 使用与中盘相同的完整深度 `DEPTH = 14` 评估每个候选。

此举显著提升了 swap 决策质量。改进前后的 64 局对比如下：

| 版本 | AIController 胜 | baseline 胜 | score |
|------|-----------------|-------------|-------|
| 早期自行实现 | 22 | 42 | -0.31 |
| baseline 原版 | - | - | 0 |
| 改进开局搜索后 | 49 | 15 | +0.53 |

#### （2）加入置换表

实现了 Zobrist 哈希与置换表（TT）。TT 大小为 `2^20` 项，存储 `[key | depth | score | flag | best_move]`，同 key 保留深度更大者。

在没有 TT 的情况下，14 层搜索在中盘容易超时；加入 TT 后，重复局面得以复用，大多数局面可在 4.4 秒内完成搜索。

#### （3）迭代加深与时间控制

中盘 `turn()` 不再固定搜索 14 层，而是采用迭代加深：从深度 2 开始，每次增加 2 层，每完成一层保留当前最佳着法。当时间接近 4.4 秒上限时，返回最近完成深度的结果，避免超时判负。

```cpp
Coordinate best = solve(0, 2, WIDTH).coor;
for (int d = 4; d <= DEPTH && !out_of_time(); d += 2) {
    best = solve(0, d, WIDTH).coor;
}
```

#### （4）中心随机开局

baseline 原版的黑第一手在 `[4,10]×[4,10]` 随机，现改为在中心 3×3 区域随机选择：

```cpp
{(6,6), (6,7), (6,8), (7,6), (7,7), (7,8), (8,6), (8,7), (8,8)}
```

该改动同时满足 先手第一手落在中心 7×7 范围内 和 引入非确定性，避免对局重复。

实测固定 `(7,7)` 开局时胜率约 50% 且波动较大，改为 9 点随机后胜率稳定在 75% 以上。

### 2.4 评估搜索与 swap 处理

#### 评估与状态传递

评估的分数权重沿用 baseline 的权重表，对每个空点四个方向的威胁值求和。搜索为带 TT 的 Minimax，窗口传递如下：

```cpp
tmp -= solve(color ^ 1, k - 1, width, tmp - high, tmp - low, new_key).w;
```

TT 存储精确分、下界和上界三种 flag，仅当满足剪枝条件时直接返回。

开局阶段根据 `turnID` 与 `ai_side` 调用不同策略函数，以处理换手的情况：

- 用**随机函数**抽取一个点作为黑第一手
- **`first_two_black()`**：黑第二手。对每个候选点假设黑下于此，计算白方（含 swap 选项）的最佳优势 `abs(solve(opp).w)`，选择该值最小的点。
- **`second()`**：白第一手。选择使局面最“失衡”的点，以利于后续 swap 决策。
- **`is_change()`**：白第二手。若 `solve(self).w > 0` 则继续下，否则返回 `(-1,-1)` 表示 swap。

#### 关键参数

```cpp
const int DEPTH = 14;              // 中盘最大搜索深度
const int FIRST_THREE_DEPTH = 10;  // 开局辅助搜索深度
const int WIDTH[DEPTH + 1] =
    {0, 0, 3, 3, 3, 3, 3, 4, 4, 5, 6, 7, 9, 11, 13};
const int FIRST_THREE_WIDTH[FIRST_THREE_DEPTH + 1] = 
    {0, 0, 3, 3, 4, 5, 5, 7, 9, 11, 13};

// 威胁权重表：WEIGHT[len-1][open_flag]
// open_flag = 0 表示两端都空；open_flag = 1 表示恰好一端空
const ll WEIGHT[4][2] = {
    {3, 1},
    {1000, 3},
    {1000000, 1000},
    {10000000000ll, 1000000}
};
```

### 2.5 对战 baseline 结果

使用 `python evaluate.py --agents-path ./baseline ./AIController --num-plays 1000 --num-workers 12` 评测的结果：（总花费时间约25分钟）

```
2026-07-08 14:32:06,391 INFO worker.py:2024 -- Started a local Ray instance.
************ Summary ************
num plays: 1000
Agent ./baseline:
score = -0.614 | wins = 193
Agent ./AIController:
score = 0.614 | wins = 807
```

可以看到 AIController 已能较为稳定击败 baseline，胜率约 80%。

---

## 3. AlphaZero AI（待补充）

*本部分将在完成 `AlphaZero/mcts_pure.py`、`policy_value_net_torch.py`、`mcts_alphaZero.py` 后补充。*

---

## 4. 项目建议（待补充）

*将在完整实现 AlphaZero 并训练后补充。*
