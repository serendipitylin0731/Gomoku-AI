# AIController 逻辑文档

## 1. 整体架构

- 启动时接收一个整数 `ai_side`：`0` 表示黑棋（先手），`1` 表示白棋（后手）。
- 每轮调用 `action(loc)`，其中 `loc` 是对手上一步的坐标；`(-1, -1)` 表示对手选择 swap（交换黑白）。
- `action` 返回本步要下的坐标，或 `(-1, -1)` 表示接受 swap。

内部实现：
```
action(loc)                            
  └── 重建棋盘（replay other + mine）
  └── 根据 turnID / ai_side 选择策略
       ├── 黑第一手：中心随机开局
       ├── 黑第二手：first_two_black()
       ├── 白第一手：second()
       ├── 白第二手（swap 决策）：is_change()
       └── 中盘：turn()

```

---

## 2. 棋盘表示与增量状态

### 2.1 基础定义

```cpp
constexpr int N = 15, M = 15;
// 颜色编码：0 = self（本方可落子颜色），1 = opponent，2 = empty
```

注意：代码里始终用 `0` 代表“我方”、用 `1` 代表“对方”。无论 `ai_side` 是黑是白，棋盘重建时都会把我方历史落子记为 `0`、对方历史落子记为 `1`，因此搜索逻辑与颜色无关。

### 2.2 `Status` 结构

对每个坐标 `(x, y)`、每个方向 `d ∈ {0,1,2,3}`、每种颜色 `c ∈ {0,1}` 维护一个 `Status`：

```cpp
struct Status {
    int l, r;            // 向左/右连续同色子数（不含自身）
    bool lblank, rblank; // 左/右相邻格是否为空
};
```

四个方向为：

| 方向 | 向量 |
|------|------|
| 0    | (0, 1)  横向 |
| 1    | (1, 0)  纵向 |
| 2    | (1, 1)  主对角 |
| 3    | (1, -1) 副对角 |

当在 `(x, y)` 落子或提子时，调用 `modify(coor, color)`：
1. 从候选空点集合 `blank` 中删除/加入该点；
2. 更新该点四个方向的 `Status`；
3. 向左右各延伸 `max(bl(), br())` 格，重新计算受影响点的 `Status`。

### 2.3 候选空点集合 `blank`

```cpp
set<Blank> blank;
struct Blank {
    Coordinate coor;
    ll w;
    bool operator<(const Blank& b) const {
        return w == b.w ? b.coor < coor : w > b.w;  // 按威胁值从大到小
    }
};
```

每个空点的权重 `w = max(calc(coor, 0), calc(coor, 1))`，即该点对我方和对方威胁的最大值。集合按威胁值降序排列，搜索时直接取前 `WIDTH[depth]` 个，实现很好的 move ordering。

---

## 3. 评估函数

### 3.1 单点单方向价值

```cpp
ll Status::w() const {
    if (len() >= 5)       return 1e15;          // 已经成五
    if (!lblank && !rblank) return 0;           // 两端都被堵死
    return WEIGHT[len() - 1][lblank ^ rblank];  // 开放程度
}
```

- `lblank ^ rblank == 0`：两端都空（或都堵，但堵的情况已返回 0）。
- `lblank ^ rblank == 1`：恰好一端空。

### 3.2 权重表

```cpp
const ll WEIGHT[4][2] = {
    {3, 1},              // 1 子：活一 / 眠一
    {1000, 3},           // 2 子：活二 / 眠二
    {1000000, 1000},     // 3 子：活三 / 眠三
    {10000000000, 1000000}  // 4 子：活四 / 冲四
};
```

相邻等级之间差距足够大（10² ~ 10⁴），保证“成五 > 活四 > 冲四 > 活三 > 眠三 > …” 的严格层次。

### 3.3 单点价值

```cpp
ll calc(const Coordinate& c, int color) {
    return sum_{d=0..3} status[c.x][c.y][d][color].w();
}
```

### 3.4 搜索节点估值

在 Negamax 的节点上，对候选点 `cur`：

```cpp
ll tmp = calc(cur, color) * (depth + 10);
```

乘以 `(depth + 10)` 是为了让靠近根节点的节点价值更高，倾向于尽早选择能直接产生高威胁的着法。`1e16` 以上视为“直接成五”的必胜/必败分，不再向下搜索。

---

## 4. 搜索算法

### 4.1 Negamax + Alpha-Beta

```cpp
Blank solve(int color, int k, const int* width,
            ll low, ll high, uint64_t key = 0);
```

- `color`：当前行棋方（`0` 或 `1`）。
- `k`：剩余搜索深度。
- `width`：每层宽度表。
- `low / high`：alpha / beta 窗口。
- `key`：当前局面的 Zobrist 哈希。

流程：
1. 若 `k == 1` 或只剩一个空点，返回最优空点估值。
2. TT probe：若命中且深度足够，直接返回。
3. 枚举 `blank` 中前 `width[k]` 个候选点。
4. 对每个候选点 `cur`：
   - 落子 `modify(cur, color)`；
   - 递归 `solve(color ^ 1, k - 1, ...)`；
   - 还原 `modify(cur, 2)`；
   - 更新当前最优值与 alpha。
5. TT store 当前节点的分数与标志（exact / lower / upper）。

### 4.2 窗口传递

```cpp
tmp -= solve(color ^ 1, k - 1, width, tmp - high, tmp - low, new_key).w;
```

父节点对子节点的窗口是 `[tmp - high, tmp - low]`，这是 Negamax 的标准零和窗口转换。

### 4.3 迭代加深（中盘）

```cpp
Coordinate turn() {
    Coordinate best = solve(0, 2, WIDTH).coor;
    for (int d = 4; d <= DEPTH && !out_of_time(); d += 2) {
        best = solve(0, d, WIDTH).coor;
    }
    return best;
}
```

- `DEPTH = 14`。
- 从浅到深逐层搜索；如果时间用完，返回最近完成深度的最佳着法。
- 由于浅层结果通常已经不错，即使没搜到 14 层也能给出合理应手。

---

## 5. 置换表（TT）

### 5.1 Zobrist 哈希

- `zobrist_board[x][y][color]`：每个坐标、每种颜色一个 64 位随机数。
- `zobrist_side`：当前轮到 `color == 0` 行棋时的 side key。
- 局面 key = 所有非空格对应随机数异或，再视当前行棋方决定是否异或 `zobrist_side`。

增量更新：

```cpp
uint64_t new_key = key
                 ^ zobrist_board[cur.x][cur.y][color]
                 ^ zobrist_side;  // 切换行棋方
```

### 5.2 TT 条目

```cpp
struct TTEntry {
    uint64_t key;
    int depth;
    int flag;       // 0 exact, 1 lower bound, 2 upper bound
    ll score;
    Coordinate best;
};
```

- 表大小：`2^20` 项（约 32 MB）。
- 替换策略：同 key 下保留深度更大的条目。
- Probe 命中可用条件：
  - `flag == 0`：精确分，直接返回。
  - `flag == 1` 且 `score >= beta`：下界能剪枝，直接返回。
  - `flag == 2` 且 `score <= alpha`：上界能剪枝，直接返回。

TT 在中盘重复局面多时效果显著，使得 14 层搜索在 4.4 秒内可行。

---

## 6. 开局与 Swap 规则

### 6.1 行棋阶段判断

在 `action()` 中：

| 条件 | 调用函数 | 说明 |
|------|----------|------|
| `turnID == 0 && ai_side == 0` | 中心随机开局 | 黑第一手 |
| `turnID <= 1 && ai_side == 0` | `first_two_black()` | 黑第二手 |
| `turnID == 0` | `second()` | 白第一手 |
| `turnID == 1` | `is_change()` | 白决定是否 swap |
| 其他 | `turn()` | 中盘 |

### 6.2 黑第一手：中心随机开局

```cpp
static const vector<Coordinate> opens = {
    Coordinate(6,6), Coordinate(6,7), Coordinate(6,8),
    Coordinate(7,6), Coordinate(7,7), Coordinate(7,8),
    Coordinate(8,6), Coordinate(8,7), Coordinate(8,8)
};
res = opens[get(0, opens.size() - 1)];
```

- 全部落在 grading 要求的中心 7×7 内。
- 随机化同时满足 `evaluate.py` 对“至少一方具有随机性”的要求。

### 6.3 黑第二手：`first_two_black()`

目标：选择黑第二手，使得白方 swap 后的优势最小。

```cpp
Coordinate first_two_black() {
    ll mn = INF;
    Coordinate ret;
    for (c in top_empty(40)) {
        modify(c, 0);  // 假设黑下在这里
        ll res = abs(solve(1, DEPTH, WIDTH).w);  // 白方（含 swap 选项）最佳优势
        modify(c, 2);
        if (res < mn || tie_break) ret = c;
    }
    return ret;
}
```

- 只评估威胁最大的 top-40 空点，避免全局 225 点过慢。
- 用完整深度 `DEPTH = 14` 评估对手后续，比 baseline 的 6 层深得多。
- 平局时优先选择更靠近中心的点（`c.w > ret.w`）。

### 6.4 白第一手：`second()`

目标：选择白第一手，使得局面尽可能“失衡”，从而 swap 决策有利。

```cpp
Coordinate second() {
    ll mx = -1;
    Coordinate ret;
    for (c in top_empty(40)) {
        modify(c, 0);  // 模拟黑方已有的第一手 + 假设我方落子 c
        ll res = abs(solve(1, DEPTH, WIDTH).w);  // 黑方第二手后的最佳优势
        modify(c, 2);
        if (res > mx || tie_break) ret = c;
    }
    return ret;
}
```

### 6.5 Swap 决策：`is_change()`

白方在黑第二手后可以选择 swap：

```cpp
Coordinate is_change() {
    auto res = solve(0, DEPTH, WIDTH);
    if (res.w > 0) return res.coor;  // 不 swap，继续下
    else return Coordinate(-1, -1);   // swap
}
```

若评估认为当前局面下“我方作为黑棋继续下”更有利（`score > 0`），则不 swap；否则返回 `(-1,-1)` 交换颜色。

### 6.6 Swap 操作

当收到对手 `(-1,-1)` 时：

```cpp
void modify(int x, int y, int color) {
    if (x == -1 && y == -1) {
        for (所有非空格)
            modify(i, j, board[i][j] ^ 1);  // 黑白翻转
    } else {
        modify(Coordinate(x, y), color);
    }
}
```

---

## 7. 时间管理

```cpp
static steady_clock::time_point move_start;
static const int TIME_LIMIT_MS = 4400;

inline bool out_of_time() {
    return duration_cast<milliseconds>(
        steady_clock::now() - move_start
    ).count() >= TIME_LIMIT_MS;
}
```

- 每步预算 **4.4 秒**，低于 grading 的 5 秒硬限制。
- `move_start` 在每次 `action()` 开始重置。
- 开局辅助函数在评估每个候选点前检查时间；若超时则立即返回当前最佳候选。
- 中盘 `turn()` 在迭代加深的层间检查时间；若某层已开始则让其完成，避免返回不完整值。

---

## 8. 关键常数

```cpp
const int DEPTH = 14;              // 中盘最大搜索深度
const int FIRST_THREE_DEPTH = 10;  // 开局辅助函数搜索深度

const int WIDTH[DEPTH + 1] =
    {0, 0, 3, 3, 3, 3, 3, 4, 4, 5, 6, 7, 9, 11, 13};

const int FIRST_THREE_WIDTH[FIRST_THREE_DEPTH + 1] =
    {0, 0, 3, 3, 4, 5, 5, 7, 9, 11, 13};
```

宽度表设计为“根节点较宽、深层较窄”，在控制分支因子的同时保证关键候选不被遗漏。
