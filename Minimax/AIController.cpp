#include <iostream>
#include <vector>
#include <set>
#include <string>
#include <algorithm>
#include "AIController.h"
#include <utility>
#include <random>
#include <chrono>
#include <cstring>

using namespace std;
using namespace std::chrono;

// 每步思考开始时间 与 4.4 秒时限（留 0.6 秒余量）
static steady_clock::time_point move_start;
static const int TIME_LIMIT_MS = 4400;

inline bool out_of_time() {
    return duration_cast<milliseconds>(steady_clock::now() - move_start).count() >= TIME_LIMIT_MS;
}

typedef long long ll;

// 搜索深度
const int DEPTH = 14;              // 中盘最大深度
const int FIRST_THREE_DEPTH = 10;  // 开局辅助搜索深度

// 每层搜索宽度：根节点稍宽，越深越窄
const int WIDTH[DEPTH + 1] = {0, 0, 3, 3, 3, 3, 3, 4, 4, 5, 6, 7, 9, 11, 13};
const int FIRST_THREE_WIDTH[FIRST_THREE_DEPTH + 1] = {0, 0, 3, 3, 4, 5, 5, 7, 9, 11, 13};

namespace rand_int {
    std::mt19937 rnd(std::chrono::system_clock::now().time_since_epoch().count());
    int get(int l, int r) { return rnd() % (r - l + 1ll) + l; }
}
using rand_int::get;

// 威胁权重表：WEIGHT[len-1][open_flag]
// open_flag = 0 表示两端都空；open_flag = 1 表示恰好一端空
const ll WEIGHT[4][2] = {
    {3, 1},
    {1000, 3},
    {1000000, 1000},
    {10000000000ll, 1000000}
};

const int N = 15;  // 棋盘行数
const int M = 15;  // 棋盘列数

// 坐标，w 表示到边界距离的乘积，越靠近中心越大
struct Coordinate {
    int x, y, w;

    Coordinate(int _x = 0, int _y = 0) : x(_x), y(_y),
        w(min(_x + 1, N - _x) * min(_y + 1, M - _y)) {}

    Coordinate operator+(const Coordinate& b) const { return Coordinate(x + b.x, y + b.y); }
    Coordinate operator-(const Coordinate& b) const { return Coordinate(x - b.x, y - b.y); }
    Coordinate operator*(int b) const { return Coordinate(x * b, y * b); }
    friend Coordinate operator*(int a, const Coordinate& b) { return b * a; }

    bool operator==(const Coordinate& b) const { return x == b.x && y == b.y; }
    bool operator<(const Coordinate& b) const {
        return w == b.w ? (x == b.x ? y < b.y : x < b.x) : w < b.w;
    }
};

// 四个方向：横、竖、主对角、副对角
const Coordinate dir[4] = {Coordinate(0, 1), Coordinate(1, 0), Coordinate(1, 1), Coordinate(1, -1)};

// 某个点在某方向、某颜色上的连续状态
struct Status {
    int l, r;            // 向左/右连续同色子数（不含自身）
    bool lblank, rblank; // 左/右相邻是否为空

    Status() : l(0), r(0), lblank(false), rblank(false) {}

    int len() const { return l + r + 1; }
    int bl() const { return l + lblank; }
    int br() const { return r + rblank; }

    ll w() const {
        if (len() >= 5) return (ll)1e15;  // 已成五
        if (!lblank && !rblank) return 0;  // 两端被堵死
        return WEIGHT[len() - 1][lblank ^ rblank];
    }
};

// 空点候选，按威胁值 w 降序排列
struct Blank {
    Coordinate coor;
    ll w;

    Blank(Coordinate _coor, ll _w) : coor(_coor), w(_w) {}

    bool operator<(const Blank& b) const {
        return w == b.w ? b.coor < coor : w > b.w;
    }
};


// Zobrist 哈希 + 置换表
static uint64_t zobrist_board[N][M][2];  // 每个坐标每种颜色一个随机数
static uint64_t zobrist_side;            // 行棋方 key

struct TTEntry {
    uint64_t key = 0;
    int depth = -1;
    int flag = 0;       // 0 精确分，1 下界，2 上界
    ll score = 0;
    Coordinate best;
};

static const int TT_SIZE = 1 << 20;
static TTEntry tt[TT_SIZE];
static bool tt_initialized = false;
static std::mt19937_64 zobrist_rng(0x20250708);

static inline size_t tt_index(uint64_t key) { return key & (TT_SIZE - 1); }

static void init_tt() {
    if (tt_initialized) return;
    tt_initialized = true;
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < M; ++j)
            for (int k = 0; k < 2; ++k)
                zobrist_board[i][j][k] = zobrist_rng();
    zobrist_side = zobrist_rng();
    memset(tt, 0, sizeof(tt));
}

static inline void tt_store(uint64_t key, int depth, ll score, int flag, Coordinate best) {
    size_t idx = tt_index(key);
    if (depth >= tt[idx].depth) {
        tt[idx] = {key, depth, flag, score, best};
    }
}

static inline bool tt_probe(uint64_t key, int depth, ll alpha, ll beta, ll& score, Coordinate& best) {
    size_t idx = tt_index(key);
    if (tt[idx].key != key || tt[idx].depth < depth) return false;
    best = tt[idx].best;
    if (tt[idx].flag == 0) {
        score = tt[idx].score;
        return true;
    } else if (tt[idx].flag == 1 && tt[idx].score >= beta) {
        score = tt[idx].score;
        return true;
    } else if (tt[idx].flag == 2 && tt[idx].score <= alpha) {
        score = tt[idx].score;
        return true;
    }
    return false;
}


// 棋盘类
class Board {
private:
    int out_of_board = -1;
    vector<vector<int> > board;      // 0=self, 1=opp, 2=empty
    vector<vector<vector<vector<Status> > > > status;  // status[x][y][方向][颜色]
    set<Blank> blank;                // 空点候选集

    int& a(const Coordinate& coor) {
        if (coor.x < 0 || coor.x >= N || coor.y < 0 || coor.y >= M) return out_of_board;
        return board[coor.x][coor.y];
    }

    vector<vector<Status> >& s(const Coordinate& coor) {
        return status[coor.x][coor.y];
    }

    // 计算某点对我方/对方的威胁值
    ll calc(const Coordinate& coor, int color) const {
        const auto& st = status[coor.x][coor.y];
        return st[0][color].w() + st[1][color].w() + st[2][color].w() + st[3][color].w();
    }

    // 重新计算某点在某方向上的 Status，并维护 blank 集合
    void update(const Coordinate& coor, int direction) {
        if (a(coor) == out_of_board) return;
        Coordinate d = dir[direction];
        if (a(coor) == 2) blank.erase(Blank(coor, max(calc(coor, 0), calc(coor, 1))));
        for (int j = 0; j < 2; ++j) {
            Status& cur = s(coor)[direction][j];
            for (cur.l = 0; a(coor - (cur.l + 1) * d) == j; ++cur.l);
            for (cur.r = 0; a(coor + (cur.r + 1) * d) == j; ++cur.r);
            cur.lblank = a(coor - (cur.l + 1) * d) == 2;
            cur.rblank = a(coor + (cur.r + 1) * d) == 2;
        }
        if (a(coor) == 2) blank.insert(Blank(coor, max(calc(coor, 0), calc(coor, 1))));
    }

    // 落子或提子，并更新相关 Status
    void modify(const Coordinate& coor, int color) {
        if (a(coor) == out_of_board) return;
        if (a(coor) == 2) blank.erase(Blank(coor, max(calc(coor, 0), calc(coor, 1))));
        if (color == 2) blank.insert(Blank(coor, max(calc(coor, 0), calc(coor, 1))));
        a(coor) = color;
        for (int i = 0; i < 4; ++i) {
            update(coor, i);
            Coordinate d = dir[i];
            vector<Status>& cur = s(coor)[i];
            for (int j = -max(cur[0].bl(), cur[1].bl()); j <= cur[0].br() || j <= cur[1].br(); ++j) {
                if (j) update(coor + j * d, i);
            }
        }
    }

public:
    Board() : board(N, vector<int>(M, 2)),
        status(N, vector<vector<vector<Status> > >(M, vector<vector<Status> >(4, vector<Status>(2)))) {
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < M; ++j)
                for (int d = 0; d < 4; ++d)
                    update(Coordinate(i, j), d);
    }

    // 外部接口：color=2 表示 empty；(-1,-1) 表示 swap（翻转所有棋子颜色）
    void modify(int x, int y, int color) {
        if (x == -1 && y == -1) {
            for (int i = 0; i < N; ++i)
                for (int j = 0; j < M; ++j)
                    if (board[i][j] != 2)
                        modify(i, j, board[i][j] ^ 1);
        } else {
            modify(Coordinate(x, y), color);
        }
    }

    // 计算当前局面 Zobrist key
    uint64_t compute_key(int color) const {
        uint64_t key = (color == 0) ? zobrist_side : 0;
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < M; ++j)
                if (board[i][j] != 2)
                    key ^= zobrist_board[i][j][board[i][j]];
        return key;
    }

    // Negamax + Alpha-Beta 核心搜索
    Blank solve(int color, int k, const int* width, ll low = -(ll)1e18, ll high = (ll)1e18, uint64_t key = 0) {
        if (key == 0) key = compute_key(color);

        if (k == 1 || blank.size() == 1) {
            return Blank(blank.begin()->coor, (k + 10) * calc(blank.begin()->coor, color));
        }

        ll orig_low = low, orig_high = high;
        Coordinate tt_best;
        ll tt_score;
        if (tt_probe(key, k, low, high, tt_score, tt_best)) {
            return Blank(tt_best, tt_score);
        }

        ll mx = -(ll)1e18;
        Coordinate choose = blank.begin()->coor;
        auto it = blank.begin();

        for (int i = 0; i < width[k] && i < int(blank.size()); ++i, ++it) {
            Coordinate cur = it->coor;
            ll tmp = calc(cur, color) * (k + 10);
            if (tmp < (ll)1e16) {  // 非直接成五才继续搜索
                modify(cur, color);
                uint64_t new_key = key ^ zobrist_board[cur.x][cur.y][color] ^ zobrist_side;
                tmp -= solve(color ^ 1, k - 1, width, tmp - high, tmp - low, new_key).w;
                modify(cur, 2);
                it = blank.find(Blank(cur, max(calc(cur, 0), calc(cur, 1))));
            }
            if (tmp > mx) {
                mx = tmp;
                choose = cur;
            }
            low = max(low, mx);
            if (low >= high) break;  // beta 剪枝
        }

        int flag;
        if (mx <= orig_low) flag = 2;
        else if (mx >= orig_high) flag = 1;
        else flag = 0;
        tt_store(key, k, mx, flag, choose);
        return Blank(choose, mx);
    }

    // 返回威胁最大的前 K 个空点
    vector<Coordinate> top_empty(int K) const {
        vector<pair<ll, Coordinate>> v;
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < M; ++j)
                if (board[i][j] == 2)
                    v.push_back({max(calc(Coordinate(i, j), 0), calc(Coordinate(i, j), 1)), Coordinate(i, j)});
        sort(v.begin(), v.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
        vector<Coordinate> ret;
        for (int i = 0; i < min(K, (int)v.size()); ++i) ret.push_back(v[i].second);
        return ret;
    }

    // 中盘着法：迭代加深
    Coordinate turn() {
        Coordinate best = solve(0, 2, WIDTH).coor;
        for (int d = 4; d <= DEPTH && !out_of_time(); d += 2) {
            best = solve(0, d, WIDTH).coor;
        }
        return best;
    }

    // 白棋第一手：选择让局面最“失衡”的点，利于 swap 决策
    Coordinate second() {
        ll mx = -1;
        Coordinate ret;
        vector<Coordinate> cand = top_empty(40);
        for (const Coordinate& c : cand) {
            if (out_of_time()) break;
            modify(c, 0);
            ll res = abs(solve(1, FIRST_THREE_DEPTH, FIRST_THREE_WIDTH).w);
            modify(c, 2);
            if (res > mx || (res == mx && c.w > ret.w)) {
                mx = res;
                ret = c;
            }
        }
        return ret;
    }

    // 白棋第二手：决定是否 swap
    Coordinate is_change() {
        auto res = solve(0, DEPTH, WIDTH);
        if (res.w > 0) return res.coor;
        else return Coordinate(-1, -1);
    }

    // 黑棋第二手：最小化对手 swap 后的优势
    Coordinate first_two_black() {
        ll mn = (ll)1e18;
        Coordinate ret;
        vector<Coordinate> cand = top_empty(40);
        for (const Coordinate& c : cand) {
            if (out_of_time()) break;
            modify(c, 0);
            ll res = abs(solve(1, FIRST_THREE_DEPTH, FIRST_THREE_WIDTH).w);
            modify(c, 2);
            if (res < mn || (res == mn && c.w > ret.w)) {
                mn = res;
                ret = c;
            }
        }
        return ret;
    }
};

extern int ai_side;
std::string ai_name = "FilteredOpening";
std::vector<std::pair<int, int>> other, mine;  // 对方 / 己方历史落子

int turnID;
Board board;

void init() {
    turnID = 0;
    init_tt();
}

// 外部接口：返回本步着法
std::pair<int, int> action(std::pair<int, int> loc) {
    move_start = steady_clock::now();
    board = Board();
    for (int i = 0; i < turnID; i++) {
        board.modify(other[i].first, other[i].second, 1);
        board.modify(mine[i].first, mine[i].second, 0);
    }
    board.modify(loc.first, loc.second, 1);
    other.push_back(loc);

    Coordinate res;
    if (turnID == 0 && ai_side == 0) {
        // 黑第一手：在中心 3x3 内随机选择，满足 central 7x7 要求并增加非确定性
        static const vector<Coordinate> opens = {
            Coordinate(6, 6), Coordinate(6, 7), Coordinate(6, 8),
            Coordinate(7, 6), Coordinate(7, 7), Coordinate(7, 8),
            Coordinate(8, 6), Coordinate(8, 7), Coordinate(8, 8)
        };
        res = opens[get(0, (int)opens.size() - 1)];
    } else if (turnID <= 1 && ai_side == 0) {
        res = board.first_two_black();
    } else if (turnID == 0) {
        res = board.second();
    } else if (turnID == 1) {
        res = board.is_change();
    } else {
        res = board.turn();
    }

    turnID++;
    mine.emplace_back(res.x, res.y);

    return std::make_pair(res.x, res.y);
}
