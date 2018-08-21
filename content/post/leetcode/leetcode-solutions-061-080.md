---
title: "Leetcode Solutions 061 080"
date: 2018-08-02T11:33:25+08:00
tags: ["leetcode"]
---

本文用来记录我的leetcode解题报告，用于之后的代码优化、查阅和复习。这里是第61题到第nnn题。

<!--more-->

## 61. Rotate List

链接：https://leetcode.com/problems/rotate-list/

思路：先计算链表长度len，双指针p1和p2，p1先走k步，开始同时走p1和p2，当p1->next指向null时，p2指向倒数第k+1个元素，调整两部分链表的前后顺序即可。

```cpp
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
class Solution {
public:
    ListNode* rotateRight(ListNode* head, int k) {
        ListNode *p1=head, *p2=head;
        int len=0; while (p1!=nullptr) { p1=p1->next; len++; }
        if (len == 0) return NULL;
        k = k % len; p1 = head;
        if (k == 0) return head;
        for (int i=0; i<k; ++i) p1 = p1->next;
        while (p1->next != nullptr) { p1 = p1->next; p2 = p2->next; }
        p1->next = head; head = p2->next; p2->next = nullptr;
        return head;
    }
};
```

## 62. Unique Paths

链接: https://leetcode.com/problems/unique-paths/description/

思路: 每一个格子的可能路径数等于左边格子路径书加上上方格子路径数。

```cpp
class Solution {
public:
    int uniquePaths(int m, int n) {
        vector<vector<int>> tbl(m, vector<int>(n, 0));
        tbl[0][0] = 1;
        for (int i = 0; i < m; i++) for (int j = 0; j < n; j++) {
            (i > 0) && (tbl[i][j] += tbl[i-1][j]);
            (j > 0) && (tbl[i][j] += tbl[i][j-1]);
        }
        return tbl.back().back();
    }
};
```

## 63. Unique Paths II

链接：https://leetcode.com/problems/unique-paths-ii/description/

思路：注意 [0, 0]位置 和 [m, n] 位置不能为1。可以复用grid作为表用，加和时需要判断一下[i, j]是否为1，已经邻居格是否为1。

```cpp
class Solution {
public:
    int uniquePathsWithObstacles(vector<vector<int>>& grid) {
        if (grid.empty() || grid.front().empty()) return 0;
        if (grid.front().front() || grid.back().back()) return 0; 
        grid[0][0] = -1;
        for (int i = 0; i < grid.size(); i++) for (int j = 0; j < grid.front().size(); j++) {
            if (grid[i][j] == 1) continue;
            (j > 0) && (grid[i][j-1] < 0) && (grid[i][j] += grid[i][j-1]);
            (i > 0) && (grid[i-1][j] < 0) && (grid[i][j] += grid[i-1][j]);
        }
        return -grid.back().back();
    }
};
```

## 65. Valid Number

链接：https://leetcode.com/problems/valid-number/

思路：用状态机，transition的条件为输入字符的类型，状态为0到9，其中0表示开始，99表示成功结束。每行表示从当前状态（行下标）遇到对应的条件跳转到的下一个状态，-1表示无效终止。其他思路，一个完整的数字包括+123.123e+123，其中符号部分是可选的，第一个123和第二个123必须出现一个，最后的e部分也是可选的。

```cpp
class Solution {
public:
    bool isNumber(const string &s) {
        enum ConditionType {
            INVALID, SPACE, SIGN, DIGIT, DOT, EXPONENT, ENDSTR, NUMINPUTS
        };
        int next[][NUMINPUTS] = {
            -1,       0,     1,    2,     3,  -1,       -1, // 0
            -1,      -1,    -1,    2,     3,  -1,       -1, // 1
            -1,       9,    -1,    2,     4,   5,       99, // 2
            -1,      -1,    -1,    6,    -1,  -1,       -1, // 3
            -1,       9,    -1,    6,    -1,   5,       99, // 4
            -1,      -1,     7,    8,    -1,  -1,       -1, // 5
            -1,       9,    -1,    6,    -1,   5,       99, // 6
            -1,      -1,    -1,    8,    -1,  -1,       -1, // 7
            -1,       9,    -1,    8,    -1,  -1,       99, // 8
            -1,       9,    -1,   -1,    -1,  -1,       99, // 9
        };
        int state = 0, condition;
        for (auto ch : s) {
            if (isspace(ch))                 condition = SPACE;
            else if (isdigit(ch))            condition = DIGIT;
            else if (ch=='+' || ch=='-')     condition = SIGN;
            else if (ch == '.')              condition = DOT;
            else if (ch == 'E' || ch == 'e') condition = EXPONENT;
            else                             condition = INVALID;
            state = next[state][condition];
            if (state == -1) return false;
        }
        return next[state][ENDSTR] == 99;
    }
};
```

## 71. Simplify Path

链接：https://leetcode.com/problems/simplify-path/

思路：将每个有效的component存储再数组中，过程中移除空目录，处理特殊的目录如".."和"."，注意vector为空时，pop_back会报异常。

```cpp
class Solution {
public:
    string simplifyPath(string& path) {
        const int len = path.length(); int l=1, h=1; vector<string> p;
        if (len==0 || path[0]!='/') return "";
        while (l < len) {
            while (h<len && path[h]!='/') h++;
            string tmp = path.substr(l, h-l);
            if      (tmp == "..")       { if (!p.empty()) p.pop_back(); }
            else if (tmp!="." && h-l>0) { p.push_back(move(tmp)); }
            l = ++h;
        }
        if (p.empty()) return "/";
        string result;
        for (auto& s : p) { result.push_back('/'); result.append(s); }
        return move(result);
    }
};
```

## 76. Minimum Window Substring

链接：https://leetcode.com/problems/minimum-window-substring/

思路：双指针，先移动i直到t中的字符全部在s[l,i]中出现，在移动l使得s[l,i]满足条件的同时i-l最小，计数使用数组。

```cpp
class Solution {
public:
    string minWindow(string& s, string& t) {
        int S[256], T[256], l=0, count=0, minl=0, minh=INT_MAX;
        fill_n(S, 256, 0); fill_n(T, 256, 0);
        for (auto c : t) T[c]++;
        for (int i=0; i<s.length(); i++) {
            S[s[i]]++;
            if (S[s[i]] <= T[s[i]]) count++;
            if (count == t.length()) {
                while (S[s[l]] > T[s[l]]) S[s[l++]]--;
                if (minh-minl > i-l) { minh = i; minl = l; }
                S[s[l++]]--; count--;
            }
        }
        if (minh == INT_MAX) return "";
        return move(s.substr(minl, minh-minl+1));
    }
};
```

## 79. Word Search

递归求解

```cpp
class Solution {
public:
    bool exist(vector<vector<char>>& board, string word) {
        if (board.empty()) return false;
        for (int x=0; x<board.size(); x++) for (int y=0; y<board[0].size(); y++)
            if (dfs(board, x, y, word, 0)) return true;
        return false;
    }
    
    bool dfs(vector<vector<char>>& board, int x, int y, string& word, int i) {
        if (board[x][y] != word[i]) return false;
        if (i == word.size()-1)     return true;
        char c = board[x][y]; board[x][y] = '*'; bool result = false;
        if (x-1 >= 0)              result = result || dfs(board, x-1, y,   word, i+1);
        if (x+1 < board.size())    result = result || dfs(board, x+1, y,   word, i+1);
        if (y-1 >= 0)              result = result || dfs(board, x,   y-1, word, i+1);
        if (y+1 < board[0].size()) result = result || dfs(board, x,   y+1, word, i+1);
        board[x][y] = c;
        return result;
    }
};
```

非递归求解

```cpp
class Solution {
public:
    int ladderLength(string& start, string& end, unordered_set<string>& words) {
        unordered_set<string> visited, curr, next;
        visited.insert(start), curr.insert(start);
        int depth = 1;
        while (!curr.empty()) {
            depth++;
            for (auto word : curr) {
                for (int i=0; i<word.size(); i++) for (char c='a'; c<='z'; c++) {
                    if (word[i] == c) continue;
                    swap(word[i], c);
                    if (word == end) return depth;
                    if (words.find(word)!=words.end() && visited.find(word)==visited.end()) 
                        next.insert(word);
                    swap(word[i], c);
                }
            }
            for (auto& word : next) visited.insert(word);
            curr.clear(); swap(curr, next);
        }
        return 0;
    }
};
```