---
title: "Leetcode Solutions 061 080"
date: 2018-08-02T11:33:25+08:00
tags: ["leetcode"]
---

本文用来记录我的leetcode解题报告，用于之后的代码优化、查阅和复习。这里是第61题到第80题。

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

## 64. Minimum Path Sum

链接：https://leetcode.com/problems/minimum-path-sum/description/

思路：和62题63题类似，区别在于从左边和上方找到较小的值加和，就是当前格子的最小路径和，注意边界的处理。比较简单，一遍通过。

```cpp
class Solution {
public:
    int minPathSum(vector<vector<int>>& grid) {
        int m = grid.size();         if (m == 0) return 0;
        int n = grid.front().size(); if (n == 0) return 0;
        for (int i=0; i<m; i++) for (int j=0; j<n; j++) {
            (i == 0 && j >  0) && (grid[i][j] += grid[i][j-1]);
            (i >  0 && j == 0) && (grid[i][j] += grid[i-1][j]);
            (i >  0 && j >  0) && (grid[i][j] += min(grid[i-1][j], grid[i][j-1]));
        }
        return grid.back().back();
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

## 66. Plus One

链接：https://leetcode.com/problems/plus-one/description/

思路：加1，逐一计算carry，比较简单，一遍过。

```cpp
class Solution {
public:
    vector<int> plusOne(vector<int>& digits) {
        const int len = digits.size();
        if (len == 0) return vector<int>();
        int carry = 0; digits[len-1] += 1;
        for (int i=len-1; i>=0; i--) {
            digits[i] += carry;
            carry = digits[i] / 10;
            digits[i] %= 10;
        }
        if (carry != 0) {
            digits.insert(begin(digits), carry);
        }
        return digits;
    }
};
```

## 67. Add Binary

链接：https://leetcode.com/problems/add-binary/description/

思路：从后依次往前加，注意carry即可。

```cpp
class Solution {
public:
    string addBinary(const string &a, const string &b) {
        int alen=a.length(), blen=b.length(), len=max(alen, blen), carry=0;
        if (len == 0) return "";
        string c(len, '0');
        for (int i=0; i<len; i++) {
            (i < alen) && (carry += a[alen-i-1]-'0');
            (i < blen) && (carry += b[blen-i-1]-'0');
            c[len-i-1] = carry % 2 + '0';
            carry = carry / 2;
        }
        if (carry != 0) c.insert(begin(c), carry+'0');
        return move(c);
    }
};
```

## 68. Text Justification

链接：https://leetcode.com/problems/text-justification/description/

思路：逐行处理，先计算出当前行最大包含的单词，然后计算总空格数，分别处理 单个单词 最后一行 常规 等三种情况。其中 单个单词 和 最后一行 的处理不能交换，是因为什么呢？

```cpp
class Solution {
public:
    vector<string> fullJustify(const vector<string>& S, const int W) {
        vector<string> result; int p = 0;
        while (p < S.size()) {
            result.push_back(nextJustifyLine(S, p, W));
        }
        return result;
    }
    
    string nextJustifyLine(const vector<string> &S, int &p, const int W) {
        string line; vector<string> w; int Nw = 0; line.reserve(W);
        while (p < S.size() && (Nw + S[p].length() + w.size() <= W)) {
            w.push_back(S[p]); Nw += S[p].length(); p++;
        }
        const int Ns = W - Nw, Np = w.size() - 1; // num of spaces and num of pos.
        if (w.size() == 1) {
            line.append(w[0]).append(Ns, ' ');
        } else if (p == S.size()) {            // last line.
            for (int i=0; i<=w.size()-2; i++) 
                line.append(w[i]).push_back(' ');
            line.append(w.back()).append(Ns-Np, ' ');
        } else {
            for (int i=0; i<=w.size()-2; i++) 
                line.append(w[i]).append(Ns / Np + (i < Ns%Np ? 1 : 0), ' ');
            line.append(w.back());
        }
        return move(line);
    }
};
```

## 69. Sqrt(x)

链接；https://leetcode.com/problems/sqrtx/description/

思路：这题比较简单，注意nxn不溢出就可以了。

```cpp
class Solution {
public:
    int mySqrt(int x) {
        for (long n = 0; n <= (x/2+2); n++) if (n * n > x) return n-1;
        return 0;
    }
};
```

## 70. Climbing Stairs

链接：https://leetcode.com/problems/climbing-stairs/description/

思路：f(n) = f(n-1) + f(n-2)，第n步的可能性等于第n-1步加上第n-2步。以此内推即可。

```cpp
class Solution {
public:
    int climbStairs(int n) {
        if (n <= 2) return n;
        int a = 3, b = 2, c  = 1;
        for (int i = 3; i <= n; i++) {
            a = b + c; c = b; b = a;
        }
        return a;
    }
};
```

## 71. Simplify Path

链接：https://leetcode.com/problems/simplify-path/

思路：将每个有效的component存储再数组中，过程中移除空目录，处理特殊的目录如".."和"."，注意vector为空时，pop_back会报异常。注意if和elseif的匹配问题，这题必须要加上大括号，想想为什么。

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

```cpp
class Solution {
public:
    string simplifyPath(string& path) {
        const int len = path.length(); int l=1, h=1; vector<string> p;
        if (len==0 || path[0]!='/') return "";
        while (l < len) {
            while (h<len && path[h]!='/') h++;
            string tmp = path.substr(l, h-l);
            if      (tmp == "..") { if (!p.empty()) p.pop_back(); }
            else if (tmp == "." ) { /* do nothing */              }
            else if (h - l > 0  ) { p.push_back(move(tmp));       }
            l = ++h;
        }
        if (p.empty()) return "/";
        string result;
        for (auto& s : p) { result.append("/").append(s); }
        return move(result);
    }
};
```

## 72. Edit Distance

链接：https://leetcode.com/problems/edit-distance/description/

思路：动态规划，`dp[i][j]`表示`src[0, i-1]`与`tar[0, j-1]`的编辑举例，

如果`src[i-1]`等于`tar[j-1]`，那么`dp[i][j] == dp[i-1][j-1]`，因为无需额外的编辑操作。

如果`src[i-1] != tar[j-1]`，那么有三种编辑可能，

1 一是在`dp[i-1][j-1]`基础上通过替换`src[i-1]`为`tar[j-1]`得来，
2 二是在`dp[i][j-1]`基础上从`src[i-1]`的位置插入一个字符得来，
3 三是从`dp[i-1][j]`基础上通过删除`src[i-1]`位置的一个字符得来。所以，取三种可能的最小值并加上一步编辑距离。

另外，初始化`dp[i][0]`表示目标字符串为空的情况，那么只要删除src中的i个字符即可，操作步骤为i。`dp[0][j]`表示源字符串为空，那么添加j个字符即可，操作步骤数为j。

最后，其实每步计算，只需要左方 右方和上方的dp值即可，可以把二维表简化为单行来计算，可以把空间从O(mn)减少到O(n)。

```cpp
class Solution {
public:
    int minDistance(const string &src, const string &tar) {
        int m=src.size(), n=tar.size();
        vector<vector<int>> dp(m+1, vector<int>(n+1, 0));
        for (int i=1; i<m+1; i++) dp[i][0] = i;
        for (int j=1; j<n+1; j++) dp[0][j] = j;
        for (int i=1; i<m+1; i++) for (int j=1; j<n+1; j++) {
            if (src[i-1] == tar[j-1]) { 
                dp[i][j] = dp[i-1][j-1]; 
            } else {
                dp[i][j] = min({dp[i][j-1], dp[i-1][j], dp[i-1][j-1]}) + 1;
            }
        }
        return dp[m][n];
    }
};
```

## 73. Set Matrix Zeroes

思路：用第一行和第一列标记，对应的列和行是否需要填充0，用r0和c0标记第一行和第一列是否需要填充0。

链接：https://leetcode.com/problems/set-matrix-zeroes/description/

```cpp
class Solution {
public:
    void setZeroes(vector<vector<int>>& M) {
        const int m = M.size(); if (m == 0) return;
        const int n = M[0].size();
        int r0 = 1, c0 = 1;
        for (int i=0; i<m; i++) if (!M[i][0]) c0 = 0;
        for (int j=0; j<n; j++) if (!M[0][j]) r0 = 0;
        
        for (int i=1; i<m; i++) for (int j=1; j<n; j++) 
            if (!M[i][j]) M[i][0] = M[0][j] = 0;
        
        for (int i=1; i<m; i++) for (int j=1; j<n; j++)
            if (!M[i][0] || !M[0][j]) M[i][j] = 0;
        
        if (!c0) for (int i=0; i<m; i++) M[i][0] = 0;
        if (!r0) for (int j=0; j<n; j++) M[0][j] = 0;
    }
};
```

## 74. Search a 2D Matrix

链接：https://leetcode.com/problems/search-a-2d-matrix/description/

思路：先顺着第一列查找，再查找对应的行，可以用binary_search提一下速。有时间的话，也可以考虑自己手撸一个二分查找。

```cpp
class Solution {
public:
    bool searchMatrix(vector<vector<int>>& M, int T) {
        const int m = M.size();    if (m == 0) return false;
        const int n = M[0].size(); if (n == 0) return false;
        int i = 0;
        for (i=0; i<m; i++) if (M[i].front()<=T && T<=M[i].back()) break;
        if (i == m) return false;
        return binary_search(begin(M[i]), end(M[i]), T);
    }
};
```

## 75. Sort Colors

链接：https://leetcode.com/problems/sort-colors/description/

思路：题目的Follow up，要单趟排序还是挺难的，和《算法导论》里的快排思路挺类似的，区别是这个有三类。不过理解了，就简单了。也就是[0,l)表示0的存放区间，(h, len-1]表示2的存放区间。i从头到尾的扫描，每次遇到就要将元素放置到正确的位置上。要注意 循环条件 `i<=h`的判断，和向后交换时`i--`保持i的位置不动。

```cpp
class Solution {
public:
    void sortColors(vector<int>& nums) {
        const int len = nums.size();
        int l=0, h=len-1;
        for (int i=0; i<=h; i++) {
            if      (nums[i] == 0) swap(nums[i], nums[l++]);
            else if (nums[i] == 2) swap(nums[i--], nums[h--]);
        }
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

## 77. Combinations

链接：https://leetcode.com/problems/combinations/description/

思路：用深搜，可以比较轻易的搜出来，就是效率比较低，over 32.4%   

```cpp
class Solution {
public:
    vector<vector<int>> combine(int n, int k) {
        vector<vector<int>> result; vector<int> path;
        dfs(n, k, 1, path, result);
        return move(result);
    }
    
    void dfs(int n, int k, int p, vector<int> &path, vector<vector<int>> &result) {
        if (path.size() == k) { result.push_back(path); return; }
        if (p > n) return;

        dfs(n, k, p+1, path, result);
        
        path.push_back(p);
        dfs(n, k, p+1, path, result);
        path.pop_back();
    }
};
```

## 78. Subsets

链接：https://leetcode.com/problems/subsets/description/

思路：可以用dfs，也可以用位操作算法，应该子集的索引刚好和 1 2 3 4 .. n 的二进制表示法中的1的个数相同。


```cpp
class Solution {
public:
    vector<vector<int>> subsets(vector<int>& nums) {
        const int len = nums.size(); vector<vector<int>> result;
        for (int i=0; i<(1<<len); i++) {
            vector<int> subset; subset.reserve(len);
            for (int j=0; j<len; j++) if (i & (1<<j)) subset.push_back(nums[j]);
            result.push_back(move(subset));
        }
        return move(result);
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

## 80. Remove Duplicates from Sorted Array II

链接：https://leetcode.com/problems/remove-duplicates-from-sorted-array-ii/description/

思路：挺直观的一道题，用p索引处理后数据，用i索引处理前数据，用count来计数即可。

```cpp
class Solution {
public:
    int removeDuplicates(vector<int>& N) {
        int p=1, count=1, len=N.size();
        if (len == 0) return 0;
        for (int i=1; i<len; i++) {
            (N[i]!=N[i-1]) && (count=1);
            (N[i]==N[i-1]) && (count++);
            if (count <= 2) N[p++] = N[i];
        }
        return p;
    }
};
```