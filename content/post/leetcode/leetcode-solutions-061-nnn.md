---
title: "Leetcode Solutions 061 nnn"
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

## 84. Largest Rectangle in Histogram

链接：https://leetcode.com/problems/largest-rectangle-in-histogram/

思路：使用栈，遍历数组，如果比top大则压入并继续遍历，如果比top小，则弹出top计算面积直到top比当前元素小时为止。

```cpp
class Solution {
public:
    int largestRectangleArea(vector<int>& heights) {
        stack<int> s; int i=0; int result=0, len=heights.size();
        while (i<len || !s.empty()) {
            if (i<len && (s.empty()||heights[s.top()]<=heights[i])) s.push(i++);
            else {
                int h = heights[s.top()]; s.pop();
                int l = s.empty() ? 0 : s.top()+1;
                result = max(result, (i-l)*h);
            }
        }
        return result;
    }
};
```

## 85. Maximal Rectangle

链接：https://leetcode.com/problems/maximal-rectangle/

思路：接上一题，按行遍历，每行用数组f存储每列1的高度，然后计算最大直方图面积。

```cpp
class Solution {
public:
    int maximalRectangle(vector<vector<char>>& matrix) {
        if (matrix.empty()) return 0;
        const int ROW=matrix.size(), COL=matrix[0].size();
        vector<int> f(COL, 0); int j=0, result=0; stack<int> s;
        for (int i=0; i<ROW; ++i) {
            for (j=0; j<COL; ++j)
                if (matrix[i][j] == '1') f[j]++; else f[j] = 0;
            j = 0;
            while (j<COL || !s.empty()) {
                if (j<COL&&(s.empty() || f[s.top()]<=f[j]))
                    s.push(j++);
                else {
                    int h = f[s.top()]; s.pop();
                    int l = s.empty() ? 0 : s.top() + 1;
                    result = max(result, h*(j-l));
                }
            }
        }
        return result;
    }
};
```

## 91. Decode Ways

链接：https://leetcode.com/problems/decode-ways/

思路：f[i] = f[i-1] + f[i-2]，也可以写成c = b + a，只要s[i]不为0，b就可以和s[i]组合。而a则需要s[i-1]s[i]组成的数字小于26。
* 使用a b b的形式，而不使用a b c的形式，可以避免判断返回b还是c。

```cpp
class Solution {
public:
    int numDecodings(string s) {
        if (s.empty() || s[0] == '0') return 0; 
        int a=1, b=1;
        for (int i=1; i<s.length(); i++) {
            int tmp = b;
            if (s[i] == '0')                             b = 0;
            if (s[i-1]=='1' || (s[i-1]=='2'&&s[i]<='6')) b += a;
            a = tmp;
        }
        return b;
    }
};
```

## 93. Restore IP Addresses

链接：https://leetcode.com/problems/restore-ip-addresses/

思路：深搜，具体看代码。需要注意 1.每个component必须小于255 2.component连续两次出现0则无效 3.剪枝：当剩余的字符数比剩余的component最大可能长度还多则无效，可以防止非常长的字符串导致搜索变慢的情况。 

```cpp
class Solution {
public:
    vector<string> restoreIpAddresses(string& s) {
        vector<string> result, ip;
        dfs(s, ip, 0, result);
        return move(result);
    }
    
    void dfs(string& s, vector<string>& ip, int l, vector<string>& result) {
        int n=0, h=l, len=s.length();
        if (l==len || (len-l)>(4-ip.size())*3) return ;
        if (ip.size() == 3) {
            while (h<len) { n = n*10 + s[h++] - '0'; if (n == 0) break; }
            if (h!=len || n>255) return ;
            string ss; for (int i=0; i<3; i++) ss.append(ip[i]).push_back('.'); ss.append(s.substr(l));
            result.push_back(move(ss));
            return ;
        }
        while (h < len) {
            n = n * 10 + s[h] - '0';
            if (n > 255) return ;
            ip.push_back(s.substr(l, h-l+1)); dfs(s, ip, h+1, result); ip.pop_back();
            if (n == 0) return ;
            h++;
        }
    }
};
```

## 97. Interleaving String

链接：

思路：动态规划，f[i][j]表示s1[i,len1]和s2[j,len2]能否interleave字符串s3[i+j,len3]。所以，`f[i][j] = (f[i+1][j]&&s1[i]==s3[i+j]) || (f[i][j+1]&&s2[j]==s3[i+j])`。

```cpp
class Solution {
public:
    bool isInterleave(string& s1, string& s2, string& s3) {
        const size_t len1=s1.length(), len2=s2.length(), len3=s3.length();
        if (len1+len2 != len3) return false;
        vector<vector<int>> f(len1+1, vector<int>(len2+1, 0));
        f[len1][len2] = 1;
        for (int i=len1; i>=0; --i) for (int j=len2; j>=0; --j) {
            int k = i + j;
            f[i][j] = (i<len1&&s1[i]==s3[k]&&f[i+1][j]) || (j<len2&&s2[j]==s3[k]&&f[i][j+1]) || f[i][j];
        }
        return f[0][0];
    }
};
```

## 98. Validate Binary Search Tree

链接：

思路：需要每个节点满足 L < P < R，同时要注意R子树中的最小节点需要比P大，L子树中的最大节点需要比P小。

```cpp
class Solution {
public:
    bool isValidBST(TreeNode* root) {
        if (root == nullptr) return true;
        int min = INT_MAX, max = INT_MIN;
        return dfs(root, min, max);
    }
    
    bool dfs(TreeNode *node, int& min, int& max) {
        int lmax=INT_MIN, lmin=INT_MAX, rmax=INT_MIN, rmin=INT_MAX; bool result;
        if (node->left) {
            result = node->left->val < node->val && dfs(node->left, lmin, lmax);
            if (!result || lmax >= node->val) return false;
        }
        if (node->right) {
            result = node->right->val > node->val && dfs(node->right, rmin, rmax);
            if (!result || rmin <= node->val) return false;
        }
        max = std::max(node->val, rmax); 
        min = std::min(node->val, lmin);
        return true;
    }
};
```