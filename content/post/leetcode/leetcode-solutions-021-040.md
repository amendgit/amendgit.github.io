---
title: "Leetcode Solutions 021 040"
date: 2018-07-01T11:33:03+08:00
tags: ["leetcode"]
---

本文用来记录我的leetcode解题报告，用于之后的代码优化、查阅和复习。这里是第21题到第40题。

<!--more-->

## 21. Merge Two Sorted Lists 
链接：https://leetcode.com/problems/merge-two-sorted-lists/#/description
使用优先级队列解决该问题

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
    ListNode* mergeTwoLists(ListNode* l1, ListNode* l2) {
        struct NodeGreater { bool operator()(ListNode *lhs, ListNode *rhs) { return lhs->val > rhs->val; } };
        priority_queue<ListNode *, vector<ListNode *>, NodeGreater> q;
        if (l1) q.push(l1); if (l2) q.push(l2);
        ListNode dummy = ListNode(0); ListNode *p = &dummy;
        while (!q.empty()) {
            ListNode *tmp = q.top(); q.pop();
            if (tmp->next) q.push(tmp->next);
            p->next = tmp; p = p->next;
        }
        p->next = nullptr;
        return dummy.next;
    }
};
```

## 22. Generate Parentheses 
链接：https://leetcode.com/problems/generate-parentheses/#/description

```cpp
class Solution {
public:
    vector<string> generateParenthesis(int n) {
        vector<string> result; string path;
        generateParenthesis(0, 0, n, path, result);
        return move(result);
    }
    
    void generateParenthesis(int o, int c, int n, string &path, vector<string>& result) {
        if (o + c == n*2) { result.push_back(path); return; }
        
        if (o < n) { 
            path.push_back('('); generateParenthesis(o+1, c, n, path, result); path.pop_back(); 
        }
        
        if (o > c && c < n) { 
            path.push_back(')'); generateParenthesis(o, c+1, n, path, result); path.pop_back(); 
        }
    }
};
```

## 23. Merge k Sorted Lists
链接：https://leetcode.com/problems/merge-k-sorted-lists/
思路：
1. 优先级队列，压入每个链表当前节点.
2. 取出最小节点，组成新链表，压入该节点的非空next.
3. 队列为空时结束.

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
    ListNode* mergeKLists(vector<ListNode*>& lists) {
        struct NodeGreater { bool operator()(ListNode *lhs, ListNode *rhs) { 
            return lhs->val > rhs->val; 
        } };
        priority_queue<ListNode*, vector<ListNode*>, NodeGreater> q;
        for (auto node : lists) if (node) q.push(node); 
        ListNode dummy(0), *p = &dummy; p->next = nullptr;
        while (!q.empty()) {
            ListNode *node = q.top(); q.pop();
            p->next = node; p = p->next;
            if (node->next) q.push(node->next);
        }
        return dummy.next;
    }
};
```

## 24. Swap Nodes in Pairs 

链接： https://leetcode.com/problems/swap-nodes-in-pairs/#/description

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
    ListNode* swapPairs(ListNode* head) {
        ListNode dummy(0); dummy.next = head;
        ListNode *p = &dummy, *a, *b, *tmp;
        while (true) {
            if (p->next) a = p->next; else break;
            if (a->next) b = a->next; else break;
            tmp = b->next;
            b->next = a; a->next = tmp; p->next = b;
            p = a;
        }
        return dummy.next;
    }
};
```

## 25. Reverse Nodes in k-Group 
链接：https://leetcode.com/problems/reverse-nodes-in-k-group/#/description

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
    ListNode* reverseKGroup(ListNode* head, int k) {
        if (k == 1) return head;
        ListNode dummy(0); dummy.next = head;
        ListNode *p = &dummy, *q, *hold, *first, *last;
        while (true) {
            q = p;
            for (int i = 0; i < k; ++i) if (q && q->next) q = q->next; else return dummy.next;
            hold = q->next; q->next = nullptr;
            reverseList(p->next, &first, &last);
            p->next = first; last->next = hold; p = last;
        }
        
        return dummy.next;
    }
    
    void reverseList(ListNode *head, ListNode **first, ListNode **last) {
        static ListNode dummy(0); dummy.next = head; 
        ListNode *a = &dummy, *b = a->next, *c = nullptr;
        while (b) { c = b->next; b->next = a; a = b; b = c; }
        *first = a; *last = head; head->next = nullptr;
    }
};
```

## 26. Remove Duplicates from Sorted Array
链接：https://leetcode.com/problems/remove-duplicates-from-sorted-array/#/description

```cpp
class Solution {
public:
    int removeDuplicates(vector<int>& nums) {
        auto len = nums.size();
        if (len <= 1) return len;
        int last = nums[0], i = 1, j = 1;
        while (j < len) {
            if (nums[j] != last) { nums[i++] = nums[j]; last = nums[j]; }
            j++;
        }
        return i;
    }
};
```

## 27. Remove Element 
链接：https://leetcode.com/problems/remove-element/#/description

```cpp
class Solution {
public:
    int removeElement(vector<int>& nums, int val) {
        int i = 0, j = 0;
        for (j = 0; j < nums.size(); j++) {
            if (nums[j] == val) continue;
            nums[i++] = nums[j];
        }
        return i;
    }
};
```

## 28. Implement strStr() 
链接：https://leetcode.com/problems/implement-strstr/#/description
思路：暴力求解
耗时3ms

```cpp
class Solution {
public:
    int strStr(const string &haystack, const string &needle) {
        int lenH = haystack.length(), lenN = needle.length();
        if (lenH < lenN) return -1;
        for (int i = 0; i <= lenH - lenN; i++) {
            bool eq = true;
            for (int j = 0; j < lenN; j++) 
                if (haystack[i+j] != needle[j]) { eq = false; break; }
            if (eq) return i;
        }
        return -1;
    }
};
```

## 29. Divide Two Integers
链接：https://leetcode.com/problems/divide-two-integers/
思路：

* unsigned int(0 ~ 4294967295), INT_MIN(-2147483648), INT_MAX(2147483647)。
* INT_MAX的二进制表示为：0111 1111 1111 1111 1111 1111 1111 1111
* INT_MIN的二进制表示为：1000 0000 0000 0000 0000 0000 0000 0000
* int型最高为为符号位，0正1负。CPU眼中只有数值，INT_MAX + 1 == INT_MIN。
* unsigned int最大表示： 1111 1111 1111 1111 1111 1111 1111 1111
* unsigned int可以防止转换为正数时溢出。
* 当unsigned int值为-INT_MIN(2147483648)再左移会溢出。
* \+ 比 << 优先级要高，详见：http://www.jb51.net/article/37282.htm

```cpp
class Solution {
public:
    int divide(int dividend, int divisor) {
        long long a = dividend>=0 ? dividend : -(long long)dividend;
        long long b = divisor>=0  ? divisor  : -(long long)divisor;
        long long r = 0; int i = 0;
        while (b<<1 < a) { i++; b<<=1; };
        while (a>=b || i>0) {
            a -= b; r += 1<<i;
            while (a < b && i > 0) { i--; b>>=1; }
        }
        if ((dividend^divisor)>>31) r = -r;
        else if (r > INT_MAX)       r = INT_MAX;
        return r;
    }
};
```

```cpp
class Solution {
public:
    int divide(int dividend, int divisor) {
        unsigned int a = dividend>=0 ? dividend : -dividend;
        unsigned int b = divisor>=0  ? divisor  : -divisor;
        int r=0; long long bb; int i;
        while (a >= b) for (i=0, bb=b; a >= bb; i++, bb<<=1) {
            a -= bb; r += 1<<i;
        }
        if      ((dividend^divisor)>>31)     r = -r;
        else if ((unsigned int)r > INT_MAX)  r = INT_MAX;
        return r;
    }
};
```

## 30. Substring with Concatenation of All Words 
链接：https://leetcode.com/problems/substring-with-concatenation-of-all-words/#/description

思路：用的map，耗时206ms，超过45%

```cpp
class Solution {
public:
    vector<int> findSubstring(const string& s, vector<string>& words) {
        map<string, int> tbl; map<string, int> vst;
        vector<int> result;
        if (words.empty()) return result;
        int lenW = words[0].length(), lenT = words.size() * lenW, lenS = s.length();
        if (s.size() < lenT) return result;
        for (auto &word : words) tbl[word] = tbl.find(word) == tbl.end() ? 1 : tbl[word]+1;
        for (int i = 0; i <= lenS - lenT; i++) {
            string tar = s.substr(i, lenW);
            if (tbl.find(tar) != tbl.end()) {
                int fnd = 1;
                vst[tar] = 1;
                for (int j = lenW; j < lenT; j += lenW) {
                    tar = s.substr(i+j, lenW);
                    if (tbl.find(tar) != tbl.end()) {
                        vst[tar] = vst.find(tar) == vst.end() ? 1 : vst[tar]+1;
                        if (tbl[tar] >= vst[tar]) fnd++; else break;
                    } else break;
                }
                vst.clear();
                if (fnd == words.size()) result.push_back(i);
            }
        }
        return result;
    }
};
```

用unorderd_map试试，好像还慢了一些，299ms，击败24%

```cpp
class Solution {
public:
    vector<int> findSubstring(const string& s, vector<string>& words) {
        unordered_map<string, int> tbl; unordered_map<string, int> vst;
        vector<int> result;
        if (words.empty()) return result;
        int lenW = words[0].length(), lenT = words.size() * lenW, lenS = s.length();
        if (s.size() < lenT) return result;
        for (auto &word : words) tbl[word] = tbl.find(word) == tbl.end() ? 1 : tbl[word]+1;
        for (int i = 0; i <= lenS - lenT; i++) {
            string tar = s.substr(i, lenW);
            if (tbl.find(tar) != tbl.end()) {
                int fnd = 1;
                vst[tar] = 1;
                for (int j = lenW; j < lenT; j += lenW) {
                    tar = s.substr(i+j, lenW);
                    if (tbl.find(tar) != tbl.end()) {
                        vst[tar] = vst.find(tar) == vst.end() ? 1 : vst[tar]+1;
                        if (tbl[tar] >= vst[tar]) fnd++; else break;
                    } else {
                        break;
                    }
                }
                vst.clear();
                if (fnd == words.size()) result.push_back(i);
            }
        }
        return result;
    }
};
```

用unordered_multiset，直接超时。

```cpp
class Solution {
public:
    vector<int> findSubstring(const string& s, vector<string>& words) {
        unordered_multiset<string> tbl; unordered_multiset<string> vst;
        vector<int> result;
        if (words.empty()) return result;
        int lenW = words[0].length(), lenT = words.size() * lenW, lenS = s.length();
        if (s.size() < lenT) return result;
        for (auto &word : words) tbl.insert(word);
        for (int i = 0; i <= lenS-lenT; i++) {
            string tar = s.substr(i, lenW);
            if (tbl.find(tar) != tbl.end()) {
                int fnd = 1;
                vst.insert(tar);
                for (int j = lenW; j < lenT; j += lenW) {
                    tar = s.substr(i+j, lenW);
                    if (tbl.find(tar) != tbl.end() && tbl.count(tar) > vst.count(tar)) {
                        vst.insert(tar);
                        fnd++;
                    } else {
                        break;
                    }
                }
                vst.clear();
                if (fnd == words.size()) result.push_back(i);
            }
        }
        return result;
    }
};
```

貌似还有O(n)的解法，没想出来。

## 31. Next Permutation
链接：https://leetcode.com/problems/next-permutation/
思路：
1. 从尾部开始找到第一个不满足逆序序列的元素i
2. 在逆序序列中找到最接近的较小元素j
3. 交换i和j，并将逆序序列反置

```cpp
class Solution {
public:
    void nextPermutation(vector<int>& nums) {
        int i=0, j=0, len=nums.size();
        i=len-2; while (i>=0 && nums[i+1]<=nums[i]) --i;
        if (i == -1) { reverse(nums.begin(), nums.end()); return; }
        j=len-1; while (j>i && nums[i]>=nums[j]) --j;
        swap(nums[i], nums[j]);
        reverse(nums.begin()+i+1, nums.end());
    }
};
```

## 32. Longest Valid Parentheses
链接：https://leetcode.com/problems/longest-valid-parentheses/
思路：动态规划，有效的括号序列是由多个单有效括号序列组成的：(...)(...)(...) 。设f[i]表示以i为结束位置的最长有效括号序列的长度，则当[l,i]有效时f[i] = f[l-1] + i-l+1

```cpp
class Solution {
public:
    int longestValidParentheses(string& s) {
        stack<int> t; vector<int> f(s.length(), 0); int result = 0;
        for (int i=0; i<s.length(); ++i) {
            if (s[i] == '(') t.push(i);
            else if (!t.empty()) {
                int l = t.top(); t.pop();
                f[i] = i - l + 1 + (l>0 ? f[l-1] : 0);
                result = max(result, f[i]);
            }
        }
        return result;
    }
};
```

## 33. Search in Rotated Sorted Array 
链接：https://leetcode.com/problems/search-in-rotated-sorted-array/#/description

```cpp
class Solution {
public:
    int search(vector<int>& nums, int target) {
        int l = 0, h = nums.size();
        while (l < h) {
            int m = (l + h) / 2;
            if (nums[m] < target) {
                if (nums[m] >= nums[0]) l = m + 1; 
                else { if (target >= nums[0]) h = m; else l = m + 1; }
            } else if (nums[m] > target) {
                if (nums[m] >= nums[0]) { if (target >= nums[0]) h = m; else l = m + 1; }
                else h = m;
            } else {
                return m;
            }
        }
        return -1;
    }
};
```

```
class Solution {
public:
    int search(vector<int>& nums, int target) {
        int l = 0, h = nums.size();
        while (l < h) {
            int m = (l + h) / 2;
            if (nums[m] < target) {
                if (nums[m] <  nums[0] && target >= nums[0]) h = m;     else l = m + 1;
            } else if (nums[m] > target) {
                if (nums[m] >= nums[0] && target <  nums[0]) l = m + 1; else h = m;
            } else {
                return m;
            }
        }
        return -1;
    }
};
```

## 34. Search for a Range
链接： https://leetcode.com/problems/search-for-a-range/#/description

```cpp
class Solution {
public:
    vector<int> searchRange(vector<int>& nums, int target) {
        int s = findStartPosition(nums, target);
        if (s == -1) return {-1, -1};
        int e = findEndPosition(nums, target);
        return vector<int>({s, e});
    }
    
    int findStartPosition(vector<int>& nums, int target) {
        int l = 0, h = nums.size(), m, result = -1;
        while (l < h) {
            m = (l + h) / 2;
            if      (nums[m] < target)  { l = m + 1; }
            else if (nums[m] > target)  { h = m; }
            else                        { h = m; result = m; }
        }
        return result;
    }
    
    int findEndPosition(vector<int> &nums, int target) {
        int l = 0, h = nums.size(), m, result = -1;
        while (l < h) {
            m = (l + h) / 2;
            if      (nums[m] < target)  { l = m + 1; }
            else if (nums[m] > target)  { h = m; }
            else                        { l = m + 1; result = m; }
        }
        return result;
    }
};
```

## 35. Search Insert Position
链接：https://leetcode.com/problems/search-insert-position/#/description

```cpp
class Solution {
public:
    int searchInsert(vector<int>& nums, int target) {
        int l = 0, h = nums.size(), m;
        while (l < h) {
            m = (l + h) / 2;
            if (nums[m] < target) {
                l = m + 1;
            } else if (nums[m] > target) {
                h = m;
            } else {
                break;
            }
        }
        return (l + h) / 2;
    }
};
```

## 36 Valid Sudoku
题目： https://leetcode.com/problems/valid-sudoku/description/
思路：分别按行、列、3x3遍历，看是否满足无重复的数字即可，9 ms。

```cpp
class Solution {
public:
    bool isValidSudoku(vector<vector<char>>& board) {
        for (int row = 0; row < 9; row++) 
            if (! isValidRowOfSudoku(board, row)) return false;
        for (int col = 0; col < 9; col++) 
            if (! isValidColOfSudoku(board, col)) return false;
        for (int row = 0; row < 9; row += 3) for (int col = 0; col < 9; col += 3) 
            if (! isVaild3x3OfSudoku(board, row, col)) return false;
        return true;
    }
    
    bool isValidRowOfSudoku(vector<vector<char>>& board, int row) {
        char tbl[9]; fill_n(tbl, 9, 1);
        for (int col = 0; col < 9; col++) {
            char ch = board[row][col];
            if      (ch == '.')   continue;
            else if (tbl[ch-'1']) tbl[ch-'1'] = 0;
            else                  return false;
        }
        return true;
    }
 
    bool isValidColOfSudoku(vector<vector<char>>& board, int col) {
        char tbl[9]; fill_n(tbl, 9, 1);
        for (int row = 0; row < 9; row++) {
            char ch = board[row][col];
            if      (ch == '.')   continue;
            else if (tbl[ch-'1']) tbl[ch-'1'] = 0;
            else                  return false;
        }
        return true;
    }
    
    bool isVaild3x3OfSudoku(vector<vector<char>>& board, int row, int col) {
        char tbl[9]; fill_n(tbl, 9, 1);
        for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) {
            char ch = board[row+i][col+j];
            if      (ch == '.')   continue;
            else if (tbl[ch-'1']) tbl[ch-'1'] = 0;
            else                  return false;
        }
        return true;
    }
};
```

思路2：根据上面的答案，将三个for循环缩小为一个，19ms，目前看来循环分开效率要高一些。

```cpp
class Solution {
public:
    bool isValidSudoku(vector<vector<char>>& board) {
        for (int i = 0; i < 9; i++) {
            char row[9] = {0}, col[9] = {0}, blk[9] = {0};
            for (int j = 0; j < 9; j++) {
                char ch;
                ch = board[i][j];
                if (ch != '.') if (!row[ch-'1']) row[ch-'1'] = 1; else return false;
                ch = board[j][i];
                if (ch != '.') if (!col[ch-'1']) col[ch-'1'] = 1; else return false;
                ch = board[i/3*3+j/3][i%3*3+j%3];
                if (ch != '.') if (!blk[ch-'1']) blk[ch-'1'] = 1; else return false;
            }
        }
        return true;
    }
};
```

## 37. Sodoku Solver
链接：https://leetcode.com/problems/sudoku-solver/description/
思路：广搜，按位置遍历，搜索'.'所有可能值，然后递归搜索下去，看看是否能够得出可能解，9ms。

```cpp
class Solution {
public:
    void solveSudoku(vector<vector<char>>& board) {
        BFS(board, 0, 0);
    }
    
    bool BFS(vector<vector<char>>& board, int i, int j) {
        if (i == 9)   return true;
        if (board[i][j] != '.') return BFS(board, i=i+(j+1)/9, (j+1)%9);
        
        char tbl[9] = {0}, ch;
        for (int k = 0; k < 9; k++) {
            ch = board[i][k]; if (ch != '.') tbl[ch-'1'] = 1;
            ch = board[k][j]; if (ch != '.') tbl[ch-'1'] = 1;
            ch = board[i/3*3+k/3][j/3*3+k%3];
            if (ch != '.') tbl[ch-'1'] = 1;
        }

        for (int k = 0; k < 9; k++) {
            if (tbl[k] == 1) continue;
            board[i][j] = '1' + k;
            if (BFS(board, i, j)) return true;
        }
        board[i][j] = '.';
        return false;
    }
};
```

## 38. Count and Say
链接：https://leetcode.com/problems/count-and-say/description/
思路：打表记录当前到n的所有可能，时间换空间。转换的时候需要考虑细节情况。

```cpp
class Solution {
public:
    string countAndSay(int n) {
        static vector<string> tbl = {"1", "11", "21", "1211", "111221"};
        for (int i = tbl.size(); i < n; i++) {
            tbl.push_back(next(tbl.back()));
        }
        return tbl[n-1];
    }
    
    string next(string& curr) {
        int cnt = 1; char ch = curr[0]; string next;
        for (int i = 1; i <= curr.length(); i++) {
            if (i < curr.length() && curr[i] == curr[i-1]) { 
                cnt++; 
            } else if (i < curr.length()) {
                next.append(to_string(cnt)).push_back(ch); 
                cnt = 1; ch = curr[i];
            } else { 
                next.append(to_string(cnt)).push_back(ch);  
            }
        }
        return move(next);
    }
};
```

## 39. Combination Sum
链接：https://leetcode.com/problems/combination-sum
思路：广度搜索，$f(n, [a,b,c]) = f(n-a, [a,b,c]) + f(n-b, [b,c]) + f(n-c, [c])$。注意去重，所有包含a的答案，已经在$f[n-a, [a,b,c]]$中包含了，所以$f(n-b,[b-c])$从b开始。
```cpp
class Solution {
public:
    vector<vector<int>> combinationSum(vector<int>& candidates, int target) {
        vector<vector<int>> result; vector<int> path;
        sort(begin(candidates), end(candidates));
        DFS(candidates, 0, target, path, result);
        return move(result);
    }
    
    void DFS(const vector<int>& candidates, int lo, int target, vector<int>& path, vector<vector<int>>& result) {
        if (target == 0) { result.push_back(path); return; }
        for (int i = lo; i < candidates.size(); i++) {
            int candidate = candidates[i];
            if (target < candidate) return;
            path.push_back(candidate);
            DFS(candidates, i, target - candidate, path, result);
            path.pop_back();
        }
    }
};
```

## 40. Combination Sum II
链接：https://leetcode.com/problems/combination-sum-ii/description/
思路：先排序，后深搜，注意去重。关键是在path选取第n个数字时，保证唯一。

```cpp
class Solution {
public:
    vector<vector<int>> combinationSum2(vector<int>& candidates, int target) {
        vector<vector<int>> result; vector<int> path;
        sort(begin(candidates), end(candidates));
        DFS(candidates, target, 0, path, result);
        return result;
    }
    
    void DFS(vector<int>& nums, int target, int curr, vector<int> &path, vector<vector<int>> &result) {
        if (target <  0) return;
        if (target == 0) { result.push_back(path); return; }
        for (int i=curr; i<nums.size(); i++) {
            if (i>curr && nums[i]==nums[i-1]) continue;
            path.push_back(nums[i]);
            DFS(nums, target-nums[i], i+1, path, result);
            path.pop_back();
        }
    }
};
```