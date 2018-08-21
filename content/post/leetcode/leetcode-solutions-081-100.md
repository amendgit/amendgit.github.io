---
title: "Leetcode Solutions 081 100"
date: 2018-08-21T14:21:12+08:00
tags: ["leetcode"]
---

本文用来记录我的leetcode解题报告，用于之后的代码优化、查阅和复习。这里是第81题到第100题。

<!--more-->

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