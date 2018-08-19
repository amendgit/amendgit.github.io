---
title: "Leetcode Solutions 041 060"
date: 2018-08-01T11:33:16+08:00
tags: ["leetcode"]
---

本文用来记录我的leetcode解题报告，用于之后的代码优化、查阅和复习。这里是第41题到第60题。

<!--more-->

## 41. First Missing Positive
链接：https://leetcode.com/problems/first-missing-positive/
思路：遍历数组将数值x放置到对应的下标x-1上，然后遍历找到第一个无序的数字，返回下标加1。

```cpp
class Solution {
public:
    int firstMissingPositive(vector<int>& nums) {
        int len=nums.size(), i=0;
        while(i < len) {
            if (nums[i]>0 && nums[i]<=len) swap(nums[nums[i]-1], nums[i]);
            if (nums[i]<=0 || nums[i]>len || nums[nums[i]-1]==nums[i]) ++i;
        }
        for (i=0; i<len; ++i) if (nums[i] != i+1) break;
        return i+1;
    }
};
```

```cpp
class Solution {
public:
    int firstMissingPositive(vector<int>& nums) {
        int len=nums.size(), i=0; vector<int> sort(len, 0);
        for (i=0; i<len; i++) if (nums[i]>0 && nums[i]<=len) sort[nums[i]-1] = nums[i];
        for (i=0; i<len; ++i) if (sort[i] != i+1) break;
        return i+1;
    }
};
```

## 42. Trapping Rain Water
链接：https://leetcode.com/problems/trapping-rain-water/discuss/
思路：用栈来跟踪两个能存水的bar的索引。
* 当栈顶比当前bar的要矮时，说明还没遇到合适的bar，先压栈。
* 当栈顶比当前的bar要高时，则弹出栈顶，一直到和当前的bar略高的元素。
    * 如果某次出栈以后，栈为空，则说明cur要比栈里所有的bar都高，可以计算两者之间的高度了。
* 如果一直遇不到更高的bar，说明栈里存在更高的元素，但是可以通过计算栈里相邻的元素之间的水量来计算总水量。

```cpp
class Solution {
public:
    int trap(vector<int>& heights) {
        int cur = 0, low = 0, sum = 0;
        stack<int> s; s.push(cur++);
        while (cur < heights.size()) {
            if      (heights[s.top()] >  heights[cur]) { s.push(cur++); }
            else if (heights[s.top()] <= heights[cur]) { 
                low = s.top(); s.pop(); 
                if (s.empty()) { 
                    (cur - low > 1) && (sum += trap(heights, low, cur)); 
                    s.push(cur++); 
                }
            }
        }
        
        cur = s.top(); s.pop();
        while (!s.empty()) {
            low =  s.top(); s.pop();
            sum += trap(heights, low, cur);
            cur =  low;
        }
        
        return sum;
    }
    
    int trap(vector<int>& heights, int low, int cur) {
        int sum = 0; int bar = min(heights[low], heights[cur]);
        for (int i = low+1; i < cur; i++)  sum += bar - heights[i];
        return sum;
    } 
};
```

看discuss里，还有个非常巧妙的思路。可以这么理解，假设在最右侧存在一个最高的bar，比数组里所有的都要高。那么计算的过程就变得很简单了，只要从左向右扫描，然后记录目前扫描过程中遇到的最高的bar。然后，与当前扫描的bar相减就得到了想要的数了。

把这个思路略微变化一下，只需要确定前面存在比当前扫描的bar都更高的bar，然后记录一个扫描过程中遇到的最高的bar。就可以通过相减的方式计算最终值了。而确保前面存在更高的bar的方式，是通过左右来回切换扫描，然后扫过更小的那个值，就能保证对面存在更高的bar值了。然后，只要用一个bar记录当前两端夹逼过程中较小的那一侧的bar值就可以了。

```cpp
class Solution {
public:
    int trap(vector<int>& heights) {
        int L = 0, R = heights.size()-1, bar = 0, sum = 0;
        while (L < R) {
            int cur = heights[heights[L] < heights[R] ? L++ : R--];
            bar = max(bar, cur);
            sum += bar - cur;
        }
        return sum;
    }
};
```

还有一种理解简单的思路，先找到一个最高点，然后分别从左边和从右边向中间扫描，这个时候只要记住一个扫描遇到的最高的bar，就是存水的较短的那个bar。

```cpp
class Solution {
public:
    int trap(vector<int>& heights) {
        int peak = 0, sum = 0;
        for (int i=0; i<heights.size(); i++) 
            if (heights[i] > heights[peak]) peak = i;
        for (int i=0, bar=0; i<peak; i++) {
            bar = max(heights[i], bar);
            sum += bar - heights[i];
        }
        for (int i=heights.size()-1, bar=0; i>peak; i--) {
            bar = max(heights[i], bar);
            sum += bar - heights[i];
        }
        return sum;
    }
};
```


## 43. Multiply Strings
链接：https://leetcode.com/problems/multiply-strings/
思路：细节考察，模拟乘法的计算过程。

```cpp
class Solution {
public:
    string multiply(string num1, string num2) {
        if (num1=="0" || num2=="0") return "0";
        int carry=0, len1=num1.length(), len2=num2.length(); string result;
        for (int i=len1-1; i>=0; --i) {
            int k=len1-1-i, j=len2-1;
            while (j>=0 || carry) {
                int tmp = 0;
                if (j>=0) tmp = (num1[i]-'0') * (num2[j]-'0');
                if (k>=result.length()) result.push_back('0');
                tmp += carry+result[k]-'0'; carry = tmp/10; tmp = tmp%10;
                result[k++] = tmp + '0';
                --j;
            } 
        }
        reverse(result.begin(), result.end());
        return move(result);
    }
};
```

```cpp
class Solution {
public:
    string multiply(string num1, string num2) {
        if (num1=="0" || num2=="0") return "0";
        reverse(num1.begin(), num1.end()); reverse(num2.begin(), num2.end());
        int carry=0; string result(num1.size()+num2.size(), '0');
        for (int i=0; i<num1.length(); ++i) {
            int k=i, j=0;
            while (j<num2.length() || carry) {
                int tmp = 0;
                if (j<num2.length()) tmp = (num1[i]-'0') * (num2[j]-'0');
                tmp += carry+result[k] - '0'; carry = tmp/10; tmp = tmp%10;
                result[k++] = tmp + '0';
                j++;
            } 
        }
        reverse(result.begin(), result.end());
        int l=0; while (result[l]=='0') l++;
        return result.substr(l);
    }
};
```

## 44. Wildcard Matching

用深搜，提示超时：

```c++
class Solution {
public:
    bool isMatch(string s, string p) {
        const int slen = s.length(), plen = p.length();
        if ((slen==0 && plen==0) || (slen==0 && p=="*")) return true;
        return DFS(s, 0, p, 0);
    }
    
    bool DFS(const string& s, int si, const string& p,int pi) {
        if (si == s.length() && pi == p.length()) return true;
        if (pi == p.length()) return false;
        if (p[pi] == '*') for (int i=0; si+i<=s.length(); i++) if (DFS(s, si+i, p, pi+1)) return true; 
        if ((p[pi] == '?' || s[si] == p[pi]) && DFS(s, si+1, p, pi+1)) return true;
        return false;
    }
};
```

用动态规划的思路，dp[i+1][j+1] <=> s[0,i] =~ p[0,j]。
那么，dp[i+1][j+1]可以通过以下来推导：

* dp[i][j] 且 ( s[i]==p[j] 或 p[j] == '\*' 或 p[j] == '?' )
* (dp[i+1][j] 或 dp[i][j+1]) 或 p[j] == '\*'

其中部分二的情况，其实是在1中覆盖了：如 `[acc, a*c]`当处于`[acc, a*]`时，2的推导不满足，因为p[j]不等于`*`，但是在`[ac, a*]`时，满足1的情况。只要有一个推导能推导出来就可以了。

初始时，需要初始化掉第一行和第一列的值。

dp[0][0]，由题意[ , ]结果是1.
第一行：当p始终为"\*"时，值就会为1，如：`[ , *]`、`[ , **]`、`[ , ***]`
第一列：空的p匹配任何非空的s，结果都是0

```cpp
class Solution {
public:
    bool isMatch(string s, string p) {
        const int slen = s.length(), plen = p.length();
        vector<vector<int>> dp(slen+1, vector<int>(plen+1, 0)); dp[0][0]   = 1;
        for (int i=0; i<slen; i++)                              dp[i+1][0] = 0;
        for (int j=0; j<plen; j++) if (p[j]=='*' && dp[0][j])   dp[0][j+1] = 1;
        for (int i=0; i<slen; i++) for (int j=0; j<plen; j++) {
            dp[i+1][j+1] = dp[i][j] && (s[i]==p[j] || p[j]=='?' || p[j]=='*');
            dp[i+1][j+1] = dp[i+1][j+1] || p[j]=='*' && (dp[i+1][j] || dp[i][j+1]);
        }
        return dp[slen][plen];
    }
};
```

稍微做点优化，因为dp默认值是0，所以很多操作可以不要：

```cpp
class Solution {
public:
    bool isMatch(string s, string p) {
        const int slen = s.length(), plen = p.length();
        vector<vector<int>> dp(slen+1, vector<int>(plen+1, 0)); dp[0][0] = 1;
        for (int j=0; j<plen; j++) if (p[j]=='*') dp[0][j+1] = 1; else break;
        for (int i=0; i<slen; i++) for (int j=0; j<plen; j++) {
            dp[i+1][j+1] = dp[i][j] && (s[i]==p[j] || p[j]=='?' || p[j]=='*');
            dp[i+1][j+1] = dp[i+1][j+1] || p[j]=='*' && (dp[i+1][j] || dp[i][j+1]);
        }
        return dp[slen][plen];
    }
};
```

## 45. Jumps Game II

link: https://leetcode.com/problems/jump-game-ii/description/

1.45%，很慢的一个解决方法。

```cpp
class Solution {
public:
    int jump(vector<int>& nums) {
        if (nums.size() == 1) return 0;
        int last = nums.size()-1, jumps = 0, i = 0;
        while (true) {
            for (i=0; i<last; i++) if (nums[i] + i >= last) break;
            last = i, jumps++;
            if (i == 0) return jumps;
        }
        return jumps;
    }
};
```

discuss里有一个很快的办法，单次循环。nums[0] + 0，是第一步能到的最远的位置。在(0, nums[0]+0]区间内，都是第一步能reach的位置，计算这中间的第i2个元素跳的最远，位置是nums[i2]+i2，那么在(nums[0]+0, nums[i2]+i2]，是第二步能跳到的区间。再继续下去，就能计算最终到达的位置。

```cpp
class Solution {
public:
    int jump(vector<int>& nums) {
        int jumps = 0, reached = 0, reaching = 0;
        for (int i = 0; i < nums.size()-1; i++) {
            reaching = max(reaching, i + nums[i]);
            if (i == reached) {
                jumps++;
                reached = reaching;
            }
        }
        return jumps;
    }
};
```

## 46. Permutations
链接：https://leetcode.com/problems/permutations/description/
思路：f(0, [1,2,3]) = f(1, [1,2,3]) + f(1, [2,1,3]) + f(1, [3,2,1]); 递归求解即可，即[1,2,3]的全排列，等于将各个元素分别提到第一个，然后和剩余元素全排列的结果拼接在一起。

```cpp
class Solution {
public:
    vector<vector<int>> permute(vector<int>& nums) {
        vector<vector<int>> result;
        dfs(nums, 0, result);
        return result;
    }
    
    void dfs(vector<int>& nums, int p, vector<vector<int>>& result) {
        if (p == nums.size()-1) { result.push_back(nums); return; }
        for (int i=p; i<nums.size(); i++) {
            swap(nums[p], nums[i]);
            dfs(nums, p+1, result);
            swap(nums[p], nums[i]);
        }
    }
};
```

## 47. Permutations II
链接：https://leetcode.com/problems/permutations/description/
思路：和46题比起来，多了重复的数字，因此要加上去重的逻辑，已经swap过的值就不再swap了。
f(0, [1,1,2]) = f(1, [1,1,2]) + ~~f(1, [1,1,2])~~ + f(1, [1,2,1]) 
注意，要用哈希表，例如 [1,1,2,2]这样的情况，在f(0,...) 到 f(1, ...) 递推的过程中，不仅仅1不能和1交换，1也不能和2交换。


```cpp
class Solution {
public:
    vector<vector<int>> permuteUnique(vector<int>& nums) {
        vector<vector<int>> result;
        dfs(nums, 0, result);
        return result;
    }
    
    void dfs(vector<int>& nums, int p, vector<vector<int>>& result) {
        if (p == nums.size()-1) { result.push_back(nums); return; }
        set<int> tbl;
        for (int i=p; i<nums.size(); i++) {
            if (tbl.insert(nums[i]).second == false) continue;
            swap(nums[i], nums[p]);
            dfs(nums, p+1, result);
            swap(nums[i], nums[p]);
        }
    }
};
```

先排序，后去重，不知道哪出问题了

```cpp
class Solution {
public:
    vector<vector<int>> permuteUnique(vector<int>& nums) {
        vector<vector<int>> result;
        sort(begin(nums), end(nums));
        dfs(nums, 0, result);
        return result;
    }
    
    void dfs(vector<int>& nums, int p, vector<vector<int>>& result) {
        if (p == nums.size()-1) { result.push_back(nums); return; }
        for (int i=p; i<nums.size(); i++) {
            if (i>p && nums[i]==nums[i-1]) continue;
            swap(nums[i], nums[p]);
            dfs(nums, p+1, result);
            swap(nums[i], nums[p]);
        }
    }
};
```

## 48. Rotate Image
链接：https://leetcode.com/problems/rotate-image/description/
思路：先沿着主对角线交换，再沿着中轴线左右交换一次，注意不要重复交换了。

```cpp
class Solution {
public:
    void rotate(vector<vector<int>>& mat) {
        if (mat.empty()) return;
        const int ROW = mat.size(), COL = mat[0].size();
        for (int i=0; i<ROW; i++) for (int j=i+1; j<COL; j++) 
            swap(mat[i][j], mat[j][i]);
        for (int i=0; i<ROW; i++) for (int j=0; j<COL/2; j++) 
            swap(mat[i][j], mat[i][COL-1-j]);
    }
};
```

## 49. Group Anagrams
链接：https://leetcode.com/problems/group-anagrams/description/
思路：将单词按字母排序后值作为key，然后挂到哈希表下的vector中，最终取出来即可。

```cpp
class Solution {
public:
    vector<vector<string>> groupAnagrams(vector<string>& strs) {
        unordered_map<string, vector<string>> hashmap;
        for (auto &str : strs) { 
            auto tmp = str; sort(begin(tmp), end(tmp)); 
            hashmap[tmp].push_back(str); 
        }
        vector<vector<string>> result;
        for (auto &pair : hashmap) result.push_back(pair.second);
        return move(result);
    }
};
```

## 50. Pow(x, n)

emmm 超时了，300 / 304 test cases passed，下面这个case超时了

```
Last executed input:
1.00000
-2147483648
```

讲道理不应该啊。。。

```cpp
class Solution {
public:
    double myPow(double x, int n) {
        double result = 1.f;
        int save = n;
        n = abs(n);
        while (n != 0) {
            (n & 1 != 0) && (result *= x);
            x *= x; n >>= 1;
        }
        if (save < 0) result = 1.f / result;
        return result;
    }
};
```

好吧，找到原因了，INT_MIN的abs。。。还是INT_MIN.

```
INT_MAX: 2147483647
INT_MIN: -2147483648
abs(INT_MIN): -2147483648
```

换成下面的可以了

```cpp
class Solution {
public:
    double myPow(double x, int n) {
        double result = 1.f;
        long l = abs((long)n);
        while (l != 0) {
            (l & 1 != 0) && (result *= x);
            x *= x; l >>= 1;
        }
        if (n < 0) result = 1.f / result;
        return result;
    }
};
```

## 51. N-Queens

链接：https://leetcode.com/problems/n-queens/description/
思路：queen所在的位置的水平、竖直、45度斜线、-45度斜线上不能存在另外一个queen。因此，我们逐行搜索，深度优先。用vertical记录第c列是否有其他queen存在、diag记录同45度上是否有其他queen存在，antidiag记录-45度斜线上是否存在其他queen。

有一个小技巧：在同一个45度线上的，r+c的值相同。在同一个-45度上的，可以通过水平翻转，[r+c] -> [r,n-1-c]，然后他们的家和也是相同的了。

```cpp
class Solution {
public:
    vector<vector<string>> solveNQueens(int n) {
        vector<vector<string>> result; vector<int> path(n, 0);
        vector<int> vertical(n, 0), diag(n+n, 0), antidiag(n+n, 0);
        dfs(0, n, path, vertical, diag, antidiag, result);
        return result;
    }

    void dfs(int r, int n, vector<int>& path, vector<int>& vertical, vector<int>& diag, 
            vector<int>& antidiag, vector<vector<string>>& result) {
        if (r == n) {
            vector<string> ans(n, string(n, '.')); for (int i=0; i<n; i++) ans[i][path[i]] = 'Q';
            result.push_back(ans);
            return;
        }
        for (int c=0; c<n; c++) {
            if (vertical[c] || diag[r+c] || antidiag[r+n-1-c]) continue;
            vertical[c] = diag[r+c] = antidiag[r+n-1-c] = 1;
            path[r] = c;
            dfs(r+1, n, path, vertical, diag, antidiag, result);
            path[r] = 0;
            vertical[c] = diag[r+c] = antidiag[r+n-1-c] = 0;
        }
    }
};
```

## 52. N-Queens II

链接：https://leetcode.com/problems/n-queens-ii/description/

思路：类似51题，区别在于统计不同ans的数量，感觉还简单一些。

```cpp
class Solution {
public:
    int totalNQueens(int n) {
        vector<int> vertical(n, 0), diag(n+n, 0), antidiag(n+n, 0); int total = 0;
        dfs(0, n, vertical, diag, antidiag, total);
        return total;
    }
    
    void dfs(int r, int n, vector<int>& vertical, vector<int>& diag, vector<int>& antidiag, int& total) {
        if (r == n) { total++; return; }
        for (int c=0; c<n; c++) {
            if (vertical[c] || diag[r+c] || antidiag[r+n-1-c]) continue;
            vertical[c] = diag[r+c] = antidiag[r+n-1-c] = 1;
            dfs(r+1, n, vertical, diag, antidiag, total);
            vertical[c] = diag[r+c] = antidiag[r+n-1-c] = 0;
        }
    }
};
```

## 53. MaxSubArray
链接: https://leetcode.com/problems/maximum-subarray/description/
思路：动态规划
last(i)表示在nums[0...i]中，尾部加和的最大值。
maxsum(i)表示在nums[0...i]中，最大子数组的和的值。

last(i) = max(last(i-1)+nums(i), nums(i)) , 只有当加上nums[i]后，值变得更小了，才会将last(i)更新为nums[i]
maxsum(i) = max(maxsum(i-1), last(i))

要注意一下，子数组是否允许为空。

```cpp
class Solution {
public:
    int maxSubArray(vector<int>& nums) {
        if (nums.empty()) return 0;
        int maxsum=nums[0], last=nums[0];
        for (int i=1; i<nums.size(); i++) {
            last += nums[i];
            last = max(last, nums[i]);
            maxsum = max(maxsum, last);
        }
        return maxsum;
    }
};
```

## 54. Spiral Matrix
链接：https://leetcode.com/problems/spiral-matrix/
用(rb, cb, re, ce)表示矩阵的矩形区域，便利矩形边界和动态更新矩形区域的大小。在向左便利和向上便利的时候，需要判断前面的操作过程中有没有使得矩形失效。

```cpp
class Solution {
public:
    vector<int> spiralOrder(vector<vector<int>>& matrix) {
        vector<int> result;
        if (matrix.empty()) return result;
        int rb=0, cb=0, re=matrix.size()-1, ce=matrix[0].size()-1, i;
        result.reserve((re+1)*(ce+1)); 
        while (cb<=ce && rb<=re) {
            for (i=cb; i<=ce; ++i) result.push_back(matrix[rb][i]); rb++;
            for (i=rb; i<=re; ++i) result.push_back(matrix[i][ce]); ce--;
            if (rb <= re) for (i=ce; i>=cb; --i) result.push_back(matrix[re][i]); re--;
            if (cb <= ce) for (i=re; i>=rb; --i) result.push_back(matrix[i][cb]); cb++;
        }
        return move(result);
    }
};
```

## 55. Jump Game

链接：https://leetcode.com/problems/jump-game/description/

思路：计算每一格子能到达的最远位置，然后记录遍历以来能到达的最远位置，只要在最远位置内的都是可到达的点。然后判断最远位置是否能到达终点，如果可以则返回true，反之返回false。

```cpp
class Solution {
public:
    bool canJump(vector<int>& nums) {
        int reach=0, n=nums.size();
        for (int i=0; i<n && i<=reach; i++) {
            reach = max(i+nums[i], reach);
            if (reach >= n-1) return true;
        }
        return false;
    }
};
```

## 56.Merge Intervals

链接：https://leetcode.com/problems/merge-intervals/description/

思路：先排序，再遍历合并就可以了。

```cpp
/**
 * Definition for an interval.
 * struct Interval {
 *     int start;
 *     int end;
 *     Interval() : start(0), end(0) {}
 *     Interval(int s, int e) : start(s), end(e) {}
 * };
 */
class Solution {
public:
    vector<Interval> merge(vector<Interval>& I) {
        if (I.size() <= 1) return I;
        sort(begin(I), end(I), [](const Interval& l, const Interval& r) {
            return l.start < r.start;
        });
        Interval curr = I.front(), next; vector<Interval> result;
        for (int i=1; i<I.size(); i++) {
            next = I[i];
            if (curr.end >= next.start) { curr.end = max(curr.end, next.end);  }
            else                        { result.push_back(curr); curr = next; }
        }
        result.push_back(curr);
        return result;
    }
};
```

## 57. Insert Interval
链接：https://leetcode.com/problems/insert-interval/
思路：先找到与start相交的interval，然后开始合并，直到与end相交的interval，再放压入剩余的interval.

```cpp
/**
 * Definition for an interval.
 * struct Interval {
 *     int start;
 *     int end;
 *     Interval() : start(0), end(0) {}
 *     Interval(int s, int e) : start(s), end(e) {}
 * };
 */
class Solution {
public:
    vector<Interval> insert(vector<Interval>& I, Interval i) {
        vector<Interval> result; int p=0, len=I.size();
        while (p<len && I[p].end<i.start) result.push_back(I[p++]);
        while (p<len && I[p].start<=i.end) { 
            i.start = min(I[p].start, i.start); 
            i.end   = max(I[p].end  , i.end); 
            p++; 
        } 
        result.push_back(i);
        while (p < len) result.push_back(I[p++]);
        return move(result);
    }
};
```

## 58. Length of Last Word

链接：https://leetcode.com/problems/length-of-last-word/description/

思路：需要注意防止尾部有空格，没有其他要注意的了。

```cpp
class Solution {
public:
    int lengthOfLastWord(const string& s) {
        int result = 0; int p = s.length()-1;
        while (p >= 0 && isspace(s[p])) p--;
        while (p >= 0 && isalpha(s[p])) { p--; result++; }
        return move(result);
    }
};
```

## 59. Sprial Matrix II

链接：https://leetcode.com/problems/spiral-matrix-ii/description/

思路：设有四根线在上下左右四侧，遍历一条边之后，就将对应的边增加。

```cpp
class Solution {
public:
    vector<vector<int>> generateMatrix(int n) {
        vector<vector<int>> result(n, vector<int>(n, 0));
        int rb=0, re=n-1, cb=0, ce=n-1, p=1;
        for (int i=0; i<(n+1)/2; i++) {
            for (int c=cb; c<=ce; c++) result[rb][c] = p++; rb++;
            for (int r=rb; r<=re; r++) result[r][ce] = p++; ce--;
            for (int c=ce; c>=cb; c--) result[re][c] = p++; re--;
            for (int r=re; r>=rb; r--) result[r][cb] = p++; cb++;
        }
        return move(result);
    }
};
```

## 60. Permutation Sequence

链接：https://leetcode.com/problems/permutation-sequence/description/

思路：共有 n * (n-1) * (n-2) * ... * 1 个数，第一层可以分为n个(n-1)!，第一位分别以1 2 3 ... n，依次内推。因此，可以先计算出在哪个区间内，并将每一层数字追加到result里。

```cpp
class Solution {
public:
    string getPermutation(int n, int k) {
        int f = 1; vector<char> digits; string result; k = k - 1;
        for (int i=1; i<= n; i++) { f *= i; digits.push_back(i+'0'); }
        while (digits.size() > 1 && k != 0) {
            f /= digits.size();
            int i = k / f; k = k % f;
            result.push_back(digits[i]); 
            digits.erase(begin(digits)+i);
        }
        for (char d : digits) result.push_back(d);
        return result;
    }
};
```