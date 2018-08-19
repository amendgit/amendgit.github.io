---
title: "Leetcode Solutions 001-020"
date: 2018-06-01T15:34:45+08:00
tags: ["leetcode"]
---

本文用来记录我的leetcode解题报告，用于之后的代码优化、查阅和复习。这里是第1题到第20题。

<!--more-->

## 1. Two Sum

链接：https://leetcode.com/problems/two-sum/

思路1：暴力搜索，586ms

```cpp
class Solution {
public:
    vector<int> twoSum(vector<int>& nums, int target) {
        int i=0, j=i+1, len=nums.size();
        vector<int> result;
        for (int i=0; i<len-1; ++i) for (j=i+1; j<len; j++)
            if (nums[i]+nums[j] == target) { result = {i,j}; break; }
        return move(result);
    }
};
```

思路2：用hash，O(n)，用时13ms,注意如果把下面的unordered_map替换成map会耗时26ms。看来用hashtable实现的unordered_map比红黑树实现的map要快一倍多。

```cpp
class Solution {
public:
    vector<int> twoSum(vector<int>& nums, int target) {
        unordered_map<int, int> tbl; vector<int> result;
        for (int i=0; i<nums.size(); ++i) {
            int need = target-nums[i];
            if (tbl.find(need) != tbl.end()) { result = {tbl[need],i}; break;}
            tbl[nums[i]] = i;
        }
        return move(result);
    }
};
```

## 2. Add Two Numbers 

链接：https://leetcode.com/problems/add-two-numbers/

思路：细节考察题，见代码。

```cpp
class Solution {
public:
    ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {
        int carry = 0; ListNode result(0), *p = &result;
        while (l1 || l2 || carry) {
            if (l1) { carry += l1->val; l1 = l1->next; }
            if (l2) { carry += l2->val; l2 = l2->next; }
            ListNode *n = new ListNode(carry % 10); carry /= 10;
            p->next = n; p = p->next;
        }
        return result.next;
    }
};
```

## 3. Longest Substring Without Repeating Characters

链接：https://leetcode.com/problems/longest-substring-without-repeating-characters/

思路：使用t[256]计数。双指针l和i，i遍历直到有重复字符。计算距离i-l和最大值r。移动l消除重复字符。

```cpp
class Solution {
public:
    int lengthOfLongestSubstring(string& s) {
        if (s.empty()) return 0;
        int t[256], l=0, r=1, len=s.length();
        fill_n(t, 256, 0); t[s[0]]++;
        for (int i=1; i<len; ++i) {
            if (t[s[i]] == 0) 
                t[s[i]]++;
            else {
                r = max(r, i-l);
                while (s[l]!=s[i]) t[s[l++]]--; l++;
            }
        }
        return max(r, len-l);
    }
};
```

## 4. Median of Two Sorted Arrays   

链接：https://leetcode.com/problems/median-of-two-sorted-arrays/

思路：转化问题为找到两个有序数组中第k个数的问题。排除法，取两数组中第k/2(必须保证加起来和为k)大小的数字作比较，第k大的数字肯定不在较小的那个k/2序列中。将其下标l提升k/2，转化为在l1和l2起始位置开始找第k-k/2大小的数字。

```cpp
class Solution {
public:
    double findMedianSortedArrays(vector<int>& nums1, vector<int>& nums2) {
        int h1 = nums1.size(), h2 = nums2.size();
        if ((h1 + h2) % 2 != 0)
            return  kthNum(nums1, 0, nums2, 0, (h1 + h2)/2+1);
        else 
            return (kthNum(nums1, 0, nums2, 0, (h1 + h2)/2) 
                +   kthNum(nums1, 0, nums2, 0, (h1 + h2)/2+1)) / 2;
    }
    
    double kthNum(vector<int>& nums1, int l1, vector<int>& nums2, int l2, int k) {
        int h1 = nums1.size(), h2 = nums2.size();
        if (h1-l1 < h2-l2) return kthNum(nums2, l2, nums1, l1, k);
        if (h2-l2 <= 0)    return nums1[l1+k-1];
        if (k == 1)        return min(nums1[l1], nums2[l2]);
        int i2 = min(h2, l2+k/2); int i1 = l1+k-(i2-l2);
        if (nums1[i1-1] < nums2[i2-1]) return kthNum(nums1, i1, nums2, l2, k-(i1-l1));
        if (nums1[i1-1] > nums2[i2-1]) return kthNum(nums1, l1, nums2, i2, k-(i2-l2));
        return nums1[i1-1];
    }
};
```

## 5. Longest Palindromic Substring

链接：https://leetcode.com/problems/longest-palindromic-substring/

思路：动态规划，设f[i,j]表示s[i,j]是否是回文字符串，`f[i,j] = s[i]==s[j] && (i>=j-1 || f[i+1,j-1])`。进一步发现f[i,j]依赖f[i+1,j-1]，可以优化为一维表，即`f[i] = s[i]==s[j] && (i>=j-1 || f[i+1])`。

```cpp
class Solution {
public:
    string longestPalindrome(const string& s) {
        const int len = s.length(); int l=0, h=0;
        int f[len] = {0};
        for (int j=0; j<len; ++j) for (int i=0; i<=j; ++i) {
            f[i] = s[i]==s[j] && (i>=j-1 || f[i+1]);
            if (f[i] && (j-i>h-l)) { l=i; h=j; }
        }
        return move(s.substr(l, h-l+1));
    }
};
```

## 6. Zig Zag Conversion

链接：https://leetcode.com/problems/zigzag-conversion/

思路：映射索引到对应的row的位置，然后存储在不同的string中，最后合并起来。

```cpp
class Solution {
public:
    string convert(const string &s, int n) {
        if (n == 1 || n >= s.length()) return s;
        vector<string> buckets(n);
        // map form [0, 1, 2, 3, 4, 5 ...] to [0, 1, 2, 1, 0, 1 ...]
        for (int i=0; i<s.length(); ++i) 
            buckets[n-1-abs(i%(n*2-2)-n+1)].push_back(s[i]);
        string result; result.reserve(s.length());
        for (auto bucket : buckets) result.append(bucket);
        return move(result);
    }
};
```

## 7. Reverse Integer

链接：https://leetcode.com/problems/reverse-integer/

细节题，abcd，一次取d c b a，在加到result中去，result\*10 + tmp。要注意符号和溢出的处理，不需要单独处理INT\_MIN，因为INT\_MIN和INT\_MAX的reverse都不存在。

```cpp
class Solution {
public:
    int reverse(int x) {
        bool flag = false; if (x < 0) { flag = true; x = -x; } 
        long long result = 0;
        while (x > 0) {
            int tmp = x % 10; x = x / 10;
            result = result * 10 + tmp;
        }
        if (result > INT_MAX) return 0;
        if (flag) result = -result;
        return result;
    }
};
```

## 8. String to Integer (atoi)

链接：https://leetcode.com/problems/string-to-integer-atoi/

思路：细节考察题，注意：空格、符号、溢出问题。

```cpp
class Solution {
public:
    int myAtoi(const string &str) {
        int i=0, sign=1; long long result=0; const int len = str.length();
        // trim spaces.
        while (i<len && str[i]==' ') i++;
        // handle sign.
        if      (i<len && str[i]=='+') i++;
        else if (i<len && str[i]=='-') { sign = -1; i++; }
        // convert.
        while (i<len && result+INT_MIN<=0 && isdigit(str[i])) { 
            result = result*10 + str[i] - '0'; i++; 
        }
        // handle overflow.
        if (sign==+1 && result-INT_MAX>=0) return INT_MAX;
        if (sign==-1 && result+INT_MIN>=0) return INT_MIN;
        // return.
        return (int)result * sign;
    }
};
```
## 9. Palindrome Number

链接：https://leetcode.com/problems/palindrome-number/

思路：前后依次计算对应位置的值，然后前后对比。

```cpp
class Solution {
public:
    bool isPalindrome(int x) {
        if (x < 0) return false;
        int p=1, q=1;
        while (x / p >= 10) p *= 10;
        while (p >= q) {
            if ((x/p)%10 != (x/q)%10) return false;
            p = p / 10; q = q * 10;
        }
        return true;
    }
};
```
## 10. Regular Expression Matching

链接：https://leetcode.com/problems/regular-expression-matching/

思路：这题状态转换比较繁，我用图简单画了下。
![empty](leanote://file/getImage?fileId=5808e2341b6f9f2c0e000000)

```cpp
class Solution {
public:
bool isMatch(const string &s, const string &p) {
    const int slen = s.length(), plen = p.length();
    vector<vector<int> > f(slen+1, vector<int>(plen+1, 0));
    f[slen][plen] = 1;
    for (int j=plen-1; j>=1; --j) f[slen][j-1] = p[j] == '*' && f[slen][j+1];
    for (int i=slen-1; i>=0; --i) for (int j=plen-1; j>=0;) {
        if (s[i] == p[j] || p[j] == '.') { 
            f[i][j] = f[i+1][j+1]; j -= 1; 
        } 
        else if (p[j] == '*') {
            if (p[j-1] == '.' || s[i] == p[j-1]) { 
                f[i][j-1] = f[i][j+1] || f[i+1][j-1] || f[i+1][j+1]; j -= 2; 
            } else if (s[i] != p[j-1]) { 
                f[i][j-1] = f[i][j+1]; j -= 2; 
            }
        }
        else { j -= 1; }
    }
    return f[0][0];
}
};
```
## 11. Container With Most Water

链接：https://leetcode.com/problems/container-with-most-water/

思路：

```cpp
class Solution {
public:
    int maxArea(vector<int>& heights) {
        int result = 0;
        int i = 0, j = heights.size()-1;
        while (i < j) {
            int hi = heights[i], hj = heights[j];
            int area = min(hi, hj) * (j-i);
            result = max(area, result);
            if      (hi < hj) i++;
            else if (hi > hj) j--;
            else              i++, j--;
        }
        return result;
    }
};
```
## 12. Integer to Roman

链接：https://leetcode.com/problems/integer-to-roman/

```cpp
class Solution {
public:
    string intToRoman(int num) {
        static string I[] = {"", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX"};
        static string X[] = {"", "X", "XX", "XXX", "XL", "L", "LX", "LXX", "LXXX", "XC"};
        static string C[] = {"", "C", "CC", "CCC", "CD", "D", "DC", "DCC", "DCCC", "CM"};
        static string M[] = {"", "M", "MM", "MMM"};
        string result; result.reserve(20);
        result.append(M[num/1000]).append(C[num%1000/100]).append(X[num%100/10]).append(I[num%10]);
        return move(result);
    }
};
```
## 13. Roman To Integer

链接：https://leetcode.com/problems/roman-to-integer/

思路1：用trie树，没写完。

```cpp
class Solution {
public:
    struct TrieNode {
        TrieNode *next[26]; int flag;
        TrieNode() { flag = 0; fill_n(next, 26, nullptr); }
        ~TrieNode() { for (auto p : next) if (p) delete p; }
    }
    
    int romanToInt(const string &s) {
        vector<string> I = {"I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX"};
        vector<string> X = {"X", "XX", "XXX", "XL", "L", "LX", "LXX", "LXXX", "XC"};
        vector<string> C = {"C", "CC", "CCC", "CD", "D", "DC", "DCC", "DCCC", "CM"};
        vector<string> M = {"M", "MM", "MMM"};
        static TrieNode *root = nullptr;
        if (root == nullptr) {
            root = new TrieNode();
            for (auto &ss : {I, X, C, M}) for (int i = 0; i < ss.length(); ++i) {
                TrieNode *p = root;
                for (char c : s) {
                    if (p->next(s-'A') == nullptr) p->next(s-'A') = new TrieNode();
                    p = p->next(s-'A');
                }
                p->flag = (i-0+1) * 10;
            }
        }
    }
};
```

思路2：从后往前

```cpp
class Solution {
public:
    int romanToInt(const string &s) {
        static unordered_map<char, int> f;
        if (f.size() == 0) 
            f = {{'I',1}, {'V',5}, {'X',10}, {'L',50}, {'C',100}, {'D',500}, {'M',1000}};

        int result = f[s.back()];
        for (int i = s.length()-2; i >= 0; --i) 
            if (f[s[i]] < f[s[i+1]]) result -= f[s[i]];
            else                     result += f[s[i]];
        return result;
    }
};

```
## 14. Longest Common Prefix

链接：https://leetcode.com/problems/longest-common-prefix/

```cpp
class Solution {
public:
    string longestCommonPrefix(vector<string>& strs) {
        if (strs.empty()) return "";
        if (strs.size() == 1) return strs.front();
        sort(strs.begin(), strs.end());
        string result, &a = strs.front(), &b = strs.back();
        int len = min(a.length(), b.length());
        for (int i = 0; i < len; i++) 
            if (a[i] == b[i]) result.push_back(a[i]); else break;
        return move(result);
    }
};
```
## 15. 3Sum  

链接：https://leetcode.com/problems/3sum/

思路：

1. 先排序.
2. 固定l，调整i和h,加和，大于0则减小h，小于0则增大i，等于0则减小h增大i，注意去重.
3. 迭代第2步，l从0到len-3.

```cpp
class Solution {
public:
    vector<vector<int>> threeSum(vector<int>& nums) {
        sort(nums.begin(), nums.end());
        vector<vector<int>> result; int l=0, len=nums.size();
        while (l <= len-3) {
            for (int i = l+1, h=len-1; i <= h-1;) {
                int sum = nums[l] + nums[i] + nums[h];
                if      (sum > 0) for (h--; i<=h-1&&nums[h]==nums[h+1]; h--);
                else if (sum < 0) for (i++; i<=h-1&&nums[i]==nums[i-1]; i++);
                else if (sum == 0) {
                    result.push_back({nums[l], nums[i], nums[h]});
                    for (i++, h--; i<=h-1&&nums[i]==nums[i-1]; i++);
                } 
            }
            for (l++; l<=len-3&&nums[l-1]==nums[l]; l++);
        }
        return result;
    }
};
```
## 16. 3Sum Closest

链接：https://leetcode.com/problems/3sum-closest/

13ms

```cpp
class Solution {
public:
    int threeSumClosest(vector<int>& nums, int target) {
        sort(nums.begin(), nums.end());
        int l = 0, len = nums.size(), result = nums[0]+nums[1]+nums[2];
        for (l = 0; l <= len-3; ++l) for (int i=l+1, h=len-1; i <= h-1;) {
            int sum = nums[l] + nums[i] + nums[h];
            result = abs(sum-target) < abs(result-target) ? sum : result;
            if      (sum < target) { for (i++; i<=h-1 && nums[i]==nums[i-1]; i++); }
            else if (sum > target) { for (h--; i<=h-1 && nums[h]==nums[h+1]; h--); }
            else                   { return sum; }
        }
        return result;
    }
};
```

和上面的差不多时间，12ms

```cpp
class Solution {
public:
    int threeSumClosest(vector<int>& nums, int target) {
        sort(nums.begin(), nums.end());
        int l = 0, len = nums.size(), result = nums[0]+nums[1]+nums[2];
        while (l <= len-3) { 
            for (int i=l+1, h=len-1; i <= h-1;) {
                int sum = nums[l] + nums[i] + nums[h];
                result = abs(sum-target) < abs(result-target) ? sum : result;
                if      (sum < target) { for (i++; i<=h-1 && nums[i]==nums[i-1]; i++); }
                else if (sum > target) { for (h--; i<=h-1 && nums[h]==nums[h+1]; h--); }
                else                   { return sum; }
            }
            for (l++; l<=len-3 && nums[l]==nums[l-1]; l++);
        }
        return result;
    }
};
```

## 17. Letter Combinations of a Phone Number

思路：对于每个数字，将当前的result和空tmp交换，然后对tmp中的每个word，追加c后，添加到result.

```cpp
class Solution {
public:
    vector<string> letterCombinations(const string& digits) {
        static vector<string> ss = {"abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz"};
        vector<string> result;
        for (auto d : digits) {
            int idx = d - '0' - 2;
            if (idx < 0 || idx >= 8) { continue; }
            string s = ss[idx];
            if (result.empty()) { result.push_back(""); }
            vector<string> tmp; result.swap(tmp);
            for (auto c : s) for (auto it : tmp) {
                it.push_back(c); result.push_back(it);
            }
        }
        return move(result);
    }
};
```
## 18. 4Sum 

链接：https://leetcode.com/problems/4sum/

思路：在3Sum的基础上，多做一层循环

```cpp
class Solution {
public:
    vector<vector<int>> fourSum(vector<int>& nums, int target) {
        sort(nums.begin(), nums.end());
        vector<vector<int>> result; int len=nums.size();
        for (int i=0; i<=len-4;) {
            for (int j=i+1; j<=len-3;) {
                for (int k=j+1, l=len-1; k<l;) {
                    int sum = nums[i] + nums[j] + nums[k] + nums[l];
                    if      (sum > target) for (l--; k<l&&nums[l]==nums[l+1]; l--);
                    else if (sum < target) for (k++; j<k&&nums[k]==nums[k-1]; k++);
                    else if (sum == target) {
                        result.push_back({nums[i], nums[j], nums[k], nums[l]});
                        for (k++, l--; k<l&&nums[k]==nums[k-1]; k++);
                    } 
                }
                for (j++; j<=len-3&&nums[j-1]==nums[j]; j++);
            }
            for (i++; i<=len-4&&nums[i-1]==nums[i]; i++);
        }
        return move(result);
    }
};
```

整理下，可以改改就能过threeSum和twoSum，还有点优化，在nSum中通过最小值和最大值乘以n和target做比较来截枝，这边就没做了。

```cpp
class Solution {
public:
    vector<vector<int>> fourSum(vector<int>& nums, int target) {
        vector<vector<int>> result; vector<int> path;
        if (nums.size() < 4) return result;
        sort(nums.begin(), nums.end());
        fourSum(nums, target, 0, path, result);
        return move(result);
    }
    
    void fourSum(vector<int> &nums, int target, int low, vector<int> &path, vector<vector<int>> &result) {
        for (int i=low; i<=nums.size()-4; i++) {
            if (i>low && nums[i-1]==nums[i]) continue;
            path.push_back(nums[i]);
            threeSum(nums, target-nums[i], i+1, path, result);
            path.pop_back();
        }
    }
    
    void threeSum(vector<int> &nums, int target, int low, vector<int> &path, vector<vector<int>> &result) {
        for (int i=low; i<=nums.size()-3; i++) {
            if (i>low && nums[i-1] == nums[i]) continue;
            path.push_back(nums[i]);
            twoSum(nums, target-nums[i], i+1, path, result);
            path.pop_back();
        }
    }
    
    void twoSum(vector<int> &nums, int target, int low, vector<int> &path, vector<vector<int>> &result) {
        int i=low, j=nums.size()-1;
        while (i < j) {
            int sum = nums[i] + nums[j];
            if (sum == target) {
                vector<int> tmp = path; tmp.push_back(nums[i]); tmp.push_back(nums[j]);
                result.push_back(tmp);
                for (i++, j--; i<j && nums[i-1]==nums[i]; ++i);
            } 
            else if (sum > target) for (j--; i<j&&nums[j]==nums[j+1]; j--);
            else if (sum < target) for (i++; i<j&&nums[i-1]==nums[i]; i++);
        }
    }
};
```

## 19. Remove Nth Node From End of List 

链接：https://leetcode.com/problems/remove-nth-node-from-end-of-list/#/description

两趟遍历

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
    ListNode* removeNthFromEnd(ListNode* head, int n) {
        ListNode *p;
        int len = 0; for (p = head; p != nullptr; p = p->next) len++;
        if (n >  len) return head;
        if (n == len) return head->next;
        p = head; for (int i = 0; i < len - n - 1; i++) p = p->next;
        ListNode *tmp = p->next;
        p->next = p->next->next;
        delete tmp;
        return head;
    }
};
```

单趟遍历

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
    ListNode* removeNthFromEnd(ListNode* head, int n) {
        ListNode *p = head;
        for (int i = 0; i < n; i++) { if (!p) return head; p = p->next; }
        ListNode dummy(0); ListNode *pre = &dummy; pre->next = head;
        for (; p != nullptr; p = p->next) { pre = pre->next; }
        ListNode *tmp = pre->next;
        pre->next = pre->next->next;
        delete tmp;
        return dummy.next;
    }
};
```

注意，如果把delete tmp注释掉，速度就能快上很多。不注释，就只能打败6%。深深怀疑，大部分同志忘记了delete tmp了。

## 20. Valid Parentheses 

链接：https://leetcode.com/problems/valid-parentheses/#/description

使用单纯的比较 beat 15.3%

```cpp 
class Solution {
public:
    bool isValid(string s) {
        stack<char> stk;
        for (auto c : s) {
            if (!stk.empty() && (stk.top()=='(' && c==')' || stk.top()=='[' && c==']' || stk.top()=='{' && c=='}')) { 
                stk.pop(); 
            } else {
                stk.push(c);
            }
        }
        return stk.empty();
    }
};
```

使用hashmap，beat 1.73%

```cpp
class Solution {
public:
    bool isValid(string s) {
        stack<char> stk;
        for (auto c : s) {
            unordered_map<char, char> tbl = {{'(',')'},{'{','}'},{'[',']'}};
            if (!stk.empty() && tbl[stk.top()] == c) { 
                stk.pop(); 
            } else {
                stk.push(c);
            }
        }
        return stk.empty();
    }
};
```

打表法 beat 15%

```cpp
class Solution {
public:
    bool isValid(string s) {
        stack<char> stk; char tbl[256] = {0};
        tbl['('] = ')'; tbl['['] = ']'; tbl['{'] = '}';
        for (auto c : s) {
            if (!stk.empty() && tbl[stk.top()] == c) { 
                stk.pop(); 
            } else {
                stk.push(c);
            }
        }
        return stk.empty();
    }
};
```

