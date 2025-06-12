# Dynamic Programming

## 📚 Theory - Dynamic Programming

**Dynamic Programming (DP)** is an algorithmic paradigm that solves complex problems by breaking them down into simpler subproblems and storing the results to avoid redundant calculations.

### Core Principles:
1. **Optimal Substructure**: Solution can be constructed from optimal solutions of subproblems
2. **Overlapping Subproblems**: Same subproblems are solved multiple times
3. **Memoization**: Store results of subproblems to avoid recomputation
4. **Bottom-up approach**: Build solution from smaller subproblems

### Key Characteristics:
- **State**: Represents a subproblem
- **Transition**: How to move from one state to another
- **Base Case**: Simplest subproblem with known solution
- **Recurrence Relation**: Mathematical formula connecting states

### DP vs Divide & Conquer:
- **Divide & Conquer**: Subproblems are independent (e.g., Merge Sort)
- **Dynamic Programming**: Subproblems overlap and are reused (e.g., Fibonacci)

### Real-World Applications:
- **Economics**: Resource allocation, portfolio optimization
- **Bioinformatics**: DNA sequence alignment, protein folding
- **Computer Graphics**: Image processing, pathfinding
- **Operations Research**: Inventory management, scheduling
- **Game Theory**: Minimax algorithms, decision trees
- **Engineering**: Network routing, signal processing

### Types of DP:
1. **Linear DP**: 1D problems (Fibonacci, Climbing Stairs)
2. **Grid DP**: 2D problems (Unique Paths, Edit Distance)
3. **Interval DP**: Range problems (Matrix Chain Multiplication)
4. **Tree DP**: Tree-based problems (Diameter, Path Sums)
5. **Bitmask DP**: Using bitmasks as states
6. **Digit DP**: Number-based constraints

## 💻 1. Memoization vs Tabulation

### Memoization (Top-down)
**Theory**: Start with original problem, recursively break into subproblems, store results in cache.

```cpp
// Fibonacci with Memoization
class FibonacciMemo {
private:
    unordered_map<int, long long> memo;
    
public:
    long long fib(int n) {
        if (n <= 1) return n;
        
        if (memo.find(n) != memo.end()) {
            return memo[n];
        }
        
        memo[n] = fib(n - 1) + fib(n - 2);
        return memo[n];
    }
};

// Generic Memoization Template
template<typename T>
class Memoization {
private:
    unordered_map<string, T> cache;
    
    string createKey(const vector<int>& params) {
        string key;
        for (int param : params) {
            key += to_string(param) + ",";
        }
        return key;
    }
    
public:
    T solve(const vector<int>& params, function<T(const vector<int>&)> compute) {
        string key = createKey(params);
        
        if (cache.find(key) != cache.end()) {
            return cache[key];
        }
        
        cache[key] = compute(params);
        return cache[key];
    }
};
```

### Tabulation (Bottom-up)
**Theory**: Start with base cases, iteratively build up to final solution.

```cpp
// Fibonacci with Tabulation
long long fibTabulation(int n) {
    if (n <= 1) return n;
    
    vector<long long> dp(n + 1);
    dp[0] = 0;
    dp[1] = 1;
    
    for (int i = 2; i <= n; i++) {
        dp[i] = dp[i - 1] + dp[i - 2];
    }
    
    return dp[n];
}

// Space Optimized Fibonacci
long long fibOptimized(int n) {
    if (n <= 1) return n;
    
    long long prev2 = 0, prev1 = 1;
    
    for (int i = 2; i <= n; i++) {
        long long curr = prev1 + prev2;
        prev2 = prev1;
        prev1 = curr;
    }
    
    return prev1;
}
```

### Comparison: Memoization vs Tabulation

| Aspect | Memoization | Tabulation |
|--------|-------------|------------|
| Approach | Top-down | Bottom-up |
| Implementation | Recursive | Iterative |
| Space | May use less (only needed states) | Uses space for all states |
| Time | May be slower (function call overhead) | Generally faster |
| Stack Space | O(depth) recursion stack | O(1) stack space |
| Easy to Code | More intuitive | Requires careful ordering |

## 💻 2. Classic DP Problems

### 0/1 Knapsack Problem
**Theory**: Given items with weights and values, maximize value in knapsack with weight capacity W.

```cpp
// Recursive with Memoization
class Knapsack {
private:
    vector<vector<int>> memo;
    
    int knapsackMemo(vector<int>& weights, vector<int>& values, int W, int n) {
        if (n == 0 || W == 0) return 0;
        
        if (memo[n][W] != -1) return memo[n][W];
        
        // If current item weight > capacity, can't include it
        if (weights[n - 1] > W) {
            memo[n][W] = knapsackMemo(weights, values, W, n - 1);
        } else {
            // Max of including or excluding current item
            int include = values[n - 1] + knapsackMemo(weights, values, W - weights[n - 1], n - 1);
            int exclude = knapsackMemo(weights, values, W, n - 1);
            memo[n][W] = max(include, exclude);
        }
        
        return memo[n][W];
    }
    
public:
    int knapsack(vector<int>& weights, vector<int>& values, int W) {
        int n = weights.size();
        memo.assign(n + 1, vector<int>(W + 1, -1));
        return knapsackMemo(weights, values, W, n);
    }
};

// Tabulation Approach
int knapsackTabulation(vector<int>& weights, vector<int>& values, int W) {
    int n = weights.size();
    vector<vector<int>> dp(n + 1, vector<int>(W + 1, 0));
    
    for (int i = 1; i <= n; i++) {
        for (int w = 1; w <= W; w++) {
            if (weights[i - 1] <= w) {
                dp[i][w] = max(
                    values[i - 1] + dp[i - 1][w - weights[i - 1]], // Include
                    dp[i - 1][w] // Exclude
                );
            } else {
                dp[i][w] = dp[i - 1][w];
            }
        }
    }
    
    return dp[n][W];
}

// Space Optimized (1D array)
int knapsackOptimized(vector<int>& weights, vector<int>& values, int W) {
    vector<int> dp(W + 1, 0);
    
    for (int i = 0; i < weights.size(); i++) {
        // Traverse in reverse to avoid using updated values
        for (int w = W; w >= weights[i]; w--) {
            dp[w] = max(dp[w], values[i] + dp[w - weights[i]]);
        }
    }
    
    return dp[W];
}

// Print selected items
vector<int> knapsackItems(vector<int>& weights, vector<int>& values, int W) {
    int n = weights.size();
    vector<vector<int>> dp(n + 1, vector<int>(W + 1, 0));
    
    // Fill DP table
    for (int i = 1; i <= n; i++) {
        for (int w = 1; w <= W; w++) {
            if (weights[i - 1] <= w) {
                dp[i][w] = max(
                    values[i - 1] + dp[i - 1][w - weights[i - 1]],
                    dp[i - 1][w]
                );
            } else {
                dp[i][w] = dp[i - 1][w];
            }
        }
    }
    
    // Backtrack to find items
    vector<int> selectedItems;
    int w = W;
    for (int i = n; i > 0 && w > 0; i--) {
        if (dp[i][w] != dp[i - 1][w]) {
            selectedItems.push_back(i - 1);
            w -= weights[i - 1];
        }
    }
    
    reverse(selectedItems.begin(), selectedItems.end());
    return selectedItems;
}
```

### Knapsack Variants
```cpp
// Unbounded Knapsack (infinite items)
int unboundedKnapsack(vector<int>& weights, vector<int>& values, int W) {
    vector<int> dp(W + 1, 0);
    
    for (int w = 1; w <= W; w++) {
        for (int i = 0; i < weights.size(); i++) {
            if (weights[i] <= w) {
                dp[w] = max(dp[w], values[i] + dp[w - weights[i]]);
            }
        }
    }
    
    return dp[W];
}

// Bounded Knapsack (limited quantity of each item)
int boundedKnapsack(vector<int>& weights, vector<int>& values, vector<int>& quantities, int W) {
    vector<int> dp(W + 1, 0);
    
    for (int i = 0; i < weights.size(); i++) {
        for (int w = W; w >= weights[i]; w--) {
            for (int k = 1; k <= quantities[i] && k * weights[i] <= w; k++) {
                dp[w] = max(dp[w], k * values[i] + dp[w - k * weights[i]]);
            }
        }
    }
    
    return dp[W];
}
```

## 💻 3. Subset Sum and Partition Problems

### Subset Sum Problem
```cpp
// Check if subset with given sum exists
bool subsetSum(vector<int>& nums, int target) {
    vector<bool> dp(target + 1, false);
    dp[0] = true; // Empty subset has sum 0
    
    for (int num : nums) {
        for (int sum = target; sum >= num; sum--) {
            dp[sum] = dp[sum] || dp[sum - num];
        }
    }
    
    return dp[target];
}

// Count number of subsets with given sum
int countSubsetSum(vector<int>& nums, int target) {
    vector<int> dp(target + 1, 0);
    dp[0] = 1; // One way to make sum 0
    
    for (int num : nums) {
        for (int sum = target; sum >= num; sum--) {
            dp[sum] += dp[sum - num];
        }
    }
    
    return dp[target];
}

// Print all subsets with given sum
void printAllSubsets(vector<int>& nums, int target) {
    vector<vector<vector<int>>> dp(target + 1);
    dp[0].push_back({}); // Empty subset for sum 0
    
    for (int num : nums) {
        for (int sum = target; sum >= num; sum--) {
            for (auto& subset : dp[sum - num]) {
                vector<int> newSubset = subset;
                newSubset.push_back(num);
                dp[sum].push_back(newSubset);
            }
        }
    }
    
    for (auto& subset : dp[target]) {
        for (int x : subset) cout << x << " ";
        cout << endl;
    }
}
```

### Partition Equal Subset Sum
```cpp
bool canPartition(vector<int>& nums) {
    int totalSum = accumulate(nums.begin(), nums.end(), 0);
    
    // If total sum is odd, can't partition equally
    if (totalSum % 2 != 0) return false;
    
    int target = totalSum / 2;
    return subsetSum(nums, target);
}

// Minimum Subset Sum Difference
int minSubsetSumDiff(vector<int>& nums) {
    int totalSum = accumulate(nums.begin(), nums.end(), 0);
    vector<bool> dp(totalSum / 2 + 1, false);
    dp[0] = true;
    
    for (int num : nums) {
        for (int sum = totalSum / 2; sum >= num; sum--) {
            dp[sum] = dp[sum] || dp[sum - num];
        }
    }
    
    // Find the largest sum <= totalSum/2 that's possible
    int maxSum = 0;
    for (int i = totalSum / 2; i >= 0; i--) {
        if (dp[i]) {
            maxSum = i;
            break;
        }
    }
    
    return totalSum - 2 * maxSum;
}
```

## 💻 4. String DP Problems

### Longest Common Subsequence (LCS)
```cpp
int longestCommonSubsequence(string text1, string text2) {
    int m = text1.length(), n = text2.length();
    vector<vector<int>> dp(m + 1, vector<int>(n + 1, 0));
    
    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= n; j++) {
            if (text1[i - 1] == text2[j - 1]) {
                dp[i][j] = 1 + dp[i - 1][j - 1];
            } else {
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1]);
            }
        }
    }
    
    return dp[m][n];
}

// Print LCS
string printLCS(string text1, string text2) {
    int m = text1.length(), n = text2.length();
    vector<vector<int>> dp(m + 1, vector<int>(n + 1, 0));
    
    // Fill DP table
    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= n; j++) {
            if (text1[i - 1] == text2[j - 1]) {
                dp[i][j] = 1 + dp[i - 1][j - 1];
            } else {
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1]);
            }
        }
    }
    
    // Backtrack to construct LCS
    string lcs;
    int i = m, j = n;
    while (i > 0 && j > 0) {
        if (text1[i - 1] == text2[j - 1]) {
            lcs = text1[i - 1] + lcs;
            i--;
            j--;
        } else if (dp[i - 1][j] > dp[i][j - 1]) {
            i--;
        } else {
            j--;
        }
    }
    
    return lcs;
}

// Space Optimized LCS
int lcsOptimized(string text1, string text2) {
    int m = text1.length(), n = text2.length();
    vector<int> prev(n + 1, 0), curr(n + 1, 0);
    
    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= n; j++) {
            if (text1[i - 1] == text2[j - 1]) {
                curr[j] = 1 + prev[j - 1];
            } else {
                curr[j] = max(prev[j], curr[j - 1]);
            }
        }
        prev = curr;
    }
    
    return curr[n];
}
```

### Longest Increasing Subsequence (LIS)
```cpp
// O(n²) DP Solution
int longestIncreasingSubsequence(vector<int>& nums) {
    int n = nums.size();
    vector<int> dp(n, 1);
    
    for (int i = 1; i < n; i++) {
        for (int j = 0; j < i; j++) {
            if (nums[j] < nums[i]) {
                dp[i] = max(dp[i], dp[j] + 1);
            }
        }
    }
    
    return *max_element(dp.begin(), dp.end());
}

// O(n log n) Binary Search Solution
int lisOptimal(vector<int>& nums) {
    vector<int> tails;
    
    for (int num : nums) {
        auto it = lower_bound(tails.begin(), tails.end(), num);
        if (it == tails.end()) {
            tails.push_back(num);
        } else {
            *it = num;
        }
    }
    
    return tails.size();
}

// Print LIS
vector<int> printLIS(vector<int>& nums) {
    int n = nums.size();
    vector<int> dp(n, 1);
    vector<int> parent(n, -1);
    
    for (int i = 1; i < n; i++) {
        for (int j = 0; j < i; j++) {
            if (nums[j] < nums[i] && dp[j] + 1 > dp[i]) {
                dp[i] = dp[j] + 1;
                parent[i] = j;
            }
        }
    }
    
    // Find the index with maximum LIS length
    int maxIndex = 0;
    for (int i = 1; i < n; i++) {
        if (dp[i] > dp[maxIndex]) {
            maxIndex = i;
        }
    }
    
    // Reconstruct LIS
    vector<int> lis;
    int curr = maxIndex;
    while (curr != -1) {
        lis.push_back(nums[curr]);
        curr = parent[curr];
    }
    
    reverse(lis.begin(), lis.end());
    return lis;
}
```

### Edit Distance (Levenshtein Distance)
```cpp
int editDistance(string word1, string word2) {
    int m = word1.length(), n = word2.length();
    vector<vector<int>> dp(m + 1, vector<int>(n + 1));
    
    // Base cases
    for (int i = 0; i <= m; i++) dp[i][0] = i;
    for (int j = 0; j <= n; j++) dp[0][j] = j;
    
    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= n; j++) {
            if (word1[i - 1] == word2[j - 1]) {
                dp[i][j] = dp[i - 1][j - 1]; // No operation needed
            } else {
                dp[i][j] = 1 + min({
                    dp[i - 1][j],     // Delete
                    dp[i][j - 1],     // Insert
                    dp[i - 1][j - 1]  // Replace
                });
            }
        }
    }
    
    return dp[m][n];
}

// Space Optimized Edit Distance
int editDistanceOptimized(string word1, string word2) {
    int m = word1.length(), n = word2.length();
    vector<int> prev(n + 1), curr(n + 1);
    
    for (int j = 0; j <= n; j++) prev[j] = j;
    
    for (int i = 1; i <= m; i++) {
        curr[0] = i;
        for (int j = 1; j <= n; j++) {
            if (word1[i - 1] == word2[j - 1]) {
                curr[j] = prev[j - 1];
            } else {
                curr[j] = 1 + min({prev[j], curr[j - 1], prev[j - 1]});
            }
        }
        prev = curr;
    }
    
    return curr[n];
}
```

## 💻 5. Matrix Chain Multiplication

### Theory
Find the optimal way to parenthesize matrix multiplications to minimize scalar multiplications.

```cpp
// Matrix Chain Multiplication
int matrixChainMultiplication(vector<int>& dimensions) {
    int n = dimensions.size() - 1; // Number of matrices
    vector<vector<int>> dp(n, vector<int>(n, 0));
    
    // l is chain length
    for (int l = 2; l <= n; l++) {
        for (int i = 0; i < n - l + 1; i++) {
            int j = i + l - 1;
            dp[i][j] = INT_MAX;
            
            for (int k = i; k < j; k++) {
                int cost = dp[i][k] + dp[k + 1][j] + 
                          dimensions[i] * dimensions[k + 1] * dimensions[j + 1];
                dp[i][j] = min(dp[i][j], cost);
            }
        }
    }
    
    return dp[0][n - 1];
}

// Print optimal parenthesization
void printOptimalParentheses(vector<vector<int>>& split, int i, int j) {
    if (i == j) {
        cout << "M" << i;
    } else {
        cout << "(";
        printOptimalParentheses(split, i, split[i][j]);
        printOptimalParentheses(split, split[i][j] + 1, j);
        cout << ")";
    }
}

pair<int, string> matrixChainWithParentheses(vector<int>& dimensions) {
    int n = dimensions.size() - 1;
    vector<vector<int>> dp(n, vector<int>(n, 0));
    vector<vector<int>> split(n, vector<int>(n, 0));
    
    for (int l = 2; l <= n; l++) {
        for (int i = 0; i < n - l + 1; i++) {
            int j = i + l - 1;
            dp[i][j] = INT_MAX;
            
            for (int k = i; k < j; k++) {
                int cost = dp[i][k] + dp[k + 1][j] + 
                          dimensions[i] * dimensions[k + 1] * dimensions[j + 1];
                
                if (cost < dp[i][j]) {
                    dp[i][j] = cost;
                    split[i][j] = k;
                }
            }
        }
    }
    
    // Build parentheses string
    function<string(int, int)> buildString = [&](int i, int j) -> string {
        if (i == j) {
            return "M" + to_string(i);
        }
        return "(" + buildString(i, split[i][j]) + 
               buildString(split[i][j] + 1, j) + ")";
    };
    
    return {dp[0][n - 1], buildString(0, n - 1)};
}
```

## 💻 6. Advanced DP Techniques

### Bitmask DP
```cpp
// Traveling Salesman Problem using Bitmask DP
int tsp(vector<vector<int>>& dist) {
    int n = dist.size();
    vector<vector<int>> dp(1 << n, vector<int>(n, INT_MAX));
    
    // Start from city 0
    dp[1][0] = 0;
    
    for (int mask = 0; mask < (1 << n); mask++) {
        for (int u = 0; u < n; u++) {
            if (!(mask & (1 << u))) continue;
            if (dp[mask][u] == INT_MAX) continue;
            
            for (int v = 0; v < n; v++) {
                if (mask & (1 << v)) continue;
                
                int newMask = mask | (1 << v);
                dp[newMask][v] = min(dp[newMask][v], dp[mask][u] + dist[u][v]);
            }
        }
    }
    
    int result = INT_MAX;
    for (int i = 1; i < n; i++) {
        result = min(result, dp[(1 << n) - 1][i] + dist[i][0]);
    }
    
    return result;
}

// Assignment Problem using Bitmask DP
int assignmentProblem(vector<vector<int>>& cost) {
    int n = cost.size();
    vector<int> dp(1 << n, INT_MAX);
    dp[0] = 0;
    
    for (int mask = 0; mask < (1 << n); mask++) {
        if (dp[mask] == INT_MAX) continue;
        
        int person = __builtin_popcount(mask);
        if (person == n) continue;
        
        for (int task = 0; task < n; task++) {
            if (mask & (1 << task)) continue;
            
            int newMask = mask | (1 << task);
            dp[newMask] = min(dp[newMask], dp[mask] + cost[person][task]);
        }
    }
    
    return dp[(1 << n) - 1];
}
```

### Digit DP
```cpp
// Count numbers <= N with digit sum = K
class DigitDP {
private:
    string num;
    int K;
    vector<vector<vector<int>>> dp;
    
    int solve(int pos, int sum, int tight) {
        if (pos == num.length()) {
            return (sum == K) ? 1 : 0;
        }
        
        if (dp[pos][sum][tight] != -1) {
            return dp[pos][sum][tight];
        }
        
        int limit = tight ? (num[pos] - '0') : 9;
        int result = 0;
        
        for (int digit = 0; digit <= limit; digit++) {
            int newTight = tight && (digit == limit);
            int newSum = sum + digit;
            
            if (newSum <= K) {
                result += solve(pos + 1, newSum, newTight);
            }
        }
        
        return dp[pos][sum][tight] = result;
    }
    
public:
    int countNumbers(int N, int targetSum) {
        num = to_string(N);
        K = targetSum;
        dp.assign(num.length(), vector<vector<int>>(K + 1, vector<int>(2, -1)));
        
        return solve(0, 0, 1);
    }
};
```

### DP on Trees
```cpp
// Maximum path sum in binary tree
struct TreeNode {
    int val;
    TreeNode* left;
    TreeNode* right;
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
};

class TreeDP {
private:
    int maxPathSum;
    
    int maxPathHelper(TreeNode* node) {
        if (!node) return 0;
        
        int leftGain = max(maxPathHelper(node->left), 0);
        int rightGain = max(maxPathHelper(node->right), 0);
        
        // Path passing through current node
        int currentMax = node->val + leftGain + rightGain;
        maxPathSum = max(maxPathSum, currentMax);
        
        // Return max gain if we continue path through this node
        return node->val + max(leftGain, rightGain);
    }
    
public:
    int maxPathSum(TreeNode* root) {
        maxPathSum = INT_MIN;
        maxPathHelper(root);
        return maxPathSum;
    }
};

// House Robber in Binary Tree
int robTree(TreeNode* root) {
    auto result = robHelper(root);
    return max(result.first, result.second);
}

pair<int, int> robHelper(TreeNode* node) {
    if (!node) return {0, 0};
    
    auto left = robHelper(node->left);
    auto right = robHelper(node->right);
    
    // {rob this node, don't rob this node}
    int robThis = node->val + left.second + right.second;
    int notRobThis = max(left.first, left.second) + max(right.first, right.second);
    
    return {robThis, notRobThis};
}
```

## 💻 7. Space Optimization Techniques

### Rolling Array Technique
```cpp
// 2D DP to 1D optimization
int uniquePaths(int m, int n) {
    vector<int> dp(n, 1);
    
    for (int i = 1; i < m; i++) {
        for (int j = 1; j < n; j++) {
            dp[j] += dp[j - 1];
        }
    }
    
    return dp[n - 1];
}

// Alternating arrays for 2D DP
int minPathSum(vector<vector<int>>& grid) {
    int m = grid.size(), n = grid[0].size();
    vector<int> prev(n), curr(n);
    
    // Initialize first row
    prev[0] = grid[0][0];
    for (int j = 1; j < n; j++) {
        prev[j] = prev[j - 1] + grid[0][j];
    }
    
    for (int i = 1; i < m; i++) {
        curr[0] = prev[0] + grid[i][0];
        for (int j = 1; j < n; j++) {
            curr[j] = min(prev[j], curr[j - 1]) + grid[i][j];
        }
        prev = curr;
    }
    
    return prev[n - 1];
}
```

### State Compression
```cpp
// Compress state when only few previous states matter
int fibSpaceOptimal(int n) {
    if (n <= 1) return n;
    
    int a = 0, b = 1;
    for (int i = 2; i <= n; i++) {
        int c = a + b;
        a = b;
        b = c;
    }
    return b;
}

// Bitmask for subset state compression
int maximalSquare(vector<vector<char>>& matrix) {
    int m = matrix.size(), n = matrix[0].size();
    vector<int> dp(n, 0);
    int maxSide = 0, prev = 0;
    
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            int temp = dp[j];
            if (matrix[i][j] == '1') {
                if (i == 0 || j == 0) {
                    dp[j] = 1;
                } else {
                    dp[j] = min({dp[j], dp[j - 1], prev}) + 1;
                }
                maxSide = max(maxSide, dp[j]);
            } else {
                dp[j] = 0;
            }
            prev = temp;
        }
    }
    
    return maxSide * maxSide;
}
```

## 🎯 DP Problem Patterns

### Pattern Recognition Guide

| Pattern | Key Characteristics | Example Problems |
|---------|-------------------|------------------|
| **Linear DP** | 1D array, depends on previous elements | Fibonacci, Climbing Stairs, House Robber |
| **Grid DP** | 2D grid, path/optimization problems | Unique Paths, Min Path Sum, Dungeon Game |
| **Interval DP** | Range [i,j], optimal way to process interval | Matrix Chain, Burst Balloons, Palindrome Partitioning |
| **Subsequence DP** | LIS, LCS type problems | LIS, LCS, Edit Distance |
| **Partition DP** | Divide into parts optimally | Subset Sum, Palindrome Partitioning |
| **Game DP** | Two players, optimal strategy | Stone Game, Minimax |
| **Tree DP** | Tree-based optimization | Max Path Sum, Tree Diameter |
| **Bitmask DP** | Use bitmask as state | TSP, Assignment Problem |

### Problem-Solving Framework

1. **Identify the problem type**: Is it optimization, counting, or decision?
2. **Define the state**: What parameters uniquely identify a subproblem?
3. **Find the recurrence**: How does current state relate to previous states?
4. **Determine base cases**: What are the simplest subproblems?
5. **Choose approach**: Memoization (top-down) or tabulation (bottom-up)?
6. **Optimize space**: Can we reduce space complexity?

## 📝 Interview Tips

1. **Start with brute force**: Identify overlapping subproblems
2. **Define clear states**: Be precise about what each DP state represents
3. **Write recurrence relation**: Mathematical formula is crucial
4. **Handle base cases carefully**: Edge cases often trip up candidates
5. **Consider space optimization**: Show you understand memory trade-offs
6. **Practice pattern recognition**: Learn to identify DP problem types quickly

## 🎪 Common Interview Questions

**Q**: How do you know when to use DP?
**A**: Look for optimal substructure and overlapping subproblems. If you can break problem into smaller similar problems and same subproblems are solved repeatedly.

**Q**: What's the difference between memoization and tabulation?
**A**: Memoization is top-down recursive with caching. Tabulation is bottom-up iterative. Tabulation is generally more space and time efficient.

**Q**: How do you optimize space in DP?
**A**: Use rolling arrays when you only need previous few states, compress states using bitmasks, or use variables instead of arrays when possible.

## 🚀 Advanced Problem Examples

### Egg Dropping Problem
```cpp
int eggDrop(int eggs, int floors) {
    vector<vector<int>> dp(eggs + 1, vector<int>(floors + 1, 0));
    
    // Base cases
    for (int i = 1; i <= eggs; i++) {
        dp[i][0] = 0; // 0 floors
        dp[i][1] = 1; // 1 floor
    }
    
    for (int j = 1; j <= floors; j++) {
        dp[1][j] = j; // 1 egg
    }
    
    for (int i = 2; i <= eggs; i++) {
        for (int j = 2; j <= floors; j++) {
            dp[i][j] = INT_MAX;
            
            for (int x = 1; x <= j; x++) {
                int worst = 1 + max(dp[i - 1][x - 1], dp[i][j - x]);
                dp[i][j] = min(dp[i][j], worst);
            }
        }
    }
    
    return dp[eggs][floors];
}
```

This comprehensive guide covers all major DP concepts and techniques needed for placement interviews, with detailed theory, implementations, and optimization strategies. 