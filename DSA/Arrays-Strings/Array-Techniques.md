# Array & String Techniques

## 📚 Theory

Arrays and strings are fundamental data structures that form the basis of many algorithmic problems. Mastering key techniques like Kadane's algorithm, two pointers, sliding window, and prefix sums is crucial for solving complex problems efficiently.

## 🔍 Real-Life Examples

**Kadane's Algorithm**: Finding the best consecutive days for maximum profit
**Two Pointer**: Meeting in the middle of a bridge from both ends
**Sliding Window**: Looking through a window that moves along a train
**Prefix Sums**: Keeping running totals in a bank account

## 💻 1. Kadane's Algorithm (Maximum Subarray)

### Theory
Finds the contiguous subarray with the largest sum in O(n) time. Key insight: if current sum becomes negative, start fresh from next element.

### Code Implementation
```cpp
// Basic Kadane's Algorithm
int maxSubarraySum(vector<int>& arr) {
    int maxSoFar = arr[0];
    int maxEndingHere = arr[0];
    
    for (int i = 1; i < arr.size(); i++) {
        maxEndingHere = max(arr[i], maxEndingHere + arr[i]);
        maxSoFar = max(maxSoFar, maxEndingHere);
    }
    
    return maxSoFar;
}

// Return the actual subarray indices
pair<int, int> maxSubarrayIndices(vector<int>& arr) {
    int maxSoFar = arr[0];
    int maxEndingHere = arr[0];
    int start = 0, end = 0, tempStart = 0;
    
    for (int i = 1; i < arr.size(); i++) {
        if (maxEndingHere < 0) {
            maxEndingHere = arr[i];
            tempStart = i;
        } else {
            maxEndingHere += arr[i];
        }
        
        if (maxEndingHere > maxSoFar) {
            maxSoFar = maxEndingHere;
            start = tempStart;
            end = i;
        }
    }
    
    return {start, end};
}

// Handle all negative numbers
int maxSubarraySumAllNegative(vector<int>& arr) {
    int maxSoFar = INT_MIN;
    int maxEndingHere = 0;
    
    for (int num : arr) {
        maxEndingHere += num;
        maxSoFar = max(maxSoFar, maxEndingHere);
        
        if (maxEndingHere < 0) {
            maxEndingHere = 0;
        }
    }
    
    // If all elements are negative, return the maximum element
    if (maxSoFar < 0) {
        return *max_element(arr.begin(), arr.end());
    }
    
    return maxSoFar;
}

// Maximum circular subarray sum
int maxCircularSum(vector<int>& arr) {
    int n = arr.size();
    
    // Case 1: Maximum subarray is non-circular
    int maxKadane = maxSubarraySum(arr);
    
    // Case 2: Maximum subarray is circular
    int totalSum = 0;
    for (int i = 0; i < n; i++) {
        totalSum += arr[i];
        arr[i] = -arr[i];  // Invert array
    }
    
    int maxCircular = totalSum + maxSubarraySum(arr);  // Add because array is inverted
    
    return max(maxKadane, maxCircular);
}
```

### Applications
- Stock trading problems
- Maximum sum rectangle in 2D array
- Maximum product subarray

## 💻 2. Two Pointer Technique

### Theory
Uses two pointers moving towards each other or in the same direction to solve problems in O(n) time instead of O(n²).

### Code Implementation
```cpp
// Two Sum in sorted array
vector<int> twoSumSorted(vector<int>& arr, int target) {
    int left = 0, right = arr.size() - 1;
    
    while (left < right) {
        int sum = arr[left] + arr[right];
        
        if (sum == target) {
            return {left, right};
        } else if (sum < target) {
            left++;
        } else {
            right--;
        }
    }
    
    return {-1, -1};  // Not found
}

// Remove duplicates from sorted array
int removeDuplicates(vector<int>& arr) {
    if (arr.empty()) return 0;
    
    int writeIndex = 1;
    
    for (int readIndex = 1; readIndex < arr.size(); readIndex++) {
        if (arr[readIndex] != arr[readIndex - 1]) {
            arr[writeIndex] = arr[readIndex];
            writeIndex++;
        }
    }
    
    return writeIndex;
}

// Three Sum (find triplets that sum to zero)
vector<vector<int>> threeSum(vector<int>& arr) {
    vector<vector<int>> result;
    sort(arr.begin(), arr.end());
    
    for (int i = 0; i < arr.size() - 2; i++) {
        if (i > 0 && arr[i] == arr[i - 1]) continue;  // Skip duplicates
        
        int left = i + 1, right = arr.size() - 1;
        
        while (left < right) {
            int sum = arr[i] + arr[left] + arr[right];
            
            if (sum == 0) {
                result.push_back({arr[i], arr[left], arr[right]});
                
                // Skip duplicates
                while (left < right && arr[left] == arr[left + 1]) left++;
                while (left < right && arr[right] == arr[right - 1]) right--;
                
                left++;
                right--;
            } else if (sum < 0) {
                left++;
            } else {
                right--;
            }
        }
    }
    
    return result;
}

// Container with most water
int maxArea(vector<int>& height) {
    int left = 0, right = height.size() - 1;
    int maxWater = 0;
    
    while (left < right) {
        int width = right - left;
        int currentWater = width * min(height[left], height[right]);
        maxWater = max(maxWater, currentWater);
        
        // Move pointer with smaller height
        if (height[left] < height[right]) {
            left++;
        } else {
            right--;
        }
    }
    
    return maxWater;
}
```

## 💻 3. Sliding Window Technique

### Theory
Maintains a window of elements and slides it across the array. Useful for problems involving subarrays with specific properties.

### Code Implementation
```cpp
// Maximum sum of k consecutive elements
int maxSumKElements(vector<int>& arr, int k) {
    if (arr.size() < k) return -1;
    
    // Calculate sum of first window
    int windowSum = 0;
    for (int i = 0; i < k; i++) {
        windowSum += arr[i];
    }
    
    int maxSum = windowSum;
    
    // Slide the window
    for (int i = k; i < arr.size(); i++) {
        windowSum = windowSum - arr[i - k] + arr[i];
        maxSum = max(maxSum, windowSum);
    }
    
    return maxSum;
}

// Longest substring without repeating characters
int lengthOfLongestSubstring(string s) {
    unordered_set<char> window;
    int left = 0, maxLength = 0;
    
    for (int right = 0; right < s.length(); right++) {
        // Shrink window until no duplicates
        while (window.count(s[right])) {
            window.erase(s[left]);
            left++;
        }
        
        window.insert(s[right]);
        maxLength = max(maxLength, right - left + 1);
    }
    
    return maxLength;
}

// Minimum window substring
string minWindow(string s, string t) {
    if (s.empty() || t.empty()) return "";
    
    unordered_map<char, int> required;
    for (char c : t) {
        required[c]++;
    }
    
    int left = 0, right = 0;
    int formed = 0;  // Number of unique characters in current window with desired frequency
    unordered_map<char, int> windowCounts;
    
    int minLen = INT_MAX;
    int minLeft = 0;
    
    while (right < s.length()) {
        char c = s[right];
        windowCounts[c]++;
        
        if (required.count(c) && windowCounts[c] == required[c]) {
            formed++;
        }
        
        // Try to shrink window
        while (left <= right && formed == required.size()) {
            if (right - left + 1 < minLen) {
                minLen = right - left + 1;
                minLeft = left;
            }
            
            char leftChar = s[left];
            windowCounts[leftChar]--;
            if (required.count(leftChar) && windowCounts[leftChar] < required[leftChar]) {
                formed--;
            }
            left++;
        }
        
        right++;
    }
    
    return minLen == INT_MAX ? "" : s.substr(minLeft, minLen);
}

// Subarray with given sum (positive numbers)
vector<int> subarraySum(vector<int>& arr, int targetSum) {
    int left = 0, currentSum = 0;
    
    for (int right = 0; right < arr.size(); right++) {
        currentSum += arr[right];
        
        while (currentSum > targetSum && left <= right) {
            currentSum -= arr[left];
            left++;
        }
        
        if (currentSum == targetSum) {
            return {left, right};
        }
    }
    
    return {-1, -1};
}
```

## 💻 4. Prefix Sums

### Theory
Precomputes cumulative sums to answer range sum queries in O(1) time after O(n) preprocessing.

### Code Implementation
```cpp
class PrefixSum {
private:
    vector<long long> prefixSum;
    
public:
    PrefixSum(vector<int>& arr) {
        int n = arr.size();
        prefixSum.resize(n + 1, 0);
        
        for (int i = 0; i < n; i++) {
            prefixSum[i + 1] = prefixSum[i] + arr[i];
        }
    }
    
    // Get sum of elements from index left to right (inclusive)
    long long rangeSum(int left, int right) {
        return prefixSum[right + 1] - prefixSum[left];
    }
};

// Subarray sum equals K
int subarraySum(vector<int>& arr, int k) {
    unordered_map<int, int> prefixSumCount;
    prefixSumCount[0] = 1;  // Empty prefix
    
    int count = 0, prefixSum = 0;
    
    for (int num : arr) {
        prefixSum += num;
        
        if (prefixSumCount.count(prefixSum - k)) {
            count += prefixSumCount[prefixSum - k];
        }
        
        prefixSumCount[prefixSum]++;
    }
    
    return count;
}

// Maximum size subarray sum equals k
int maxSubArrayLen(vector<int>& arr, int k) {
    unordered_map<int, int> prefixSumIndex;
    prefixSumIndex[0] = -1;  // Empty prefix at index -1
    
    int maxLen = 0, prefixSum = 0;
    
    for (int i = 0; i < arr.size(); i++) {
        prefixSum += arr[i];
        
        if (prefixSumIndex.count(prefixSum - k)) {
            maxLen = max(maxLen, i - prefixSumIndex[prefixSum - k]);
        }
        
        // Only store first occurrence for maximum length
        if (!prefixSumIndex.count(prefixSum)) {
            prefixSumIndex[prefixSum] = i;
        }
    }
    
    return maxLen;
}
```

## 💻 5. Difference Array

### Theory
Efficiently handles range updates on arrays. Instead of updating each element in range, we update only the boundaries.

### Code Implementation
```cpp
class DifferenceArray {
private:
    vector<int> diff;
    int n;
    
public:
    DifferenceArray(vector<int>& arr) {
        n = arr.size();
        diff.resize(n + 1, 0);
        
        diff[0] = arr[0];
        for (int i = 1; i < n; i++) {
            diff[i] = arr[i] - arr[i - 1];
        }
    }
    
    // Add value to range [left, right]
    void rangeUpdate(int left, int right, int value) {
        diff[left] += value;
        if (right + 1 < n) {
            diff[right + 1] -= value;
        }
    }
    
    // Get the final array after all updates
    vector<int> getArray() {
        vector<int> result(n);
        result[0] = diff[0];
        
        for (int i = 1; i < n; i++) {
            result[i] = result[i - 1] + diff[i];
        }
        
        return result;
    }
};

// Range addition queries
vector<int> getModifiedArray(int length, vector<vector<int>>& updates) {
    vector<int> arr(length, 0);
    DifferenceArray diffArr(arr);
    
    for (auto& update : updates) {
        diffArr.rangeUpdate(update[0], update[1], update[2]);
    }
    
    return diffArr.getArray();
}
```

## 💻 6. Advanced Array Techniques

### XOR Properties
```cpp
// Find single number (all others appear twice)
int singleNumber(vector<int>& arr) {
    int result = 0;
    for (int num : arr) {
        result ^= num;
    }
    return result;
}

// Find two single numbers (all others appear twice)
vector<int> singleNumberII(vector<int>& arr) {
    int xorAll = 0;
    for (int num : arr) {
        xorAll ^= num;
    }
    
    // Find rightmost set bit
    int rightmostBit = xorAll & (-xorAll);
    
    int num1 = 0, num2 = 0;
    for (int num : arr) {
        if (num & rightmostBit) {
            num1 ^= num;
        } else {
            num2 ^= num;
        }
    }
    
    return {num1, num2};
}

// Subarray XOR equals K
int subarrayXOR(vector<int>& arr, int k) {
    unordered_map<int, int> prefixXORCount;
    prefixXORCount[0] = 1;
    
    int count = 0, prefixXOR = 0;
    
    for (int num : arr) {
        prefixXOR ^= num;
        
        if (prefixXORCount.count(prefixXOR ^ k)) {
            count += prefixXORCount[prefixXOR ^ k];
        }
        
        prefixXORCount[prefixXOR]++;
    }
    
    return count;
}
```

## 🎯 Problem Categories

| Technique | Use Cases | Time Complexity |
|-----------|-----------|-----------------|
| Kadane's Algorithm | Maximum subarray, stock problems | O(n) |
| Two Pointer | Sorted array problems, palindromes | O(n) |
| Sliding Window | Substring/subarray with constraints | O(n) |
| Prefix Sums | Range queries, subarray sums | O(1) query |
| Difference Array | Range updates | O(1) update |

## 📝 Interview Tips

1. **Identify the pattern**: Look for keywords like "contiguous", "subarray", "substring"
2. **Consider constraints**: Sorted vs unsorted, positive vs negative numbers
3. **Think about optimization**: Can brute force O(n²) be reduced to O(n)?
4. **Handle edge cases**: Empty arrays, single elements, all negatives
5. **Practice variations**: Maximum, minimum, count, indices

## 🎪 Common Interview Questions

**Q**: Find the maximum sum of any contiguous subarray.
**A**: Use Kadane's algorithm - keep track of maximum ending here and maximum so far.

**Q**: Find all triplets that sum to zero.
**A**: Sort array, fix first element, use two pointers for remaining two elements.

**Q**: Find the longest substring without repeating characters.
**A**: Use sliding window with hash set to track characters in current window. 