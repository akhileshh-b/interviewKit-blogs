# Linear Sorting Algorithms

## 📚 Theory

Linear sorting algorithms achieve O(n) or O(n+k) time complexity by making assumptions about the input data. They don't use comparisons and instead use the actual values or distribution of elements.

**Key Insight**: When we know something about the input (range, distribution), we can sort faster than O(n log n).

## 🔍 Real-Life Examples

**Counting Sort**: Like counting votes in an election with known candidates
**Radix Sort**: Like sorting papers by date (year, then month, then day)
**Bucket Sort**: Like organizing students by grade ranges (A, B, C, D, F)

## 💻 1. Counting Sort

### Theory
Counts occurrences of each element, then reconstructs sorted array. Works when range of elements (k) is not significantly larger than number of elements (n).

### Code Implementation
```cpp
void countingSort(vector<int>& arr) {
    if (arr.empty()) return;
    
    // Find range
    int maxVal = *max_element(arr.begin(), arr.end());
    int minVal = *min_element(arr.begin(), arr.end());
    int range = maxVal - minVal + 1;
    
    // Count occurrences
    vector<int> count(range, 0);
    for (int num : arr) {
        count[num - minVal]++;
    }
    
    // Reconstruct sorted array
    int index = 0;
    for (int i = 0; i < range; i++) {
        while (count[i]-- > 0) {
            arr[index++] = i + minVal;
        }
    }
}

// Stable version (preserves relative order)
void stableCountingSort(vector<int>& arr) {
    if (arr.empty()) return;
    
    int maxVal = *max_element(arr.begin(), arr.end());
    int minVal = *min_element(arr.begin(), arr.end());
    int range = maxVal - minVal + 1;
    
    vector<int> count(range, 0);
    vector<int> output(arr.size());
    
    // Count occurrences
    for (int num : arr) {
        count[num - minVal]++;
    }
    
    // Convert to cumulative count
    for (int i = 1; i < range; i++) {
        count[i] += count[i - 1];
    }
    
    // Build output array (traverse from right to maintain stability)
    for (int i = arr.size() - 1; i >= 0; i--) {
        output[count[arr[i] - minVal] - 1] = arr[i];
        count[arr[i] - minVal]--;
    }
    
    // Copy back to original array
    arr = output;
}
```

### Complexity Analysis
- **Time**: O(n + k) where k is the range of input
- **Space**: O(k) for counting array
- **Stable**: Yes (in stable version)

## 💻 2. Radix Sort

### Theory
Sorts by processing digits from least significant to most significant. Uses a stable sorting algorithm (like counting sort) as a subroutine.

### Code Implementation
```cpp
// Get maximum number to know number of digits
int getMax(vector<int>& arr) {
    return *max_element(arr.begin(), arr.end());
}

// Counting sort for a specific digit (exp = 10^i)
void countingSortByDigit(vector<int>& arr, int exp) {
    int n = arr.size();
    vector<int> output(n);
    vector<int> count(10, 0);
    
    // Count occurrences of each digit
    for (int i = 0; i < n; i++) {
        count[(arr[i] / exp) % 10]++;
    }
    
    // Convert to cumulative count
    for (int i = 1; i < 10; i++) {
        count[i] += count[i - 1];
    }
    
    // Build output array
    for (int i = n - 1; i >= 0; i--) {
        output[count[(arr[i] / exp) % 10] - 1] = arr[i];
        count[(arr[i] / exp) % 10]--;
    }
    
    // Copy back to original array
    for (int i = 0; i < n; i++) {
        arr[i] = output[i];
    }
}

void radixSort(vector<int>& arr) {
    if (arr.empty()) return;
    
    int maxVal = getMax(arr);
    
    // Do counting sort for every digit
    for (int exp = 1; maxVal / exp > 0; exp *= 10) {
        countingSortByDigit(arr, exp);
    }
}

// Radix sort for strings (MSD - Most Significant Digit)
void radixSortStrings(vector<string>& arr, int maxLen) {
    for (int pos = maxLen - 1; pos >= 0; pos--) {
        vector<vector<string>> buckets(256);  // ASCII characters
        
        for (const string& str : arr) {
            int charIndex = (pos < str.length()) ? str[pos] : 0;
            buckets[charIndex].push_back(str);
        }
        
        arr.clear();
        for (auto& bucket : buckets) {
            for (const string& str : bucket) {
                arr.push_back(str);
            }
        }
    }
}
```

### Complexity Analysis
- **Time**: O(d × (n + k)) where d is number of digits, k is range of each digit
- **Space**: O(n + k)
- **Stable**: Yes

## 💻 3. Bucket Sort

### Theory
Distributes elements into buckets, sorts individual buckets, then concatenates. Works well when input is uniformly distributed.

### Code Implementation
```cpp
void bucketSort(vector<float>& arr) {
    if (arr.empty()) return;
    
    int n = arr.size();
    vector<vector<float>> buckets(n);
    
    // Put array elements in different buckets
    for (float num : arr) {
        int bucketIndex = n * num;  // Assuming input is in [0, 1)
        buckets[bucketIndex].push_back(num);
    }
    
    // Sort individual buckets
    for (auto& bucket : buckets) {
        sort(bucket.begin(), bucket.end());
    }
    
    // Concatenate all buckets
    int index = 0;
    for (const auto& bucket : buckets) {
        for (float num : bucket) {
            arr[index++] = num;
        }
    }
}

// Generic bucket sort for integers
void bucketSortIntegers(vector<int>& arr) {
    if (arr.empty()) return;
    
    int maxVal = *max_element(arr.begin(), arr.end());
    int minVal = *min_element(arr.begin(), arr.end());
    int range = maxVal - minVal + 1;
    int bucketCount = arr.size();
    
    vector<vector<int>> buckets(bucketCount);
    
    // Distribute elements into buckets
    for (int num : arr) {
        int bucketIndex = (bucketCount * (num - minVal)) / range;
        if (bucketIndex == bucketCount) bucketIndex--;  // Handle edge case
        buckets[bucketIndex].push_back(num);
    }
    
    // Sort individual buckets and concatenate
    arr.clear();
    for (auto& bucket : buckets) {
        sort(bucket.begin(), bucket.end());
        for (int num : bucket) {
            arr.push_back(num);
        }
    }
}

// Bucket sort with custom bucket size
void bucketSortCustom(vector<int>& arr, int bucketSize) {
    if (arr.empty()) return;
    
    int maxVal = *max_element(arr.begin(), arr.end());
    int minVal = *min_element(arr.begin(), arr.end());
    int bucketCount = (maxVal - minVal) / bucketSize + 1;
    
    vector<vector<int>> buckets(bucketCount);
    
    for (int num : arr) {
        int bucketIndex = (num - minVal) / bucketSize;
        buckets[bucketIndex].push_back(num);
    }
    
    arr.clear();
    for (auto& bucket : buckets) {
        sort(bucket.begin(), bucket.end());
        for (int num : bucket) {
            arr.push_back(num);
        }
    }
}
```

### Complexity Analysis
- **Time**: O(n + k) average case, O(n²) worst case
- **Space**: O(n + k)
- **Stable**: Yes (if underlying sort is stable)

## 🎯 Comparison Table

| Algorithm | Time (Avg) | Time (Worst) | Space | Stable | When to Use |
|-----------|------------|--------------|-------|--------|-------------|
| Counting Sort | O(n + k) | O(n + k) | O(k) | Yes | Small range of integers |
| Radix Sort | O(d(n + k)) | O(d(n + k)) | O(n + k) | Yes | Fixed-width integers |
| Bucket Sort | O(n + k) | O(n²) | O(n) | Yes | Uniformly distributed data |

## 🚀 When to Use Each

### Counting Sort
✅ **Use for**: 
- Small range of integers (k ≈ n)
- When stability is required
- Frequency counting problems

❌ **Avoid for**: 
- Large range of values
- Floating-point numbers
- Memory-constrained environments

### Radix Sort
✅ **Use for**: 
- Fixed-width integers or strings
- Large datasets with small digit range
- When comparison-based sorting is too slow

❌ **Avoid for**: 
- Variable-length data
- When digit extraction is expensive

### Bucket Sort
✅ **Use for**: 
- Uniformly distributed floating-point numbers
- When input distribution is known
- Parallel processing scenarios

❌ **Avoid for**: 
- Highly skewed data
- Unknown data distribution

## 🎪 Advanced Applications

### 1. Sorting by Multiple Keys
```cpp
struct Student {
    string name;
    int grade;
    int age;
};

void sortStudents(vector<Student>& students) {
    // Sort by age first (stable)
    stableCountingSort(students, [](const Student& s) { return s.age; });
    
    // Then by grade (stable)
    stableCountingSort(students, [](const Student& s) { return s.grade; });
}
```

### 2. External Sorting with Buckets
```cpp
void externalBucketSort(const string& inputFile, const string& outputFile) {
    // Read data, distribute to bucket files
    // Sort each bucket file individually
    // Merge bucket files in order
}
```

### 3. Parallel Radix Sort
```cpp
void parallelRadixSort(vector<int>& arr) {
    // Distribute work across multiple threads
    // Each thread handles subset of data
    // Synchronize after each digit pass
}
```

## 📝 Interview Tips

1. **Identify when linear sorting is applicable**
2. **Know the constraints and assumptions**
3. **Understand stability requirements**
4. **Discuss space-time tradeoffs**
5. **Explain when to prefer over comparison-based sorts**

## 🧮 Step-by-Step Example

**Array**: [170, 45, 75, 90, 2, 802, 24, 66]

### Radix Sort Trace:
```
Original: [170, 45, 75, 90, 2, 802, 24, 66]

Sort by 1s place:
[170, 90, 2, 802, 24, 45, 75, 66]

Sort by 10s place:
[2, 802, 24, 45, 66, 170, 75, 90]

Sort by 100s place:
[2, 24, 45, 66, 75, 90, 170, 802]
```

## 🎯 Common Interview Questions

**Q**: When would you use counting sort over quick sort?

**A**: When the range of input values is small (k ≈ n) and you need O(n) time complexity. For example, sorting grades (0-100) for students.

**Q**: How does radix sort achieve linear time complexity?

**A**: It processes d digits, each taking O(n + k) time using counting sort. Total: O(d(n + k)). Since d and k are constants for fixed-width integers, it's effectively O(n).

**Q**: What's the main limitation of bucket sort?

**A**: It requires knowledge of input distribution. If data is not uniformly distributed, some buckets may have many elements while others are empty, leading to O(n²) worst-case performance. 