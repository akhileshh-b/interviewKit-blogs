# Searching Algorithms

## 📚 Theory

Searching algorithms find the position of a target element in a data structure. The choice of algorithm depends on whether the data is sorted and the specific requirements of the application.

## 🔍 Real-Life Examples

**Linear Search**: Looking for a book by checking each shelf one by one
**Binary Search**: Finding a word in dictionary by opening to middle and narrowing down
**Exponential Search**: Finding a page in a thick book by doubling page jumps
**Interpolation Search**: Guessing where a name appears in phone book based on alphabetical position

## 💻 1. Linear Search

### Theory
Sequentially checks each element until target is found or end is reached. Works on both sorted and unsorted arrays.

### Code Implementation
```cpp
int linearSearch(vector<int>& arr, int target) {
    for (int i = 0; i < arr.size(); i++) {
        if (arr[i] == target) {
            return i;  // Return index of found element
        }
    }
    return -1;  // Element not found
}

// Find all occurrences
vector<int> linearSearchAll(vector<int>& arr, int target) {
    vector<int> indices;
    for (int i = 0; i < arr.size(); i++) {
        if (arr[i] == target) {
            indices.push_back(i);
        }
    }
    return indices;
}

// Linear search with early termination for sorted array
int linearSearchSorted(vector<int>& arr, int target) {
    for (int i = 0; i < arr.size(); i++) {
        if (arr[i] == target) {
            return i;
        }
        if (arr[i] > target) {
            break;  // No point searching further in sorted array
        }
    }
    return -1;
}
```

### Complexity Analysis
- **Time**: O(n) - worst case checks all elements
- **Space**: O(1) - constant extra space
- **Best Case**: O(1) - element found at first position

## 💻 2. Binary Search

### Theory
Repeatedly divides search space in half by comparing target with middle element. Only works on sorted arrays.

### Code Implementation
```cpp
// Iterative binary search
int binarySearch(vector<int>& arr, int target) {
    int left = 0, right = arr.size() - 1;
    
    while (left <= right) {
        int mid = left + (right - left) / 2;  // Avoid overflow
        
        if (arr[mid] == target) {
            return mid;
        }
        
        if (arr[mid] < target) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    
    return -1;  // Element not found
}

// Recursive binary search
int binarySearchRecursive(vector<int>& arr, int target, int left, int right) {
    if (left > right) {
        return -1;
    }
    
    int mid = left + (right - left) / 2;
    
    if (arr[mid] == target) {
        return mid;
    }
    
    if (arr[mid] < target) {
        return binarySearchRecursive(arr, target, mid + 1, right);
    } else {
        return binarySearchRecursive(arr, target, left, mid - 1);
    }
}

// Find first occurrence of target
int findFirst(vector<int>& arr, int target) {
    int left = 0, right = arr.size() - 1;
    int result = -1;
    
    while (left <= right) {
        int mid = left + (right - left) / 2;
        
        if (arr[mid] == target) {
            result = mid;
            right = mid - 1;  // Continue searching left
        } else if (arr[mid] < target) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    
    return result;
}

// Find last occurrence of target
int findLast(vector<int>& arr, int target) {
    int left = 0, right = arr.size() - 1;
    int result = -1;
    
    while (left <= right) {
        int mid = left + (right - left) / 2;
        
        if (arr[mid] == target) {
            result = mid;
            left = mid + 1;  // Continue searching right
        } else if (arr[mid] < target) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    
    return result;
}
```

### Complexity Analysis
- **Time**: O(log n) - halves search space each iteration
- **Space**: O(1) iterative, O(log n) recursive (call stack)
- **Prerequisite**: Array must be sorted

## 💻 3. Exponential Search

### Theory
First finds range where element might exist by exponentially increasing the bound, then performs binary search in that range.

### Code Implementation
```cpp
int exponentialSearch(vector<int>& arr, int target) {
    int n = arr.size();
    
    // If element is at first position
    if (arr[0] == target) {
        return 0;
    }
    
    // Find range for binary search by repeated doubling
    int bound = 1;
    while (bound < n && arr[bound] < target) {
        bound *= 2;
    }
    
    // Perform binary search in found range
    int left = bound / 2;
    int right = min(bound, n - 1);
    
    return binarySearch(arr, target, left, right);
}

// Helper function for binary search in range
int binarySearch(vector<int>& arr, int target, int left, int right) {
    while (left <= right) {
        int mid = left + (right - left) / 2;
        
        if (arr[mid] == target) {
            return mid;
        }
        
        if (arr[mid] < target) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    
    return -1;
}
```

### Complexity Analysis
- **Time**: O(log n) - O(log i) to find range + O(log i) for binary search
- **Space**: O(1)
- **Use Case**: When size of array is unknown or very large

## 💻 4. Interpolation Search

### Theory
Estimates position of target based on its value relative to the range of values. Works best with uniformly distributed data.

### Code Implementation
```cpp
int interpolationSearch(vector<int>& arr, int target) {
    int left = 0, right = arr.size() - 1;
    
    while (left <= right && target >= arr[left] && target <= arr[right]) {
        // If array has only one element
        if (left == right) {
            return (arr[left] == target) ? left : -1;
        }
        
        // Estimate position using interpolation formula
        int pos = left + ((double)(target - arr[left]) / 
                         (arr[right] - arr[left])) * (right - left);
        
        if (arr[pos] == target) {
            return pos;
        }
        
        if (arr[pos] < target) {
            left = pos + 1;
        } else {
            right = pos - 1;
        }
    }
    
    return -1;
}

// Safe interpolation search (handles edge cases)
int safeInterpolationSearch(vector<int>& arr, int target) {
    int left = 0, right = arr.size() - 1;
    
    while (left <= right && target >= arr[left] && target <= arr[right]) {
        if (left == right) {
            return (arr[left] == target) ? left : -1;
        }
        
        // Avoid division by zero
        if (arr[right] == arr[left]) {
            return (arr[left] == target) ? left : -1;
        }
        
        int pos = left + ((double)(target - arr[left]) / 
                         (arr[right] - arr[left])) * (right - left);
        
        // Ensure pos is within bounds
        pos = max(left, min(pos, right));
        
        if (arr[pos] == target) {
            return pos;
        }
        
        if (arr[pos] < target) {
            left = pos + 1;
        } else {
            right = pos - 1;
        }
    }
    
    return -1;
}
```

### Complexity Analysis
- **Time**: O(log log n) average case, O(n) worst case
- **Space**: O(1)
- **Best For**: Uniformly distributed sorted data

## 🎯 Comparison Table

| Algorithm | Time (Best) | Time (Avg) | Time (Worst) | Space | Prerequisite |
|-----------|-------------|------------|--------------|-------|--------------|
| Linear Search | O(1) | O(n) | O(n) | O(1) | None |
| Binary Search | O(1) | O(log n) | O(log n) | O(1) | Sorted array |
| Exponential Search | O(1) | O(log n) | O(log n) | O(1) | Sorted array |
| Interpolation Search | O(1) | O(log log n) | O(n) | O(1) | Sorted, uniform |

## 🚀 Advanced Search Techniques

### 1. Ternary Search
```cpp
int ternarySearch(vector<int>& arr, int target, int left, int right) {
    if (left > right) return -1;
    
    int mid1 = left + (right - left) / 3;
    int mid2 = right - (right - left) / 3;
    
    if (arr[mid1] == target) return mid1;
    if (arr[mid2] == target) return mid2;
    
    if (target < arr[mid1]) {
        return ternarySearch(arr, target, left, mid1 - 1);
    } else if (target > arr[mid2]) {
        return ternarySearch(arr, target, mid2 + 1, right);
    } else {
        return ternarySearch(arr, target, mid1 + 1, mid2 - 1);
    }
}
```

### 2. Jump Search
```cpp
int jumpSearch(vector<int>& arr, int target) {
    int n = arr.size();
    int step = sqrt(n);
    int prev = 0;
    
    // Find block where element may be present
    while (arr[min(step, n) - 1] < target) {
        prev = step;
        step += sqrt(n);
        if (prev >= n) return -1;
    }
    
    // Linear search in identified block
    while (arr[prev] < target) {
        prev++;
        if (prev == min(step, n)) return -1;
    }
    
    return (arr[prev] == target) ? prev : -1;
}
```

### 3. Fibonacci Search
```cpp
int fibonacciSearch(vector<int>& arr, int target) {
    int n = arr.size();
    
    // Initialize Fibonacci numbers
    int fib2 = 0;  // (m-2)th Fibonacci number
    int fib1 = 1;  // (m-1)th Fibonacci number
    int fib = fib2 + fib1;  // mth Fibonacci number
    
    // Find smallest Fibonacci number >= n
    while (fib < n) {
        fib2 = fib1;
        fib1 = fib;
        fib = fib2 + fib1;
    }
    
    int offset = -1;
    
    while (fib > 1) {
        int i = min(offset + fib2, n - 1);
        
        if (arr[i] < target) {
            fib = fib1;
            fib1 = fib2;
            fib2 = fib - fib1;
            offset = i;
        } else if (arr[i] > target) {
            fib = fib2;
            fib1 = fib1 - fib2;
            fib2 = fib - fib1;
        } else {
            return i;
        }
    }
    
    // Check last element
    if (fib1 && offset + 1 < n && arr[offset + 1] == target) {
        return offset + 1;
    }
    
    return -1;
}
```

## 📝 Interview Tips

1. **Always ask if array is sorted**
2. **Consider edge cases (empty array, single element)**
3. **Understand when to use each algorithm**
4. **Know how to find first/last occurrence**
5. **Practice binary search variations**

## 🧮 Binary Search Applications

### 1. Search in Rotated Sorted Array
```cpp
int searchRotated(vector<int>& arr, int target) {
    int left = 0, right = arr.size() - 1;
    
    while (left <= right) {
        int mid = left + (right - left) / 2;
        
        if (arr[mid] == target) return mid;
        
        // Left half is sorted
        if (arr[left] <= arr[mid]) {
            if (target >= arr[left] && target < arr[mid]) {
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
        // Right half is sorted
        else {
            if (target > arr[mid] && target <= arr[right]) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
    }
    
    return -1;
}
```

### 2. Find Peak Element
```cpp
int findPeakElement(vector<int>& arr) {
    int left = 0, right = arr.size() - 1;
    
    while (left < right) {
        int mid = left + (right - left) / 2;
        
        if (arr[mid] > arr[mid + 1]) {
            right = mid;
        } else {
            left = mid + 1;
        }
    }
    
    return left;
}
```

## 🎯 Common Interview Questions

**Q**: When would you use exponential search over binary search?

**A**: When you don't know the size of the array or when the target is likely to be near the beginning of a very large sorted array.

**Q**: Why is interpolation search better than binary search for uniformly distributed data?

**A**: It can estimate the position more accurately, reducing the search space more effectively than just halving it.

**Q**: How do you search in an infinite sorted array?

**A**: Use exponential search to find bounds, then binary search within those bounds. 