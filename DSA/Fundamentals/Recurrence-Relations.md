# Recurrence Relations & Master Theorem

## 📚 Theory

**Recurrence Relations** express the time complexity of recursive algorithms in terms of smaller subproblems.

**General Form**: T(n) = aT(n/b) + f(n)
- **a**: Number of subproblems
- **n/b**: Size of each subproblem  
- **f(n)**: Work done outside recursive calls

## 🎯 Master Theorem

For recurrence T(n) = aT(n/b) + f(n) where a ≥ 1, b > 1:

### Case 1: f(n) = O(n^c) where c < log_b(a)
**Result**: T(n) = Θ(n^(log_b(a)))

### Case 2: f(n) = Θ(n^c) where c = log_b(a)  
**Result**: T(n) = Θ(n^c × log n)

### Case 3: f(n) = Ω(n^c) where c > log_b(a)
**Result**: T(n) = Θ(f(n))

## 🔍 Real-Life Examples

**Binary Search**: Searching phone book by repeatedly halving pages
**Merge Sort**: Dividing tasks among team members, then combining results
**Quick Sort**: Partitioning work, then solving smaller parts

## 💻 Code Examples & Analysis

### Binary Search
```cpp
int binarySearch(vector<int>& arr, int target, int left, int right) {
    if (left > right) return -1;
    
    int mid = left + (right - left) / 2;
    if (arr[mid] == target) return mid;
    
    if (arr[mid] > target)
        return binarySearch(arr, target, left, mid - 1);
    else
        return binarySearch(arr, target, mid + 1, right);
}

// Recurrence: T(n) = T(n/2) + O(1)
// a=1, b=2, f(n)=O(1)=O(n^0)
// log_2(1) = 0, so c = log_b(a) → Case 2
// Result: T(n) = O(log n)
```

### Merge Sort
```cpp
void mergeSort(vector<int>& arr, int left, int right) {
    if (left >= right) return;
    
    int mid = left + (right - left) / 2;
    mergeSort(arr, left, mid);      // T(n/2)
    mergeSort(arr, mid + 1, right); // T(n/2)
    merge(arr, left, mid, right);   // O(n)
}

void merge(vector<int>& arr, int left, int mid, int right) {
    vector<int> temp(right - left + 1);
    int i = left, j = mid + 1, k = 0;
    
    while (i <= mid && j <= right) {
        if (arr[i] <= arr[j]) temp[k++] = arr[i++];
        else temp[k++] = arr[j++];
    }
    
    while (i <= mid) temp[k++] = arr[i++];
    while (j <= right) temp[k++] = arr[j++];
    
    for (int i = 0; i < k; i++) {
        arr[left + i] = temp[i];
    }
}

// Recurrence: T(n) = 2T(n/2) + O(n)
// a=2, b=2, f(n)=O(n)=O(n^1)
// log_2(2) = 1, so c = log_b(a) → Case 2
// Result: T(n) = O(n log n)
```

### Quick Sort (Average Case)
```cpp
int partition(vector<int>& arr, int low, int high) {
    int pivot = arr[high];
    int i = low - 1;
    
    for (int j = low; j < high; j++) {
        if (arr[j] < pivot) {
            i++;
            swap(arr[i], arr[j]);
        }
    }
    swap(arr[i + 1], arr[high]);
    return i + 1;
}

void quickSort(vector<int>& arr, int low, int high) {
    if (low < high) {
        int pi = partition(arr, low, high);
        quickSort(arr, low, pi - 1);
        quickSort(arr, pi + 1, high);
    }
}

// Average Case: T(n) = 2T(n/2) + O(n) → O(n log n)
// Worst Case: T(n) = T(n-1) + O(n) → O(n²)
```

## 🧮 Iteration Method Examples

### Example 1: T(n) = T(n-1) + c
```
T(n) = T(n-1) + c
     = T(n-2) + c + c = T(n-2) + 2c
     = T(n-3) + 3c
     = ...
     = T(1) + (n-1)c
     = O(n)
```

### Example 2: T(n) = 2T(n/2) + n
```
T(n) = 2T(n/2) + n
     = 2[2T(n/4) + n/2] + n = 4T(n/4) + 2n
     = 4[2T(n/8) + n/4] + 2n = 8T(n/8) + 3n
     = ...
     = 2^k × T(n/2^k) + k×n

When n/2^k = 1 → k = log n
T(n) = n × T(1) + n log n = O(n log n)
```

## 🎯 Common Recurrence Patterns

| Pattern | Example | Complexity |
|---------|---------|------------|
| T(n) = T(n-1) + O(1) | Linear Search | O(n) |
| T(n) = T(n/2) + O(1) | Binary Search | O(log n) |
| T(n) = 2T(n/2) + O(n) | Merge Sort | O(n log n) |
| T(n) = 2T(n/2) + O(1) | Tree Height | O(n) |
| T(n) = T(n-1) + O(n) | Selection Sort | O(n²) |

## 🚀 Interview Tips

1. **Identify the recurrence relation first**
2. **Count recursive calls and their sizes**
3. **Determine work done outside recursion**
4. **Apply Master Theorem when applicable**
5. **Use iteration method for non-standard forms**

## 📝 Step-by-Step Analysis

1. **Write the recurrence**: T(n) = ?
2. **Identify a, b, f(n)**
3. **Calculate log_b(a)**
4. **Compare f(n) with n^(log_b(a))**
5. **Apply appropriate Master Theorem case**

## 🎪 Common Interview Questions

**Q**: What's the time complexity of this recursive function?
```cpp
int mystery(int n) {
    if (n <= 1) return 1;
    return mystery(n/3) + mystery(n/3) + mystery(n/3) + n;
}
```

**A**: T(n) = 3T(n/3) + O(n)
- a=3, b=3, f(n)=O(n)
- log₃(3) = 1, so Case 2 applies
- **Result**: O(n log n) 