# Advanced Sorting Algorithms

## 📚 Theory

Advanced sorting algorithms use divide-and-conquer or heap-based approaches to achieve better time complexity than basic sorting algorithms. They're essential for handling large datasets efficiently.

## 🔍 Real-Life Examples

**Merge Sort**: Like organizing two sorted piles of papers into one sorted pile
**Quick Sort**: Like organizing people by height using a reference person
**Heap Sort**: Like repeatedly picking the tallest person from a group

## 💻 1. Merge Sort

### Theory
Divides array into halves, recursively sorts them, then merges the sorted halves. Classic divide-and-conquer algorithm.

### Code Implementation
```cpp
void merge(vector<int>& arr, int left, int mid, int right) {
    int n1 = mid - left + 1;
    int n2 = right - mid;
    
    // Create temporary arrays
    vector<int> leftArr(n1), rightArr(n2);
    
    // Copy data to temporary arrays
    for (int i = 0; i < n1; i++)
        leftArr[i] = arr[left + i];
    for (int j = 0; j < n2; j++)
        rightArr[j] = arr[mid + 1 + j];
    
    // Merge the temporary arrays back
    int i = 0, j = 0, k = left;
    
    while (i < n1 && j < n2) {
        if (leftArr[i] <= rightArr[j]) {
            arr[k] = leftArr[i];
            i++;
        } else {
            arr[k] = rightArr[j];
            j++;
        }
        k++;
    }
    
    // Copy remaining elements
    while (i < n1) {
        arr[k] = leftArr[i];
        i++;
        k++;
    }
    
    while (j < n2) {
        arr[k] = rightArr[j];
        j++;
        k++;
    }
}

void mergeSort(vector<int>& arr, int left, int right) {
    if (left < right) {
        int mid = left + (right - left) / 2;
        
        mergeSort(arr, left, mid);
        mergeSort(arr, mid + 1, right);
        merge(arr, left, mid, right);
    }
}

// Wrapper function
void mergeSort(vector<int>& arr) {
    mergeSort(arr, 0, arr.size() - 1);
}
```

### Complexity Analysis
- **Time**: O(n log n) in all cases
- **Space**: O(n) - requires additional space for merging
- **Stable**: Yes - maintains relative order of equal elements

## 💻 2. Quick Sort

### Theory
Selects a 'pivot' element, partitions array around pivot, then recursively sorts partitions. Pivot ends up in its final sorted position.

### Code Implementation
```cpp
int partition(vector<int>& arr, int low, int high) {
    int pivot = arr[high];  // Choose last element as pivot
    int i = low - 1;        // Index of smaller element
    
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

// Wrapper function
void quickSort(vector<int>& arr) {
    quickSort(arr, 0, arr.size() - 1);
}

// Randomized Quick Sort (Better average case)
int randomizedPartition(vector<int>& arr, int low, int high) {
    int randomIndex = low + rand() % (high - low + 1);
    swap(arr[randomIndex], arr[high]);
    return partition(arr, low, high);
}

void randomizedQuickSort(vector<int>& arr, int low, int high) {
    if (low < high) {
        int pi = randomizedPartition(arr, low, high);
        randomizedQuickSort(arr, low, pi - 1);
        randomizedQuickSort(arr, pi + 1, high);
    }
}
```

### Complexity Analysis
- **Time**: O(n log n) average, O(n²) worst case
- **Space**: O(log n) average (recursion stack)
- **Stable**: No - relative order may change during partitioning

## 💻 3. Heap Sort

### Theory
Builds a max heap from array, then repeatedly extracts maximum element and places it at the end.

### Code Implementation
```cpp
void heapify(vector<int>& arr, int n, int i) {
    int largest = i;        // Initialize largest as root
    int left = 2 * i + 1;   // Left child
    int right = 2 * i + 2;  // Right child
    
    // If left child is larger than root
    if (left < n && arr[left] > arr[largest])
        largest = left;
    
    // If right child is larger than largest so far
    if (right < n && arr[right] > arr[largest])
        largest = right;
    
    // If largest is not root
    if (largest != i) {
        swap(arr[i], arr[largest]);
        heapify(arr, n, largest);  // Recursively heapify affected subtree
    }
}

void heapSort(vector<int>& arr) {
    int n = arr.size();
    
    // Build heap (rearrange array)
    for (int i = n / 2 - 1; i >= 0; i--)
        heapify(arr, n, i);
    
    // Extract elements from heap one by one
    for (int i = n - 1; i > 0; i--) {
        swap(arr[0], arr[i]);  // Move current root to end
        heapify(arr, i, 0);    // Call heapify on reduced heap
    }
}

// Iterative heapify (space optimized)
void heapifyIterative(vector<int>& arr, int n, int i) {
    while (true) {
        int largest = i;
        int left = 2 * i + 1;
        int right = 2 * i + 2;
        
        if (left < n && arr[left] > arr[largest])
            largest = left;
        
        if (right < n && arr[right] > arr[largest])
            largest = right;
        
        if (largest == i) break;
        
        swap(arr[i], arr[largest]);
        i = largest;
    }
}
```

### Complexity Analysis
- **Time**: O(n log n) in all cases
- **Space**: O(1) - in-place sorting
- **Stable**: No - heap operations may change relative order

## 🎯 Comparison Table

| Algorithm | Best Case | Average Case | Worst Case | Space | Stable | In-Place |
|-----------|-----------|--------------|------------|-------|--------|----------|
| Merge Sort | O(n log n) | O(n log n) | O(n log n) | O(n) | Yes | No |
| Quick Sort | O(n log n) | O(n log n) | O(n²) | O(log n) | No | Yes |
| Heap Sort | O(n log n) | O(n log n) | O(n log n) | O(1) | No | Yes |

## 🚀 When to Use Each

### Merge Sort
✅ **Use for**: When stability is required, guaranteed O(n log n), external sorting
❌ **Avoid for**: When space is limited

### Quick Sort
✅ **Use for**: General purpose, when average case matters more than worst case
❌ **Avoid for**: When worst-case guarantee is needed

### Heap Sort
✅ **Use for**: When space is limited, guaranteed O(n log n)
❌ **Avoid for**: When stability is required

## 🎪 Optimizations & Variations

### 1. Hybrid Quick Sort
```cpp
void hybridQuickSort(vector<int>& arr, int low, int high) {
    if (high - low < 10) {
        // Use insertion sort for small arrays
        insertionSort(arr, low, high);
    } else if (low < high) {
        int pi = partition(arr, low, high);
        hybridQuickSort(arr, low, pi - 1);
        hybridQuickSort(arr, pi + 1, high);
    }
}
```

### 2. 3-Way Quick Sort (Dutch National Flag)
```cpp
void quickSort3Way(vector<int>& arr, int low, int high) {
    if (low >= high) return;
    
    int lt = low, gt = high;
    int pivot = arr[low];
    int i = low;
    
    while (i <= gt) {
        if (arr[i] < pivot) {
            swap(arr[lt++], arr[i++]);
        } else if (arr[i] > pivot) {
            swap(arr[i], arr[gt--]);
        } else {
            i++;
        }
    }
    
    quickSort3Way(arr, low, lt - 1);
    quickSort3Way(arr, gt + 1, high);
}
```

### 3. Bottom-Up Merge Sort
```cpp
void bottomUpMergeSort(vector<int>& arr) {
    int n = arr.size();
    
    for (int size = 1; size < n; size *= 2) {
        for (int left = 0; left < n - size; left += 2 * size) {
            int mid = left + size - 1;
            int right = min(left + 2 * size - 1, n - 1);
            merge(arr, left, mid, right);
        }
    }
}
```

## 📝 Interview Tips

1. **Know when to use each algorithm**
2. **Understand the partitioning logic in Quick Sort**
3. **Explain heap property and heapify operation**
4. **Discuss optimizations and variations**
5. **Analyze space-time tradeoffs**

## 🧮 Step-by-Step Example

**Array**: [38, 27, 43, 3, 9, 82, 10]

### Quick Sort Trace:
```
Initial: [38, 27, 43, 3, 9, 82, 10]
Pivot=10: [3, 9, 10, 38, 27, 43, 82]
Left=[3, 9], Right=[38, 27, 43, 82]
Continue recursively...
```

## 🎯 Common Interview Questions

**Q**: Why is Quick Sort preferred over Merge Sort?

**A**: 
- Quick Sort is in-place (O(1) space vs O(n))
- Better cache performance
- Faster in practice for random data

**Q**: When would you choose Heap Sort over Quick Sort?

**A**:
- When you need guaranteed O(n log n) worst-case
- When space is extremely limited
- In real-time systems where predictable performance is crucial

**Q**: How do you handle duplicate elements in Quick Sort?

**A**: Use 3-way partitioning (Dutch National Flag algorithm) to handle duplicates efficiently.

## 🚀 Advanced Concepts

### Introspective Sort (Introsort)
```cpp
// Hybrid of Quick Sort, Heap Sort, and Insertion Sort
// Used in C++ STL sort()
void introsort(vector<int>& arr, int low, int high, int depth) {
    if (high - low < 16) {
        insertionSort(arr, low, high);
    } else if (depth == 0) {
        heapSort(arr, low, high);
    } else {
        int pi = partition(arr, low, high);
        introsort(arr, low, pi - 1, depth - 1);
        introsort(arr, pi + 1, high, depth - 1);
    }
}
``` 