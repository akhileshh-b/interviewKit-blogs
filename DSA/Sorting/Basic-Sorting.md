# Basic Sorting Algorithms

## 📚 Theory

Basic sorting algorithms are fundamental building blocks that help understand sorting principles. While not efficient for large datasets, they're important for interviews and understanding optimization techniques.

## 🔍 Real-Life Examples

**Bubble Sort**: Like bubbles rising to surface - largest elements "bubble up"
**Selection Sort**: Like picking the smallest person from a group repeatedly
**Insertion Sort**: Like arranging playing cards in your hand one by one

## 💻 1. Bubble Sort

### Theory
Repeatedly compares adjacent elements and swaps if they're in wrong order. After each pass, the largest element "bubbles" to its correct position.

### Code Implementation
```cpp
void bubbleSort(vector<int>& arr) {
    int n = arr.size();
    
    for (int i = 0; i < n - 1; i++) {
        bool swapped = false;  // Optimization flag
        
        for (int j = 0; j < n - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                swap(arr[j], arr[j + 1]);
                swapped = true;
            }
        }
        
        // If no swapping occurred, array is sorted
        if (!swapped) break;
    }
}

// Example usage
void demonstrateBubbleSort() {
    vector<int> arr = {64, 34, 25, 12, 22, 11, 90};
    cout << "Original: ";
    for (int x : arr) cout << x << " ";
    
    bubbleSort(arr);
    
    cout << "\nSorted: ";
    for (int x : arr) cout << x << " ";
}
```

### Complexity Analysis
- **Time**: O(n²) worst/average, O(n) best (optimized version)
- **Space**: O(1) - in-place sorting
- **Stable**: Yes - equal elements maintain relative order

## 💻 2. Selection Sort

### Theory
Finds the minimum element and places it at the beginning. Repeats for remaining unsorted portion.

### Code Implementation
```cpp
void selectionSort(vector<int>& arr) {
    int n = arr.size();
    
    for (int i = 0; i < n - 1; i++) {
        int minIndex = i;
        
        // Find minimum element in remaining array
        for (int j = i + 1; j < n; j++) {
            if (arr[j] < arr[minIndex]) {
                minIndex = j;
            }
        }
        
        // Swap minimum element with first element
        if (minIndex != i) {
            swap(arr[i], arr[minIndex]);
        }
    }
}

// Variation: Find maximum and place at end
void selectionSortMax(vector<int>& arr) {
    int n = arr.size();
    
    for (int i = n - 1; i > 0; i--) {
        int maxIndex = 0;
        
        for (int j = 1; j <= i; j++) {
            if (arr[j] > arr[maxIndex]) {
                maxIndex = j;
            }
        }
        
        swap(arr[i], arr[maxIndex]);
    }
}
```

### Complexity Analysis
- **Time**: O(n²) in all cases - always scans entire remaining array
- **Space**: O(1) - in-place sorting
- **Stable**: No - relative order of equal elements may change

## 💻 3. Insertion Sort

### Theory
Builds sorted array one element at a time by inserting each element into its correct position among previously sorted elements.

### Code Implementation
```cpp
void insertionSort(vector<int>& arr) {
    int n = arr.size();
    
    for (int i = 1; i < n; i++) {
        int key = arr[i];
        int j = i - 1;
        
        // Move elements greater than key one position ahead
        while (j >= 0 && arr[j] > key) {
            arr[j + 1] = arr[j];
            j--;
        }
        
        arr[j + 1] = key;
    }
}

// Binary insertion sort - optimization using binary search
void binaryInsertionSort(vector<int>& arr) {
    int n = arr.size();
    
    for (int i = 1; i < n; i++) {
        int key = arr[i];
        int left = 0, right = i - 1;
        
        // Binary search for insertion position
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (arr[mid] > key) {
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
        
        // Shift elements and insert
        for (int j = i - 1; j >= left; j--) {
            arr[j + 1] = arr[j];
        }
        arr[left] = key;
    }
}
```

### Complexity Analysis
- **Time**: O(n²) worst/average, O(n) best (nearly sorted)
- **Space**: O(1) - in-place sorting
- **Stable**: Yes - equal elements maintain relative order

## 🎯 Comparison Table

| Algorithm | Best Case | Average Case | Worst Case | Space | Stable | Adaptive |
|-----------|-----------|--------------|------------|-------|--------|----------|
| Bubble Sort | O(n) | O(n²) | O(n²) | O(1) | Yes | Yes |
| Selection Sort | O(n²) | O(n²) | O(n²) | O(1) | No | No |
| Insertion Sort | O(n) | O(n²) | O(n²) | O(1) | Yes | Yes |

## 🚀 When to Use Each

### Bubble Sort
✅ **Use for**: Educational purposes, very small datasets
❌ **Avoid for**: Any practical application

### Selection Sort  
✅ **Use for**: When memory writes are expensive, small datasets
❌ **Avoid for**: When stability is required

### Insertion Sort
✅ **Use for**: Small datasets, nearly sorted data, as subroutine in hybrid algorithms
❌ **Avoid for**: Large random datasets

## 🎪 Interview Tips

1. **Know the basic implementations by heart**
2. **Understand stability and adaptiveness**
3. **Explain when each algorithm is preferred**
4. **Discuss optimizations (early termination, binary insertion)**
5. **Compare with advanced algorithms**

## 📝 Common Interview Questions

**Q**: Which sorting algorithm would you choose for a small array of 10 elements?

**A**: Insertion sort - it's simple, adaptive, and performs well on small datasets.

**Q**: How can you optimize bubble sort?

**A**: 
1. Add a flag to detect if array is already sorted
2. Reduce the range in each pass (last i elements are already sorted)

**Q**: Why is selection sort not stable?

**A**: When we swap the minimum element with the first element, we might change the relative order of equal elements.

## 🧮 Step-by-Step Example

**Array**: [64, 25, 12, 22, 11]

### Insertion Sort Trace:
```
Initial: [64, 25, 12, 22, 11]
Pass 1:  [25, 64, 12, 22, 11]  // Insert 25
Pass 2:  [12, 25, 64, 22, 11]  // Insert 12
Pass 3:  [12, 22, 25, 64, 11]  // Insert 22
Pass 4:  [11, 12, 22, 25, 64]  // Insert 11
```

## 🎯 Optimization Techniques

### 1. Early Termination (Bubble Sort)
```cpp
if (!swapped) break;  // Array is sorted
```

### 2. Binary Search (Insertion Sort)
```cpp
// Find position using binary search instead of linear search
```

### 3. Cocktail Sort (Bubble Sort Variant)
```cpp
// Bubble in both directions alternately
void cocktailSort(vector<int>& arr) {
    bool swapped = true;
    int start = 0, end = arr.size() - 1;
    
    while (swapped) {
        swapped = false;
        
        // Forward pass
        for (int i = start; i < end; i++) {
            if (arr[i] > arr[i + 1]) {
                swap(arr[i], arr[i + 1]);
                swapped = true;
            }
        }
        end--;
        
        if (!swapped) break;
        
        // Backward pass
        for (int i = end; i > start; i--) {
            if (arr[i] < arr[i - 1]) {
                swap(arr[i], arr[i - 1]);
                swapped = true;
            }
        }
        start++;
    }
}
``` 