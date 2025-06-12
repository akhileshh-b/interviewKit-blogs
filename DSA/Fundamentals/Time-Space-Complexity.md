# Time & Space Complexity Analysis

## 📚 Theory

**Time Complexity** measures how the runtime of an algorithm grows with input size.
**Space Complexity** measures how much extra memory an algorithm uses relative to input size.

### Big O Notation Types:
- **Best Case (Ω)**: Minimum time/space needed
- **Average Case (Θ)**: Expected time/space for typical input
- **Worst Case (O)**: Maximum time/space needed

### Common Complexities (Best to Worst):
1. **O(1)** - Constant
2. **O(log n)** - Logarithmic
3. **O(n)** - Linear
4. **O(n log n)** - Linearithmic
5. **O(n²)** - Quadratic
6. **O(2ⁿ)** - Exponential

## 🔍 Real-Life Examples

**O(1)**: Finding a book when you know the exact shelf number
**O(log n)**: Finding a word in dictionary (binary search)
**O(n)**: Reading every page of a book
**O(n²)**: Comparing every student with every other student in class

## 💻 Code Examples

### O(1) - Constant Time
```cpp
int getFirstElement(vector<int>& arr) {
    return arr[0];  // Always takes same time
}
```

### O(log n) - Logarithmic Time
```cpp
int binarySearch(vector<int>& arr, int target) {
    int left = 0, right = arr.size() - 1;
    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (arr[mid] == target) return mid;
        if (arr[mid] < target) left = mid + 1;
        else right = mid - 1;
    }
    return -1;
}
```

### O(n) - Linear Time
```cpp
int findMax(vector<int>& arr) {
    int maxVal = arr[0];
    for (int i = 1; i < arr.size(); i++) {
        maxVal = max(maxVal, arr[i]);
    }
    return maxVal;
}
```

### O(n²) - Quadratic Time
```cpp
void bubbleSort(vector<int>& arr) {
    int n = arr.size();
    for (int i = 0; i < n-1; i++) {
        for (int j = 0; j < n-i-1; j++) {
            if (arr[j] > arr[j+1]) {
                swap(arr[j], arr[j+1]);
            }
        }
    }
}
```

## 🎯 Space Complexity Examples

### O(1) - Constant Space
```cpp
void swapElements(int& a, int& b) {
    int temp = a;  // Only using constant extra space
    a = b;
    b = temp;
}
```

### O(n) - Linear Space
```cpp
vector<int> createCopy(vector<int>& arr) {
    vector<int> copy = arr;  // Creating copy needs O(n) space
    return copy;
}
```

### O(n) - Recursive Space
```cpp
int factorial(int n) {
    if (n <= 1) return 1;
    return n * factorial(n-1);  // Call stack uses O(n) space
}
```

## 🚀 Interview Tips

1. **Always analyze both time and space complexity**
2. **Consider best, average, and worst cases**
3. **Drop constants and lower-order terms**: O(2n + 5) → O(n)
4. **Recursive algorithms**: Space complexity includes call stack
5. **In-place algorithms**: Modify input without extra space

## 📝 Quick Analysis Checklist

- **Loops**: Each nested loop multiplies complexity
- **Recursion**: Depth × work per call
- **Data Structures**: Know their operation complexities
- **Divide & Conquer**: Often O(n log n)

## 🎪 Common Mistakes to Avoid

❌ Confusing time with space complexity
❌ Forgetting about recursive call stack space
❌ Not considering all cases (best/avg/worst)
❌ Including input size in space complexity calculation 