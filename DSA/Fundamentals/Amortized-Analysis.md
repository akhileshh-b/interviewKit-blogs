# Amortized Analysis

## 📚 Theory

**Amortized Analysis** calculates the average time per operation over a sequence of operations, even when individual operations might be expensive.

**Key Idea**: Some operations are cheap, others expensive, but the expensive ones are rare enough that the average cost remains low.

## 🎯 Three Methods

### 1. Aggregate Method
- Calculate total cost of n operations
- Divide by n to get average cost per operation

### 2. Accounting Method  
- Assign "amortized cost" to each operation
- Some operations "pay" for future expensive operations
- Maintain non-negative credit balance

### 3. Potential Method
- Define potential function Φ(D)
- Amortized cost = Actual cost + ΔΦ
- Potential represents "stored energy" for future operations

## 🔍 Real-Life Examples

**Bank Account**: Daily small deposits, occasional large withdrawals
**Traffic**: Mostly smooth flow, occasional traffic jams
**Restaurant**: Quick orders most times, complex orders occasionally

## 💻 Code Examples

### Dynamic Array (Vector) - Aggregate Method
```cpp
class DynamicArray {
private:
    int* arr;
    int size;
    int capacity;
    
public:
    DynamicArray() : size(0), capacity(1) {
        arr = new int[capacity];
    }
    
    void push_back(int val) {
        if (size == capacity) {
            // Expensive operation - resize
            capacity *= 2;
            int* newArr = new int[capacity];
            for (int i = 0; i < size; i++) {
                newArr[i] = arr[i];
            }
            delete[] arr;
            arr = newArr;
        }
        arr[size++] = val;
    }
};

/*
Analysis:
- Most insertions: O(1)
- Resize operations: O(n) but rare
- After n insertions:
  * Resizes happen at: 1, 2, 4, 8, 16, ..., n
  * Total copy cost: 1 + 2 + 4 + ... + n = 2n - 1
  * Amortized cost per insertion: (2n-1)/n = O(1)
*/
```

### Stack with Multi-Pop - Accounting Method
```cpp
class StackWithMultiPop {
private:
    vector<int> stack;
    
public:
    void push(int val) {
        stack.push_back(val);
        // Accounting: Pay $2 for this operation
        // $1 for push, $1 credit for future pop
    }
    
    int pop() {
        if (stack.empty()) return -1;
        int val = stack.back();
        stack.pop_back();
        // Use the $1 credit from push
        return val;
    }
    
    void multiPop(int k) {
        while (k > 0 && !stack.empty()) {
            pop();  // Each pop uses its prepaid credit
            k--;
        }
    }
};

/*
Accounting Analysis:
- Push: Amortized cost = $2 (actual $1 + $1 credit)
- Pop: Amortized cost = $0 (uses credit)
- MultiPop: Amortized cost = $0 (uses credits)
- Total amortized cost for n operations = O(n)
*/
```

### Binary Counter - Potential Method
```cpp
class BinaryCounter {
private:
    vector<int> bits;
    int value;
    
public:
    BinaryCounter(int size) : bits(size, 0), value(0) {}
    
    void increment() {
        int i = 0;
        while (i < bits.size() && bits[i] == 1) {
            bits[i] = 0;  // Flip 1 to 0
            i++;
        }
        if (i < bits.size()) {
            bits[i] = 1;  // Set first 0 to 1
        }
        value++;
    }
    
    int getOnes() {
        int count = 0;
        for (int bit : bits) {
            if (bit == 1) count++;
        }
        return count;
    }
};

/*
Potential Method Analysis:
- Potential Φ(D) = number of 1s in counter
- Increment operation:
  * Flips k bits from 1 to 0, then sets 1 bit to 1
  * Actual cost: k + 1
  * Potential change: -k + 1 = -(k-1)
  * Amortized cost = (k+1) + (-(k-1)) = 2
- Each increment has amortized cost O(1)
*/
```

## 🧮 Classic Examples

### 1. Fibonacci Heap Operations
```cpp
// Simplified concept
class FibonacciHeap {
public:
    void insert(int key) {
        // Actual: O(1), Amortized: O(1)
    }
    
    int extractMin() {
        // Actual: O(n), Amortized: O(log n)
        // Expensive consolidation pays for future operations
    }
    
    void decreaseKey(int key, int newKey) {
        // Actual: O(n), Amortized: O(1)
        // Cascading cuts are rare
    }
};
```

### 2. Splay Tree Operations
```cpp
struct SplayNode {
    int key;
    SplayNode* left;
    SplayNode* right;
    SplayNode(int k) : key(k), left(nullptr), right(nullptr) {}
};

class SplayTree {
private:
    SplayNode* root;
    
    SplayNode* splay(SplayNode* node, int key) {
        // Rotations bring accessed node to root
        // Frequently accessed nodes stay near top
        // Amortized O(log n) per operation
        return node; // Simplified
    }
    
public:
    void insert(int key) {
        // Actual: O(n), Amortized: O(log n)
    }
    
    bool search(int key) {
        // Actual: O(n), Amortized: O(log n)
        root = splay(root, key);
        return root && root->key == key;
    }
};
```

## 🎯 When to Use Amortized Analysis

✅ **Use When**:
- Operations have varying costs
- Expensive operations are infrequent
- Want to show average-case efficiency
- Data structure maintains some invariant

❌ **Don't Use When**:
- Need worst-case guarantees
- All operations have similar cost
- Real-time systems (need predictable timing)

## 🚀 Interview Tips

1. **Identify the expensive operations**
2. **Show they happen infrequently**
3. **Calculate total cost over sequence**
4. **Choose appropriate analysis method**
5. **Explain why amortized analysis is valid**

## 📝 Common Patterns

| Data Structure | Operation | Actual | Amortized |
|----------------|-----------|---------|-----------|
| Dynamic Array | Insert | O(n) | O(1) |
| Stack | Multi-Pop | O(n) | O(1) |
| Binary Counter | Increment | O(log n) | O(1) |
| Disjoint Set | Union/Find | O(n) | O(α(n)) |

## 🎪 Interview Questions

**Q**: Why is vector.push_back() O(1) amortized?

**A**: 
- Resize happens at powers of 2: 1→2→4→8→16...
- For n insertions, total copies = 1+2+4+...+n/2 < n
- Average cost per insertion = n/n = O(1)

**Q**: Explain the difference between worst-case and amortized complexity.

**A**:
- **Worst-case**: Maximum time for any single operation
- **Amortized**: Average time per operation over sequence
- Example: Hash table insertion is O(n) worst-case but O(1) amortized 