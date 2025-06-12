# Stack & Queue Operations

## 📚 Theory - Stack

**Stack** is a linear data structure that follows the **Last In First Out (LIFO)** principle. Think of it like a stack of plates - you can only add or remove plates from the top.

### Core Characteristics:
- **LIFO**: Last element added is the first to be removed
- **Access**: Only top element is accessible
- **Operations**: Push (insert), Pop (remove), Top/Peek (view top), isEmpty, size
- **Memory**: Can be implemented using arrays or linked lists

### Real-World Applications:
- **Function calls**: Call stack in programming languages
- **Undo operations**: Text editors, browsers (back button)
- **Expression evaluation**: Converting infix to postfix
- **Backtracking**: Maze solving, game states
- **Memory management**: Stack frame allocation

### When to Use Stack:
✅ **Recursive problems** - Converting recursion to iteration  
✅ **Expression parsing** - Infix to postfix conversion  
✅ **Balanced parentheses** - Checking nested structures  
✅ **Backtracking** - DFS, maze solving  
✅ **Undo functionality** - Reversible operations  

## 💻 1. Stack Implementation

### Array-based Stack
```cpp
class Stack {
private:
    vector<int> arr;
    int topIndex;
    int capacity;
    
public:
    Stack(int size) : capacity(size), topIndex(-1) {
        arr.resize(capacity);
    }
    
    void push(int val) {
        if (isFull()) {
            cout << "Stack Overflow!" << endl;
            return;
        }
        arr[++topIndex] = val;
    }
    
    int pop() {
        if (isEmpty()) {
            cout << "Stack Underflow!" << endl;
            return -1;
        }
        return arr[topIndex--];
    }
    
    int top() {
        if (isEmpty()) return -1;
        return arr[topIndex];
    }
    
    bool isEmpty() { return topIndex == -1; }
    bool isFull() { return topIndex == capacity - 1; }
    int size() { return topIndex + 1; }
};
```

### LinkedList-based Stack
```cpp
struct StackNode {
    int data;
    StackNode* next;
    StackNode(int val) : data(val), next(nullptr) {}
};

class StackLL {
private:
    StackNode* topNode;
    int count;
    
public:
    StackLL() : topNode(nullptr), count(0) {}
    
    void push(int val) {
        StackNode* newNode = new StackNode(val);
        newNode->next = topNode;
        topNode = newNode;
        count++;
    }
    
    int pop() {
        if (isEmpty()) return -1;
        
        StackNode* temp = topNode;
        int data = temp->data;
        topNode = topNode->next;
        delete temp;
        count--;
        return data;
    }
    
    int top() {
        return isEmpty() ? -1 : topNode->data;
    }
    
    bool isEmpty() { return topNode == nullptr; }
    int size() { return count; }
};
```

## 💻 2. Infix to Postfix/Prefix Conversion

### Theory
**Infix**: Operators between operands (A + B)  
**Postfix**: Operators after operands (A B +)  
**Prefix**: Operators before operands (+ A B)  

**Why Convert?**: Postfix/Prefix eliminate the need for parentheses and operator precedence rules, making evaluation easier for computers.

### Infix to Postfix
```cpp
int precedence(char op) {
    if (op == '+' || op == '-') return 1;
    if (op == '*' || op == '/') return 2;
    if (op == '^') return 3;
    return 0;
}

bool isOperator(char c) {
    return c == '+' || c == '-' || c == '*' || c == '/' || c == '^';
}

string infixToPostfix(string infix) {
    stack<char> st;
    string postfix = "";
    
    for (char c : infix) {
        if (isalnum(c)) {
            postfix += c;
        }
        else if (c == '(') {
            st.push(c);
        }
        else if (c == ')') {
            while (!st.empty() && st.top() != '(') {
                postfix += st.top();
                st.pop();
            }
            st.pop(); // Remove '('
        }
        else if (isOperator(c)) {
            while (!st.empty() && st.top() != '(' && 
                   precedence(st.top()) >= precedence(c)) {
                postfix += st.top();
                st.pop();
            }
            st.push(c);
        }
    }
    
    while (!st.empty()) {
        postfix += st.top();
        st.pop();
    }
    
    return postfix;
}

// Evaluate postfix expression
int evaluatePostfix(string postfix) {
    stack<int> st;
    
    for (char c : postfix) {
        if (isdigit(c)) {
            st.push(c - '0');
        }
        else if (isOperator(c)) {
            int b = st.top(); st.pop();
            int a = st.top(); st.pop();
            
            switch (c) {
                case '+': st.push(a + b); break;
                case '-': st.push(a - b); break;
                case '*': st.push(a * b); break;
                case '/': st.push(a / b); break;
                case '^': st.push(pow(a, b)); break;
            }
        }
    }
    
    return st.top();
}
```

### Infix to Prefix
```cpp
string infixToPrefix(string infix) {
    // Step 1: Reverse the infix expression
    reverse(infix.begin(), infix.end());
    
    // Step 2: Replace '(' with ')' and vice versa
    for (char &c : infix) {
        if (c == '(') c = ')';
        else if (c == ')') c = '(';
    }
    
    // Step 3: Convert to postfix
    string postfix = infixToPostfix(infix);
    
    // Step 4: Reverse the result
    reverse(postfix.begin(), postfix.end());
    return postfix;
}
```

## 💻 3. Balanced Parentheses

### Theory
Check if parentheses, brackets, and braces are properly matched and nested. Each opening bracket must have a corresponding closing bracket in the correct order.

```cpp
bool isBalanced(string s) {
    stack<char> st;
    unordered_map<char, char> pairs = {
        {')', '('}, {']', '['}, {'}', '{'}
    };
    
    for (char c : s) {
        // Opening brackets
        if (c == '(' || c == '[' || c == '{') {
            st.push(c);
        }
        // Closing brackets
        else if (c == ')' || c == ']' || c == '}') {
            if (st.empty() || st.top() != pairs[c]) {
                return false;
            }
            st.pop();
        }
    }
    
    return st.empty();
}

// Enhanced version with position tracking
pair<bool, int> isBalancedWithPosition(string s) {
    stack<pair<char, int>> st;
    unordered_map<char, char> pairs = {
        {')', '('}, {']', '['}, {'}', '{'}
    };
    
    for (int i = 0; i < s.length(); i++) {
        char c = s[i];
        
        if (c == '(' || c == '[' || c == '{') {
            st.push({c, i});
        }
        else if (c == ')' || c == ']' || c == '}') {
            if (st.empty() || st.top().first != pairs[c]) {
                return {false, i};
            }
            st.pop();
        }
    }
    
    if (!st.empty()) {
        return {false, st.top().second};
    }
    
    return {true, -1};
}
```

## 💻 4. Monotonic Stack

### Theory
A stack where elements are stored in either increasing or decreasing order. Useful for problems involving "next greater/smaller element" patterns.

**Key Insight**: When we need to find the next greater element for each element in array, we can use a decreasing monotonic stack.

```cpp
// Next Greater Element to the right
vector<int> nextGreaterElement(vector<int>& arr) {
    int n = arr.size();
    vector<int> result(n, -1);
    stack<int> st; // Store indices
    
    for (int i = 0; i < n; i++) {
        // Pop elements smaller than current
        while (!st.empty() && arr[st.top()] < arr[i]) {
            result[st.top()] = arr[i];
            st.pop();
        }
        st.push(i);
    }
    
    return result;
}

// Next Greater Element to the left
vector<int> nextGreaterLeft(vector<int>& arr) {
    int n = arr.size();
    vector<int> result(n, -1);
    stack<int> st;
    
    for (int i = n - 1; i >= 0; i--) {
        while (!st.empty() && arr[st.top()] < arr[i]) {
            result[st.top()] = arr[i];
            st.pop();
        }
        st.push(i);
    }
    
    return result;
}

// Largest Rectangle in Histogram
int largestRectangleArea(vector<int>& heights) {
    stack<int> st;
    int maxArea = 0;
    int n = heights.size();
    
    for (int i = 0; i <= n; i++) {
        int h = (i == n) ? 0 : heights[i];
        
        while (!st.empty() && heights[st.top()] > h) {
            int height = heights[st.top()];
            st.pop();
            
            int width = st.empty() ? i : i - st.top() - 1;
            maxArea = max(maxArea, height * width);
        }
        
        st.push(i);
    }
    
    return maxArea;
}
```

## 💻 5. Min/Max Stack

### Theory
Stack that supports getting minimum/maximum element in O(1) time along with standard stack operations.

```cpp
class MinStack {
private:
    stack<int> dataStack;
    stack<int> minStack;
    
public:
    void push(int val) {
        dataStack.push(val);
        
        if (minStack.empty() || val <= minStack.top()) {
            minStack.push(val);
        }
    }
    
    void pop() {
        if (dataStack.empty()) return;
        
        if (dataStack.top() == minStack.top()) {
            minStack.pop();
        }
        dataStack.pop();
    }
    
    int top() {
        return dataStack.empty() ? -1 : dataStack.top();
    }
    
    int getMin() {
        return minStack.empty() ? -1 : minStack.top();
    }
};

// Space-optimized version
class MinStackOptimized {
private:
    stack<long long> st;
    long long minElement;
    
public:
    void push(int val) {
        if (st.empty()) {
            minElement = val;
            st.push(val);
        } else {
            if (val < minElement) {
                st.push(2LL * val - minElement);
                minElement = val;
            } else {
                st.push(val);
            }
        }
    }
    
    void pop() {
        if (st.empty()) return;
        
        long long top = st.top();
        st.pop();
        
        if (top < minElement) {
            minElement = 2 * minElement - top;
        }
    }
    
    int top() {
        if (st.empty()) return -1;
        
        long long top = st.top();
        return (top < minElement) ? minElement : top;
    }
    
    int getMin() {
        return st.empty() ? -1 : minElement;
    }
};
```

## 📚 Theory - Queue

**Queue** is a linear data structure that follows the **First In First Out (FIFO)** principle. Think of it like a line at a ticket counter - first person in line gets served first.

### Core Characteristics:
- **FIFO**: First element added is the first to be removed
- **Access**: Elements added at rear, removed from front
- **Operations**: Enqueue (insert), Dequeue (remove), Front (view front), Rear (view rear)
- **Types**: Simple Queue, Circular Queue, Priority Queue, Deque

### Real-World Applications:
- **CPU scheduling**: Process management in operating systems
- **Print queues**: Managing print jobs
- **Breadth-First Search**: Graph traversal
- **Buffer**: IO operations, streaming data
- **Call centers**: Handling customer calls in order

## 💻 6. Queue Implementation

### Array-based Queue
```cpp
class Queue {
private:
    vector<int> arr;
    int front, rear, capacity, count;
    
public:
    Queue(int size) : capacity(size), front(0), rear(-1), count(0) {
        arr.resize(capacity);
    }
    
    void enqueue(int val) {
        if (isFull()) {
            cout << "Queue Overflow!" << endl;
            return;
        }
        rear = (rear + 1) % capacity;
        arr[rear] = val;
        count++;
    }
    
    int dequeue() {
        if (isEmpty()) {
            cout << "Queue Underflow!" << endl;
            return -1;
        }
        int data = arr[front];
        front = (front + 1) % capacity;
        count--;
        return data;
    }
    
    int frontElement() {
        return isEmpty() ? -1 : arr[front];
    }
    
    int rearElement() {
        return isEmpty() ? -1 : arr[rear];
    }
    
    bool isEmpty() { return count == 0; }
    bool isFull() { return count == capacity; }
    int size() { return count; }
};
```

## 💻 7. Queue using Stack & Vice Versa

### Queue using Two Stacks
```cpp
class QueueUsingStacks {
private:
    stack<int> input, output;
    
    void transfer() {
        while (!input.empty()) {
            output.push(input.top());
            input.pop();
        }
    }
    
public:
    void enqueue(int val) {
        input.push(val);
    }
    
    int dequeue() {
        if (output.empty()) {
            transfer();
        }
        
        if (output.empty()) return -1;
        
        int data = output.top();
        output.pop();
        return data;
    }
    
    int front() {
        if (output.empty()) {
            transfer();
        }
        return output.empty() ? -1 : output.top();
    }
    
    bool empty() {
        return input.empty() && output.empty();
    }
};
```

### Stack using Two Queues
```cpp
class StackUsingQueues {
private:
    queue<int> q1, q2;
    
public:
    void push(int val) {
        q1.push(val);
    }
    
    int pop() {
        if (q1.empty()) return -1;
        
        // Move all elements except last to q2
        while (q1.size() > 1) {
            q2.push(q1.front());
            q1.pop();
        }
        
        int data = q1.front();
        q1.pop();
        
        // Swap queues
        swap(q1, q2);
        return data;
    }
    
    int top() {
        if (q1.empty()) return -1;
        
        while (q1.size() > 1) {
            q2.push(q1.front());
            q1.pop();
        }
        
        int data = q1.front();
        q2.push(data);
        q1.pop();
        
        swap(q1, q2);
        return data;
    }
    
    bool empty() {
        return q1.empty();
    }
};
```

## 💻 8. Circular Queue

### Theory
Circular queue treats the storage as circular, where the last position is connected to the first position. This maximizes memory utilization.

```cpp
class CircularQueue {
private:
    vector<int> arr;
    int front, rear, capacity;
    
public:
    CircularQueue(int size) : capacity(size), front(-1), rear(-1) {
        arr.resize(capacity);
    }
    
    bool enqueue(int val) {
        if (isFull()) return false;
        
        if (isEmpty()) {
            front = rear = 0;
        } else {
            rear = (rear + 1) % capacity;
        }
        
        arr[rear] = val;
        return true;
    }
    
    bool dequeue() {
        if (isEmpty()) return false;
        
        if (front == rear) {
            front = rear = -1;  // Queue becomes empty
        } else {
            front = (front + 1) % capacity;
        }
        
        return true;
    }
    
    int frontElement() {
        return isEmpty() ? -1 : arr[front];
    }
    
    int rearElement() {
        return isEmpty() ? -1 : arr[rear];
    }
    
    bool isEmpty() {
        return front == -1;
    }
    
    bool isFull() {
        return (rear + 1) % capacity == front;
    }
};
```

## 💻 9. Sliding Window Maximum

### Theory
Find the maximum element in every window of size k in an array. Using deque (double-ended queue) for efficient O(n) solution.

```cpp
vector<int> slidingWindowMaximum(vector<int>& nums, int k) {
    deque<int> dq; // Store indices
    vector<int> result;
    
    for (int i = 0; i < nums.size(); i++) {
        // Remove indices outside current window
        while (!dq.empty() && dq.front() <= i - k) {
            dq.pop_front();
        }
        
        // Remove indices of elements smaller than current
        while (!dq.empty() && nums[dq.back()] < nums[i]) {
            dq.pop_back();
        }
        
        dq.push_back(i);
        
        // Add to result if window is complete
        if (i >= k - 1) {
            result.push_back(nums[dq.front()]);
        }
    }
    
    return result;
}

// First negative number in every window of size k
vector<int> firstNegativeInWindow(vector<int>& arr, int k) {
    queue<int> negatives; // Store indices of negative numbers
    vector<int> result;
    
    for (int i = 0; i < arr.size(); i++) {
        // Remove indices outside current window
        while (!negatives.empty() && negatives.front() <= i - k) {
            negatives.pop();
        }
        
        // Add current negative number
        if (arr[i] < 0) {
            negatives.push(i);
        }
        
        // Add to result if window is complete
        if (i >= k - 1) {
            if (negatives.empty()) {
                result.push_back(0);
            } else {
                result.push_back(arr[negatives.front()]);
            }
        }
    }
    
    return result;
}
```

## 🎯 Complexity Analysis

| Operation | Stack | Queue | Circular Queue |
|-----------|-------|-------|----------------|
| Push/Enqueue | O(1) | O(1) | O(1) |
| Pop/Dequeue | O(1) | O(1) | O(1) |
| Top/Front | O(1) | O(1) | O(1) |
| Space | O(n) | O(n) | O(n) |

## 📝 Interview Tips

1. **Understand LIFO vs FIFO** - Know when to use each
2. **Practice conversions** - Infix to postfix is commonly asked
3. **Master monotonic stack** - Key for "next greater/smaller" problems
4. **Know implementation trade-offs** - Array vs LinkedList based
5. **Handle edge cases** - Empty stack/queue, overflow/underflow

## 🎪 Common Interview Questions

**Q**: How do you implement two stacks in one array?
**A**: Use two pointers, one from start and one from end, growing towards each other.

**Q**: Design a stack that supports getMin() in O(1)?
**A**: Use auxiliary stack to keep track of minimum elements or use mathematical approach.

**Q**: How do you reverse a queue using a stack?
**A**: Push all queue elements to stack, then pop from stack and enqueue back to queue. 