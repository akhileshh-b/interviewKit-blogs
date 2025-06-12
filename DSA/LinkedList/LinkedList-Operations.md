# LinkedList Operations

## 📚 Theory

LinkedList is a linear data structure where elements are stored in nodes, and each node contains data and a pointer to the next node. Unlike arrays, LinkedLists don't store elements in contiguous memory locations.

**Key Advantages**: Dynamic size, efficient insertion/deletion at beginning
**Key Disadvantages**: No random access, extra memory for pointers

## 🔍 Real-Life Examples

**Singly LinkedList**: Train cars connected in sequence
**Doubly LinkedList**: People holding hands in both directions
**Circular LinkedList**: People sitting around a circular table
**LRU Cache**: Most recently used items at front, least used at back

## 💻 1. Basic LinkedList Structure

### Node Definition
```cpp
struct ListNode {
    int val;
    ListNode* next;
    
    ListNode() : val(0), next(nullptr) {}
    ListNode(int x) : val(x), next(nullptr) {}
    ListNode(int x, ListNode* next) : val(x), next(next) {}
};

// Doubly LinkedList Node
struct DoublyListNode {
    int val;
    DoublyListNode* prev;
    DoublyListNode* next;
    
    DoublyListNode(int x) : val(x), prev(nullptr), next(nullptr) {}
};
```

### Basic Operations
```cpp
class LinkedList {
private:
    ListNode* head;
    int size;
    
public:
    LinkedList() : head(nullptr), size(0) {}
    
    // Insert at beginning
    void insertAtHead(int val) {
        ListNode* newNode = new ListNode(val);
        newNode->next = head;
        head = newNode;
        size++;
    }
    
    // Insert at end
    void insertAtTail(int val) {
        ListNode* newNode = new ListNode(val);
        
        if (!head) {
            head = newNode;
        } else {
            ListNode* curr = head;
            while (curr->next) {
                curr = curr->next;
            }
            curr->next = newNode;
        }
        size++;
    }
    
    // Insert at specific position
    void insertAt(int pos, int val) {
        if (pos < 0 || pos > size) return;
        
        if (pos == 0) {
            insertAtHead(val);
            return;
        }
        
        ListNode* newNode = new ListNode(val);
        ListNode* curr = head;
        
        for (int i = 0; i < pos - 1; i++) {
            curr = curr->next;
        }
        
        newNode->next = curr->next;
        curr->next = newNode;
        size++;
    }
    
    // Delete by value
    void deleteValue(int val) {
        if (!head) return;
        
        if (head->val == val) {
            ListNode* temp = head;
            head = head->next;
            delete temp;
            size--;
            return;
        }
        
        ListNode* curr = head;
        while (curr->next && curr->next->val != val) {
            curr = curr->next;
        }
        
        if (curr->next) {
            ListNode* temp = curr->next;
            curr->next = curr->next->next;
            delete temp;
            size--;
        }
    }
    
    // Search for value
    bool search(int val) {
        ListNode* curr = head;
        while (curr) {
            if (curr->val == val) return true;
            curr = curr->next;
        }
        return false;
    }
    
    // Display list
    void display() {
        ListNode* curr = head;
        while (curr) {
            cout << curr->val << " -> ";
            curr = curr->next;
        }
        cout << "NULL" << endl;
    }
};
```

## 💻 2. LinkedList Reversal

### Iterative Reversal
```cpp
ListNode* reverseListIterative(ListNode* head) {
    ListNode* prev = nullptr;
    ListNode* curr = head;
    
    while (curr) {
        ListNode* nextTemp = curr->next;
        curr->next = prev;
        prev = curr;
        curr = nextTemp;
    }
    
    return prev;  // New head
}

// Reverse first k nodes
ListNode* reverseKNodes(ListNode* head, int k) {
    ListNode* prev = nullptr;
    ListNode* curr = head;
    int count = 0;
    
    // Reverse first k nodes
    while (curr && count < k) {
        ListNode* nextTemp = curr->next;
        curr->next = prev;
        prev = curr;
        curr = nextTemp;
        count++;
    }
    
    // If there are more nodes, recursively reverse them
    if (curr) {
        head->next = reverseKNodes(curr, k);
    }
    
    return prev;
}
```

### Recursive Reversal
```cpp
ListNode* reverseListRecursive(ListNode* head) {
    // Base case
    if (!head || !head->next) {
        return head;
    }
    
    // Recursively reverse the rest
    ListNode* newHead = reverseListRecursive(head->next);
    
    // Reverse current connection
    head->next->next = head;
    head->next = nullptr;
    
    return newHead;
}

// Reverse between positions m and n
ListNode* reverseBetween(ListNode* head, int m, int n) {
    if (m == n) return head;
    
    ListNode* dummy = new ListNode(0);
    dummy->next = head;
    ListNode* prev = dummy;
    
    // Move to position m-1
    for (int i = 1; i < m; i++) {
        prev = prev->next;
    }
    
    ListNode* curr = prev->next;
    
    // Reverse n-m nodes
    for (int i = 0; i < n - m; i++) {
        ListNode* nextTemp = curr->next;
        curr->next = nextTemp->next;
        nextTemp->next = prev->next;
        prev->next = nextTemp;
    }
    
    return dummy->next;
}
```

## 💻 3. Cycle Detection (Floyd's Algorithm)

### Detect Cycle
```cpp
bool hasCycle(ListNode* head) {
    if (!head || !head->next) return false;
    
    ListNode* slow = head;
    ListNode* fast = head;
    
    while (fast && fast->next) {
        slow = slow->next;
        fast = fast->next->next;
        
        if (slow == fast) {
            return true;
        }
    }
    
    return false;
}

// Find start of cycle
ListNode* detectCycle(ListNode* head) {
    if (!head || !head->next) return nullptr;
    
    ListNode* slow = head;
    ListNode* fast = head;
    
    // Detect if cycle exists
    while (fast && fast->next) {
        slow = slow->next;
        fast = fast->next->next;
        
        if (slow == fast) {
            break;
        }
    }
    
    // No cycle found
    if (!fast || !fast->next) {
        return nullptr;
    }
    
    // Find start of cycle
    slow = head;
    while (slow != fast) {
        slow = slow->next;
        fast = fast->next;
    }
    
    return slow;
}

// Find length of cycle
int cycleLength(ListNode* head) {
    if (!hasCycle(head)) return 0;
    
    ListNode* slow = head;
    ListNode* fast = head;
    
    // Find meeting point
    do {
        slow = slow->next;
        fast = fast->next->next;
    } while (slow != fast);
    
    // Count nodes in cycle
    int length = 1;
    fast = fast->next;
    while (slow != fast) {
        fast = fast->next;
        length++;
    }
    
    return length;
}
```

## 💻 4. Finding Intersection Point

### Two Pointer Approach
```cpp
ListNode* getIntersectionNode(ListNode* headA, ListNode* headB) {
    if (!headA || !headB) return nullptr;
    
    ListNode* ptrA = headA;
    ListNode* ptrB = headB;
    
    // When one pointer reaches end, redirect to other list's head
    while (ptrA != ptrB) {
        ptrA = ptrA ? ptrA->next : headB;
        ptrB = ptrB ? ptrB->next : headA;
    }
    
    return ptrA;  // Either intersection point or nullptr
}

// Using length difference
ListNode* getIntersectionByLength(ListNode* headA, ListNode* headB) {
    int lenA = getLength(headA);
    int lenB = getLength(headB);
    
    // Align both lists to same starting position
    while (lenA > lenB) {
        headA = headA->next;
        lenA--;
    }
    
    while (lenB > lenA) {
        headB = headB->next;
        lenB--;
    }
    
    // Find intersection
    while (headA && headB) {
        if (headA == headB) {
            return headA;
        }
        headA = headA->next;
        headB = headB->next;
    }
    
    return nullptr;
}

int getLength(ListNode* head) {
    int length = 0;
    while (head) {
        length++;
        head = head->next;
    }
    return length;
}
```

## 💻 5. Clone LinkedList with Random Pointer

### Problem & Solution
```cpp
struct RandomListNode {
    int val;
    RandomListNode* next;
    RandomListNode* random;
    
    RandomListNode(int x) : val(x), next(nullptr), random(nullptr) {}
};

// Method 1: Using HashMap
RandomListNode* copyRandomList(RandomListNode* head) {
    if (!head) return nullptr;
    
    unordered_map<RandomListNode*, RandomListNode*> nodeMap;
    
    // First pass: create all nodes
    RandomListNode* curr = head;
    while (curr) {
        nodeMap[curr] = new RandomListNode(curr->val);
        curr = curr->next;
    }
    
    // Second pass: set next and random pointers
    curr = head;
    while (curr) {
        if (curr->next) {
            nodeMap[curr]->next = nodeMap[curr->next];
        }
        if (curr->random) {
            nodeMap[curr]->random = nodeMap[curr->random];
        }
        curr = curr->next;
    }
    
    return nodeMap[head];
}

// Method 2: O(1) space - Interweaving nodes
RandomListNode* copyRandomListOptimal(RandomListNode* head) {
    if (!head) return nullptr;
    
    // Step 1: Create copy nodes and interweave
    RandomListNode* curr = head;
    while (curr) {
        RandomListNode* copy = new RandomListNode(curr->val);
        copy->next = curr->next;
        curr->next = copy;
        curr = copy->next;
    }
    
    // Step 2: Set random pointers for copy nodes
    curr = head;
    while (curr) {
        if (curr->random) {
            curr->next->random = curr->random->next;
        }
        curr = curr->next->next;
    }
    
    // Step 3: Separate original and copy lists
    RandomListNode* dummy = new RandomListNode(0);
    RandomListNode* copyPrev = dummy;
    curr = head;
    
    while (curr) {
        RandomListNode* copy = curr->next;
        curr->next = copy->next;
        copyPrev->next = copy;
        copyPrev = copy;
        curr = curr->next;
    }
    
    return dummy->next;
}
```

## 💻 6. Merge Sort in LinkedList

### Implementation
```cpp
ListNode* mergeSort(ListNode* head) {
    if (!head || !head->next) {
        return head;
    }
    
    // Find middle and split
    ListNode* mid = findMiddle(head);
    ListNode* left = head;
    ListNode* right = mid->next;
    mid->next = nullptr;
    
    // Recursively sort both halves
    left = mergeSort(left);
    right = mergeSort(right);
    
    // Merge sorted halves
    return mergeTwoSortedLists(left, right);
}

ListNode* findMiddle(ListNode* head) {
    ListNode* slow = head;
    ListNode* fast = head;
    ListNode* prev = nullptr;
    
    while (fast && fast->next) {
        prev = slow;
        slow = slow->next;
        fast = fast->next->next;
    }
    
    return prev;  // Return node before middle for proper splitting
}

ListNode* mergeTwoSortedLists(ListNode* l1, ListNode* l2) {
    ListNode* dummy = new ListNode(0);
    ListNode* curr = dummy;
    
    while (l1 && l2) {
        if (l1->val <= l2->val) {
            curr->next = l1;
            l1 = l1->next;
        } else {
            curr->next = l2;
            l2 = l2->next;
        }
        curr = curr->next;
    }
    
    // Attach remaining nodes
    curr->next = l1 ? l1 : l2;
    
    return dummy->next;
}
```

## 💻 7. LRU Cache Implementation

### Using LinkedList + HashMap
```cpp
class LRUCache {
private:
    struct Node {
        int key, value;
        Node* prev;
        Node* next;
        Node(int k, int v) : key(k), value(v), prev(nullptr), next(nullptr) {}
    };
    
    int capacity;
    unordered_map<int, Node*> cache;
    Node* head;
    Node* tail;
    
    void addNode(Node* node) {
        // Add node right after head
        node->prev = head;
        node->next = head->next;
        head->next->prev = node;
        head->next = node;
    }
    
    void removeNode(Node* node) {
        // Remove node from list
        node->prev->next = node->next;
        node->next->prev = node->prev;
    }
    
    void moveToHead(Node* node) {
        // Move node to head (mark as recently used)
        removeNode(node);
        addNode(node);
    }
    
    Node* popTail() {
        // Remove last node (least recently used)
        Node* lastNode = tail->prev;
        removeNode(lastNode);
        return lastNode;
    }
    
public:
    LRUCache(int capacity) : capacity(capacity) {
        head = new Node(0, 0);
        tail = new Node(0, 0);
        head->next = tail;
        tail->prev = head;
    }
    
    int get(int key) {
        if (cache.find(key) != cache.end()) {
            Node* node = cache[key];
            moveToHead(node);
            return node->value;
        }
        return -1;
    }
    
    void put(int key, int value) {
        if (cache.find(key) != cache.end()) {
            // Update existing key
            Node* node = cache[key];
            node->value = value;
            moveToHead(node);
        } else {
            // Add new key
            Node* newNode = new Node(key, value);
            
            if (cache.size() >= capacity) {
                // Remove least recently used
                Node* tail = popTail();
                cache.erase(tail->key);
                delete tail;
            }
            
            addNode(newNode);
            cache[key] = newNode;
        }
    }
};
```

## 🎯 Complexity Analysis

| Operation | Time | Space | Notes |
|-----------|------|-------|-------|
| Insert at head | O(1) | O(1) | Most efficient insertion |
| Insert at tail | O(n) | O(1) | Need to traverse to end |
| Delete by value | O(n) | O(1) | Need to search first |
| Reverse | O(n) | O(1) | In-place operation |
| Cycle detection | O(n) | O(1) | Floyd's algorithm |
| Merge sort | O(n log n) | O(log n) | Recursive stack space |

## 📝 Interview Tips

1. **Always check for null pointers**
2. **Use dummy nodes to simplify edge cases**
3. **Draw diagrams for complex operations**
4. **Practice both iterative and recursive approaches**
5. **Understand when to use slow/fast pointers**

## 🎪 Common Interview Questions

**Q**: How do you detect if a LinkedList has a cycle?
**A**: Use Floyd's cycle detection (tortoise and hare) - slow pointer moves 1 step, fast pointer moves 2 steps.

**Q**: How do you find the middle of a LinkedList?
**A**: Use two pointers - slow moves 1 step, fast moves 2 steps. When fast reaches end, slow is at middle.

**Q**: How do you reverse a LinkedList?
**A**: Use three pointers (prev, curr, next) to reverse the links iteratively, or use recursion. 