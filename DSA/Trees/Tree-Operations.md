# Tree Operations

## 📚 Theory - Trees

**Tree** is a hierarchical data structure consisting of nodes connected by edges. It's a non-linear data structure that simulates a hierarchical tree structure with a root value and subtrees of children.

### Core Characteristics:
- **Hierarchical**: Elements arranged in levels
- **Root**: Topmost node with no parent
- **Parent-Child**: Each node can have multiple children but only one parent
- **Leaves**: Nodes with no children
- **Acyclic**: No cycles allowed
- **Connected**: Path exists between any two nodes

### Tree Terminology:
- **Node**: Basic unit containing data
- **Root**: Top node of the tree
- **Parent**: Node with children
- **Child**: Node connected to parent
- **Leaf**: Node with no children
- **Subtree**: Tree formed by a node and its descendants
- **Height**: Longest path from node to leaf
- **Depth**: Distance from root to node
- **Level**: All nodes at same depth

### Real-World Applications:
- **File systems**: Directory structure
- **Organization charts**: Company hierarchy
- **HTML DOM**: Web page structure
- **Expression parsing**: Syntax trees
- **Database indexing**: B-trees, B+ trees
- **Decision making**: Decision trees
- **Game AI**: Minimax trees

### Types of Trees:
- **Binary Tree**: Each node has at most 2 children
- **Binary Search Tree**: Ordered binary tree
- **AVL Tree**: Self-balancing BST
- **Red-Black Tree**: Self-balancing BST
- **Heap**: Complete binary tree with heap property
- **Trie**: Prefix tree for strings
- **Segment Tree**: Range query tree

## 💻 1. Basic Tree Structure

### Binary Tree Node
```cpp
struct TreeNode {
    int val;
    TreeNode* left;
    TreeNode* right;
    
    TreeNode() : val(0), left(nullptr), right(nullptr) {}
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
    TreeNode(int x, TreeNode* l, TreeNode* r) : val(x), left(l), right(r) {}
};

// N-ary Tree Node
struct NaryTreeNode {
    int val;
    vector<NaryTreeNode*> children;
    
    NaryTreeNode(int x) : val(x) {}
};
```

### Basic Tree Operations
```cpp
class BinaryTree {
private:
    TreeNode* root;
    
public:
    BinaryTree() : root(nullptr) {}
    
    // Insert in level order
    void insert(int val) {
        if (!root) {
            root = new TreeNode(val);
            return;
        }
        
        queue<TreeNode*> q;
        q.push(root);
        
        while (!q.empty()) {
            TreeNode* node = q.front();
            q.pop();
            
            if (!node->left) {
                node->left = new TreeNode(val);
                return;
            } else if (!node->right) {
                node->right = new TreeNode(val);
                return;
            } else {
                q.push(node->left);
                q.push(node->right);
            }
        }
    }
    
    // Search for a value
    bool search(int val) {
        return searchHelper(root, val);
    }
    
private:
    bool searchHelper(TreeNode* node, int val) {
        if (!node) return false;
        if (node->val == val) return true;
        return searchHelper(node->left, val) || searchHelper(node->right, val);
    }
};
```

## 💻 2. Tree Traversals

### Theory
Tree traversal is the process of visiting each node exactly once. Different traversal methods serve different purposes:

- **Preorder (Root-Left-Right)**: Used for copying trees, prefix expressions
- **Inorder (Left-Root-Right)**: Gives sorted order in BST
- **Postorder (Left-Right-Root)**: Used for deleting trees, postfix expressions
- **Level Order**: Used for level-wise processing

### Recursive Traversals
```cpp
// Preorder Traversal (Root -> Left -> Right)
void preorderTraversal(TreeNode* root, vector<int>& result) {
    if (!root) return;
    
    result.push_back(root->val);        // Visit root
    preorderTraversal(root->left, result);   // Traverse left
    preorderTraversal(root->right, result);  // Traverse right
}

// Inorder Traversal (Left -> Root -> Right)
void inorderTraversal(TreeNode* root, vector<int>& result) {
    if (!root) return;
    
    inorderTraversal(root->left, result);    // Traverse left
    result.push_back(root->val);        // Visit root
    inorderTraversal(root->right, result);   // Traverse right
}

// Postorder Traversal (Left -> Right -> Root)
void postorderTraversal(TreeNode* root, vector<int>& result) {
    if (!root) return;
    
    postorderTraversal(root->left, result);  // Traverse left
    postorderTraversal(root->right, result); // Traverse right
    result.push_back(root->val);        // Visit root
}
```

### Iterative Traversals
```cpp
// Iterative Preorder
vector<int> preorderIterative(TreeNode* root) {
    vector<int> result;
    if (!root) return result;
    
    stack<TreeNode*> st;
    st.push(root);
    
    while (!st.empty()) {
        TreeNode* node = st.top();
        st.pop();
        
        result.push_back(node->val);
        
        // Push right first, then left (stack is LIFO)
        if (node->right) st.push(node->right);
        if (node->left) st.push(node->left);
    }
    
    return result;
}

// Iterative Inorder
vector<int> inorderIterative(TreeNode* root) {
    vector<int> result;
    stack<TreeNode*> st;
    TreeNode* curr = root;
    
    while (curr || !st.empty()) {
        // Go to leftmost node
        while (curr) {
            st.push(curr);
            curr = curr->left;
        }
        
        // Process current node
        curr = st.top();
        st.pop();
        result.push_back(curr->val);
        
        // Move to right subtree
        curr = curr->right;
    }
    
    return result;
}

// Iterative Postorder (Two stacks)
vector<int> postorderIterative(TreeNode* root) {
    vector<int> result;
    if (!root) return result;
    
    stack<TreeNode*> st1, st2;
    st1.push(root);
    
    while (!st1.empty()) {
        TreeNode* node = st1.top();
        st1.pop();
        st2.push(node);
        
        if (node->left) st1.push(node->left);
        if (node->right) st1.push(node->right);
    }
    
    while (!st2.empty()) {
        result.push_back(st2.top()->val);
        st2.pop();
    }
    
    return result;
}
```

### Level Order Traversal
```cpp
// Level Order (BFS)
vector<vector<int>> levelOrder(TreeNode* root) {
    vector<vector<int>> result;
    if (!root) return result;
    
    queue<TreeNode*> q;
    q.push(root);
    
    while (!q.empty()) {
        int levelSize = q.size();
        vector<int> currentLevel;
        
        for (int i = 0; i < levelSize; i++) {
            TreeNode* node = q.front();
            q.pop();
            
            currentLevel.push_back(node->val);
            
            if (node->left) q.push(node->left);
            if (node->right) q.push(node->right);
        }
        
        result.push_back(currentLevel);
    }
    
    return result;
}

// Zigzag Level Order
vector<vector<int>> zigzagLevelOrder(TreeNode* root) {
    vector<vector<int>> result;
    if (!root) return result;
    
    queue<TreeNode*> q;
    q.push(root);
    bool leftToRight = true;
    
    while (!q.empty()) {
        int levelSize = q.size();
        vector<int> currentLevel(levelSize);
        
        for (int i = 0; i < levelSize; i++) {
            TreeNode* node = q.front();
            q.pop();
            
            int index = leftToRight ? i : levelSize - 1 - i;
            currentLevel[index] = node->val;
            
            if (node->left) q.push(node->left);
            if (node->right) q.push(node->right);
        }
        
        leftToRight = !leftToRight;
        result.push_back(currentLevel);
    }
    
    return result;
}
```

## 💻 3. Diameter of Tree

### Theory
**Diameter** is the longest path between any two nodes in a tree. The path may or may not pass through the root.

**Key Insight**: For each node, the diameter passing through it is the sum of heights of left and right subtrees plus 1.

```cpp
class DiameterSolution {
private:
    int maxDiameter;
    
    int height(TreeNode* node) {
        if (!node) return 0;
        
        int leftHeight = height(node->left);
        int rightHeight = height(node->right);
        
        // Update diameter: path through current node
        maxDiameter = max(maxDiameter, leftHeight + rightHeight);
        
        // Return height of current node
        return 1 + max(leftHeight, rightHeight);
    }
    
public:
    int diameterOfBinaryTree(TreeNode* root) {
        maxDiameter = 0;
        height(root);
        return maxDiameter;
    }
};

// Alternative: Return both height and diameter
pair<int, int> diameterAndHeight(TreeNode* root) {
    if (!root) return {0, 0}; // {diameter, height}
    
    auto left = diameterAndHeight(root->left);
    auto right = diameterAndHeight(root->right);
    
    int diameter = max({left.first, right.first, left.second + right.second});
    int height = 1 + max(left.second, right.second);
    
    return {diameter, height};
}
```

## 💻 4. Lowest Common Ancestor (LCA)

### Theory
**LCA** is the lowest (i.e., deepest) node that has both given nodes as descendants.

### LCA in Binary Tree
```cpp
TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
    if (!root || root == p || root == q) {
        return root;
    }
    
    TreeNode* left = lowestCommonAncestor(root->left, p, q);
    TreeNode* right = lowestCommonAncestor(root->right, p, q);
    
    if (left && right) return root;  // p and q in different subtrees
    return left ? left : right;      // Both in same subtree
}

// LCA with path storage
bool findPath(TreeNode* root, TreeNode* target, vector<TreeNode*>& path) {
    if (!root) return false;
    
    path.push_back(root);
    
    if (root == target) return true;
    
    if (findPath(root->left, target, path) || 
        findPath(root->right, target, path)) {
        return true;
    }
    
    path.pop_back();
    return false;
}

TreeNode* lcaWithPath(TreeNode* root, TreeNode* p, TreeNode* q) {
    vector<TreeNode*> pathP, pathQ;
    
    if (!findPath(root, p, pathP) || !findPath(root, q, pathQ)) {
        return nullptr;
    }
    
    TreeNode* lca = nullptr;
    int minLen = min(pathP.size(), pathQ.size());
    
    for (int i = 0; i < minLen; i++) {
        if (pathP[i] == pathQ[i]) {
            lca = pathP[i];
        } else {
            break;
        }
    }
    
    return lca;
}
```

### LCA in Binary Search Tree
```cpp
TreeNode* lowestCommonAncestorBST(TreeNode* root, TreeNode* p, TreeNode* q) {
    if (!root) return nullptr;
    
    // Both nodes are in left subtree
    if (p->val < root->val && q->val < root->val) {
        return lowestCommonAncestorBST(root->left, p, q);
    }
    
    // Both nodes are in right subtree
    if (p->val > root->val && q->val > root->val) {
        return lowestCommonAncestorBST(root->right, p, q);
    }
    
    // One node is in left, other in right (or one is root)
    return root;
}

// Iterative version
TreeNode* lcaBSTIterative(TreeNode* root, TreeNode* p, TreeNode* q) {
    while (root) {
        if (p->val < root->val && q->val < root->val) {
            root = root->left;
        } else if (p->val > root->val && q->val > root->val) {
            root = root->right;
        } else {
            return root;
        }
    }
    return nullptr;
}
```

## 💻 5. Serialize and Deserialize

### Theory
**Serialization** converts tree to string format. **Deserialization** reconstructs tree from string. Useful for storing/transmitting tree data.

```cpp
class Codec {
public:
    // Serialize tree to string
    string serialize(TreeNode* root) {
        if (!root) return "null,";
        
        return to_string(root->val) + "," + 
               serialize(root->left) + 
               serialize(root->right);
    }
    
    // Deserialize string to tree
    TreeNode* deserialize(string data) {
        queue<string> nodes;
        string token;
        
        // Split string by comma
        for (char c : data) {
            if (c == ',') {
                nodes.push(token);
                token.clear();
            } else {
                token += c;
            }
        }
        
        return deserializeHelper(nodes);
    }
    
private:
    TreeNode* deserializeHelper(queue<string>& nodes) {
        if (nodes.empty()) return nullptr;
        
        string val = nodes.front();
        nodes.pop();
        
        if (val == "null") return nullptr;
        
        TreeNode* root = new TreeNode(stoi(val));
        root->left = deserializeHelper(nodes);
        root->right = deserializeHelper(nodes);
        
        return root;
    }
};

// Level order serialization
class CodecLevelOrder {
public:
    string serialize(TreeNode* root) {
        if (!root) return "";
        
        string result;
        queue<TreeNode*> q;
        q.push(root);
        
        while (!q.empty()) {
            TreeNode* node = q.front();
            q.pop();
            
            if (node) {
                result += to_string(node->val) + ",";
                q.push(node->left);
                q.push(node->right);
            } else {
                result += "null,";
            }
        }
        
        return result;
    }
    
    TreeNode* deserialize(string data) {
        if (data.empty()) return nullptr;
        
        vector<string> nodes;
        string token;
        
        for (char c : data) {
            if (c == ',') {
                nodes.push_back(token);
                token.clear();
            } else {
                token += c;
            }
        }
        
        TreeNode* root = new TreeNode(stoi(nodes[0]));
        queue<TreeNode*> q;
        q.push(root);
        
        int i = 1;
        while (!q.empty() && i < nodes.size()) {
            TreeNode* node = q.front();
            q.pop();
            
            if (nodes[i] != "null") {
                node->left = new TreeNode(stoi(nodes[i]));
                q.push(node->left);
            }
            i++;
            
            if (i < nodes.size() && nodes[i] != "null") {
                node->right = new TreeNode(stoi(nodes[i]));
                q.push(node->right);
            }
            i++;
        }
        
        return root;
    }
};
```

## 💻 6. Tree to Doubly Linked List

### Theory
Convert binary tree to doubly linked list where inorder traversal gives the sorted order. Left pointer becomes previous, right pointer becomes next.

```cpp
class TreeToDLL {
private:
    TreeNode* prev = nullptr;
    
public:
    TreeNode* treeToDoublyList(TreeNode* root) {
        if (!root) return nullptr;
        
        prev = nullptr;
        TreeNode* head = nullptr;
        
        // Convert to DLL
        convertToDLL(root, head);
        
        // Make it circular
        TreeNode* tail = head;
        while (tail->right) {
            tail = tail->right;
        }
        
        head->left = tail;
        tail->right = head;
        
        return head;
    }
    
private:
    void convertToDLL(TreeNode* root, TreeNode*& head) {
        if (!root) return;
        
        // Process left subtree
        convertToDLL(root->left, head);
        
        // Process current node
        if (!head) {
            head = root;  // First node in inorder
        } else {
            prev->right = root;
            root->left = prev;
        }
        prev = root;
        
        // Process right subtree
        convertToDLL(root->right, head);
    }
};

// Non-circular version
TreeNode* binaryTreeToDLL(TreeNode* root) {
    if (!root) return nullptr;
    
    TreeNode* prev = nullptr;
    TreeNode* head = nullptr;
    
    function<void(TreeNode*)> inorder = [&](TreeNode* node) {
        if (!node) return;
        
        inorder(node->left);
        
        if (!head) {
            head = node;
        } else {
            prev->right = node;
            node->left = prev;
        }
        prev = node;
        
        inorder(node->right);
    };
    
    inorder(root);
    return head;
}
```

## 💻 7. Segment Tree

### Theory
**Segment Tree** is used for range queries and updates. It's a binary tree where each node represents a segment/range of the array.

**Applications**: Range sum queries, range minimum queries, range updates

```cpp
class SegmentTree {
private:
    vector<int> tree;
    int n;
    
    void build(vector<int>& arr, int node, int start, int end) {
        if (start == end) {
            tree[node] = arr[start];
        } else {
            int mid = (start + end) / 2;
            build(arr, 2*node, start, mid);
            build(arr, 2*node+1, mid+1, end);
            tree[node] = tree[2*node] + tree[2*node+1];
        }
    }
    
    void updateHelper(int node, int start, int end, int idx, int val) {
        if (start == end) {
            tree[node] = val;
        } else {
            int mid = (start + end) / 2;
            if (idx <= mid) {
                updateHelper(2*node, start, mid, idx, val);
            } else {
                updateHelper(2*node+1, mid+1, end, idx, val);
            }
            tree[node] = tree[2*node] + tree[2*node+1];
        }
    }
    
    int queryHelper(int node, int start, int end, int l, int r) {
        if (r < start || end < l) {
            return 0; // Out of range
        }
        if (l <= start && end <= r) {
            return tree[node]; // Complete overlap
        }
        
        int mid = (start + end) / 2;
        int p1 = queryHelper(2*node, start, mid, l, r);
        int p2 = queryHelper(2*node+1, mid+1, end, l, r);
        return p1 + p2;
    }
    
public:
    SegmentTree(vector<int>& arr) {
        n = arr.size();
        tree.resize(4 * n);
        build(arr, 1, 0, n-1);
    }
    
    void update(int idx, int val) {
        updateHelper(1, 0, n-1, idx, val);
    }
    
    int query(int l, int r) {
        return queryHelper(1, 0, n-1, l, r);
    }
};
```

### Lazy Propagation
```cpp
class SegmentTreeLazy {
private:
    vector<long long> tree, lazy;
    int n;
    
    void updateRange(int node, int start, int end, int l, int r, int val) {
        // Apply pending update
        if (lazy[node] != 0) {
            tree[node] += (end - start + 1) * lazy[node];
            if (start != end) {
                lazy[2*node] += lazy[node];
                lazy[2*node+1] += lazy[node];
            }
            lazy[node] = 0;
        }
        
        // No overlap
        if (start > r || end < l) return;
        
        // Complete overlap
        if (start >= l && end <= r) {
            tree[node] += (end - start + 1) * val;
            if (start != end) {
                lazy[2*node] += val;
                lazy[2*node+1] += val;
            }
            return;
        }
        
        // Partial overlap
        int mid = (start + end) / 2;
        updateRange(2*node, start, mid, l, r, val);
        updateRange(2*node+1, mid+1, end, l, r, val);
        tree[node] = tree[2*node] + tree[2*node+1];
    }
    
    long long queryRange(int node, int start, int end, int l, int r) {
        if (start > r || end < l) return 0;
        
        // Apply pending update
        if (lazy[node] != 0) {
            tree[node] += (end - start + 1) * lazy[node];
            if (start != end) {
                lazy[2*node] += lazy[node];
                lazy[2*node+1] += lazy[node];
            }
            lazy[node] = 0;
        }
        
        if (start >= l && end <= r) {
            return tree[node];
        }
        
        int mid = (start + end) / 2;
        long long p1 = queryRange(2*node, start, mid, l, r);
        long long p2 = queryRange(2*node+1, mid+1, end, l, r);
        return p1 + p2;
    }
    
public:
    SegmentTreeLazy(int size) {
        n = size;
        tree.resize(4 * n, 0);
        lazy.resize(4 * n, 0);
    }
    
    void updateRange(int l, int r, int val) {
        updateRange(1, 0, n-1, l, r, val);
    }
    
    long long queryRange(int l, int r) {
        return queryRange(1, 0, n-1, l, r);
    }
};
```

## 💻 8. Binary Indexed Tree (Fenwick Tree)

### Theory
**BIT** provides efficient methods for calculation and manipulation of prefix sums in O(log n) time.

```cpp
class BinaryIndexedTree {
private:
    vector<int> bit;
    int n;
    
public:
    BinaryIndexedTree(int size) {
        n = size;
        bit.assign(n + 1, 0);
    }
    
    BinaryIndexedTree(vector<int>& arr) {
        n = arr.size();
        bit.assign(n + 1, 0);
        for (int i = 0; i < n; i++) {
            update(i, arr[i]);
        }
    }
    
    void update(int idx, int delta) {
        for (++idx; idx <= n; idx += idx & -idx) {
            bit[idx] += delta;
        }
    }
    
    int query(int idx) {
        int sum = 0;
        for (++idx; idx > 0; idx -= idx & -idx) {
            sum += bit[idx];
        }
        return sum;
    }
    
    int rangeQuery(int l, int r) {
        return query(r) - query(l - 1);
    }
};

// 2D Binary Indexed Tree
class BIT2D {
private:
    vector<vector<int>> bit;
    int n, m;
    
public:
    BIT2D(int rows, int cols) {
        n = rows;
        m = cols;
        bit.assign(n + 1, vector<int>(m + 1, 0));
    }
    
    void update(int x, int y, int delta) {
        for (int i = x + 1; i <= n; i += i & -i) {
            for (int j = y + 1; j <= m; j += j & -j) {
                bit[i][j] += delta;
            }
        }
    }
    
    int query(int x, int y) {
        int sum = 0;
        for (int i = x + 1; i > 0; i -= i & -i) {
            for (int j = y + 1; j > 0; j -= j & -j) {
                sum += bit[i][j];
            }
        }
        return sum;
    }
};
```

## 💻 9. Trie (Prefix Tree)

### Theory
**Trie** is a tree-like data structure for storing strings. Each node represents a character, and paths from root to leaves represent complete words.

**Applications**: Auto-complete, spell checkers, IP routing, pattern matching

```cpp
struct TrieNode {
    TrieNode* children[26];
    bool isEndOfWord;
    
    TrieNode() {
        for (int i = 0; i < 26; i++) {
            children[i] = nullptr;
        }
        isEndOfWord = false;
    }
};

class Trie {
private:
    TrieNode* root;
    
public:
    Trie() {
        root = new TrieNode();
    }
    
    void insert(string word) {
        TrieNode* curr = root;
        
        for (char c : word) {
            int index = c - 'a';
            if (!curr->children[index]) {
                curr->children[index] = new TrieNode();
            }
            curr = curr->children[index];
        }
        
        curr->isEndOfWord = true;
    }
    
    bool search(string word) {
        TrieNode* curr = root;
        
        for (char c : word) {
            int index = c - 'a';
            if (!curr->children[index]) {
                return false;
            }
            curr = curr->children[index];
        }
        
        return curr->isEndOfWord;
    }
    
    bool startsWith(string prefix) {
        TrieNode* curr = root;
        
        for (char c : prefix) {
            int index = c - 'a';
            if (!curr->children[index]) {
                return false;
            }
            curr = curr->children[index];
        }
        
        return true;
    }
    
    // Find all words with given prefix
    vector<string> wordsWithPrefix(string prefix) {
        vector<string> result;
        TrieNode* curr = root;
        
        // Navigate to prefix end
        for (char c : prefix) {
            int index = c - 'a';
            if (!curr->children[index]) {
                return result;
            }
            curr = curr->children[index];
        }
        
        // DFS to find all words
        dfs(curr, prefix, result);
        return result;
    }
    
private:
    void dfs(TrieNode* node, string current, vector<string>& result) {
        if (node->isEndOfWord) {
            result.push_back(current);
        }
        
        for (int i = 0; i < 26; i++) {
            if (node->children[i]) {
                dfs(node->children[i], current + char('a' + i), result);
            }
        }
    }
};

// Trie with count (for frequency)
class TrieWithCount {
    struct TrieNode {
        TrieNode* children[26];
        int count;
        bool isEnd;
        
        TrieNode() {
            for (int i = 0; i < 26; i++) children[i] = nullptr;
            count = 0;
            isEnd = false;
        }
    };
    
    TrieNode* root;
    
public:
    TrieWithCount() {
        root = new TrieNode();
    }
    
    void insert(string word) {
        TrieNode* curr = root;
        
        for (char c : word) {
            int idx = c - 'a';
            if (!curr->children[idx]) {
                curr->children[idx] = new TrieNode();
            }
            curr = curr->children[idx];
            curr->count++;
        }
        
        curr->isEnd = true;
    }
    
    int countWordsWithPrefix(string prefix) {
        TrieNode* curr = root;
        
        for (char c : prefix) {
            int idx = c - 'a';
            if (!curr->children[idx]) {
                return 0;
            }
            curr = curr->children[idx];
        }
        
        return curr->count;
    }
};
```

## 🎯 Complexity Analysis

| Operation | Binary Tree | BST (Balanced) | Segment Tree | BIT | Trie |
|-----------|-------------|----------------|--------------|-----|------|
| Search | O(n) | O(log n) | O(log n) | O(log n) | O(m) |
| Insert | O(1) | O(log n) | O(log n) | O(log n) | O(m) |
| Delete | O(n) | O(log n) | O(log n) | O(log n) | O(m) |
| Traversal | O(n) | O(n) | O(n) | O(n) | O(n) |

*m = length of string in Trie*

## 📝 Interview Tips

1. **Master traversals** - Know all four types by heart
2. **Understand tree properties** - Height, depth, balanced vs unbalanced
3. **Practice recursion** - Most tree problems use recursive solutions
4. **Know when to use each structure** - BST vs Heap vs Trie
5. **Handle edge cases** - Empty trees, single nodes, null pointers

## 🎪 Common Interview Questions

**Q**: How do you check if a binary tree is balanced?
**A**: For each node, check if height difference between left and right subtrees is at most 1.

**Q**: What's the difference between complete and full binary trees?
**A**: Complete: All levels filled except possibly last (filled left to right). Full: Every node has 0 or 2 children.

**Q**: How do you find the kth smallest element in BST?
**A**: Use inorder traversal and count nodes, or use augmented BST with subtree sizes. 