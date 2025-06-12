# Graph Algorithms

## 📚 Theory - Graphs

**Graph** is a non-linear data structure consisting of vertices (nodes) connected by edges. It's one of the most important data structures for modeling relationships between entities.

### Core Characteristics:
- **Vertices (V)**: Nodes or points in the graph
- **Edges (E)**: Connections between vertices
- **Degree**: Number of edges connected to a vertex
- **Path**: Sequence of vertices connected by edges
- **Cycle**: Path that starts and ends at the same vertex
- **Connected**: Path exists between every pair of vertices

### Graph Types:
- **Directed vs Undirected**: Edges have direction or not
- **Weighted vs Unweighted**: Edges have weights/costs or not
- **Cyclic vs Acyclic**: Contains cycles or not
- **Connected vs Disconnected**: All vertices reachable or not
- **Dense vs Sparse**: Many edges vs few edges
- **Simple vs Multigraph**: At most one edge between vertices or multiple allowed

### Real-World Applications:
- **Social Networks**: Facebook friends, Twitter followers
- **Transportation**: Road networks, flight routes
- **Computer Networks**: Internet topology, LAN connections
- **Web Pages**: Hyperlink structure, PageRank
- **Biology**: Protein interactions, gene networks
- **Scheduling**: Task dependencies, project planning
- **Games**: State spaces, decision trees

### Graph Properties:
- **Complete Graph**: Every vertex connected to every other vertex
- **Tree**: Connected acyclic graph with V-1 edges
- **Forest**: Collection of trees
- **Bipartite**: Vertices can be divided into two disjoint sets
- **Planar**: Can be drawn without edge crossings

## 💻 1. Graph Representation

### Adjacency Matrix
```cpp
class GraphMatrix {
private:
    vector<vector<int>> adjMatrix;
    int V; // Number of vertices
    bool isDirected;
    
public:
    GraphMatrix(int vertices, bool directed = false) {
        V = vertices;
        isDirected = directed;
        adjMatrix.assign(V, vector<int>(V, 0));
    }
    
    void addEdge(int u, int v, int weight = 1) {
        adjMatrix[u][v] = weight;
        if (!isDirected) {
            adjMatrix[v][u] = weight;
        }
    }
    
    void removeEdge(int u, int v) {
        adjMatrix[u][v] = 0;
        if (!isDirected) {
            adjMatrix[v][u] = 0;
        }
    }
    
    bool hasEdge(int u, int v) {
        return adjMatrix[u][v] != 0;
    }
    
    vector<int> getNeighbors(int vertex) {
        vector<int> neighbors;
        for (int i = 0; i < V; i++) {
            if (adjMatrix[vertex][i] != 0) {
                neighbors.push_back(i);
            }
        }
        return neighbors;
    }
    
    void printGraph() {
        for (int i = 0; i < V; i++) {
            for (int j = 0; j < V; j++) {
                cout << adjMatrix[i][j] << " ";
            }
            cout << endl;
        }
    }
};
```

### Adjacency List
```cpp
class GraphList {
private:
    vector<vector<pair<int, int>>> adjList; // {neighbor, weight}
    int V;
    bool isDirected;
    
public:
    GraphList(int vertices, bool directed = false) {
        V = vertices;
        isDirected = directed;
        adjList.resize(V);
    }
    
    void addEdge(int u, int v, int weight = 1) {
        adjList[u].push_back({v, weight});
        if (!isDirected) {
            adjList[v].push_back({u, weight});
        }
    }
    
    void removeEdge(int u, int v) {
        adjList[u].erase(
            remove_if(adjList[u].begin(), adjList[u].end(),
                     [v](const pair<int, int>& p) { return p.first == v; }),
            adjList[u].end()
        );
        
        if (!isDirected) {
            adjList[v].erase(
                remove_if(adjList[v].begin(), adjList[v].end(),
                         [u](const pair<int, int>& p) { return p.first == u; }),
                adjList[v].end()
            );
        }
    }
    
    bool hasEdge(int u, int v) {
        for (auto& edge : adjList[u]) {
            if (edge.first == v) return true;
        }
        return false;
    }
    
    vector<pair<int, int>> getNeighbors(int vertex) {
        return adjList[vertex];
    }
    
    void printGraph() {
        for (int i = 0; i < V; i++) {
            cout << i << ": ";
            for (auto& edge : adjList[i]) {
                cout << "(" << edge.first << "," << edge.second << ") ";
            }
            cout << endl;
        }
    }
    
    int getVertexCount() { return V; }
    vector<vector<pair<int, int>>>& getAdjList() { return adjList; }
};
```

## 💻 2. Graph Traversal - BFS and DFS

### Breadth-First Search (BFS)
**Theory**: Explores graph level by level, visiting all neighbors before moving to next level. Uses queue (FIFO).

**Applications**: Shortest path in unweighted graphs, level-order traversal, connected components

```cpp
// BFS Traversal
vector<int> bfsTraversal(GraphList& graph, int start) {
    int V = graph.getVertexCount();
    vector<bool> visited(V, false);
    vector<int> result;
    queue<int> q;
    
    visited[start] = true;
    q.push(start);
    
    while (!q.empty()) {
        int vertex = q.front();
        q.pop();
        result.push_back(vertex);
        
        for (auto& neighbor : graph.getNeighbors(vertex)) {
            if (!visited[neighbor.first]) {
                visited[neighbor.first] = true;
                q.push(neighbor.first);
            }
        }
    }
    
    return result;
}

// BFS with distance tracking
vector<int> bfsDistance(GraphList& graph, int start) {
    int V = graph.getVertexCount();
    vector<int> distance(V, -1);
    queue<int> q;
    
    distance[start] = 0;
    q.push(start);
    
    while (!q.empty()) {
        int vertex = q.front();
        q.pop();
        
        for (auto& neighbor : graph.getNeighbors(vertex)) {
            if (distance[neighbor.first] == -1) {
                distance[neighbor.first] = distance[vertex] + 1;
                q.push(neighbor.first);
            }
        }
    }
    
    return distance;
}

// BFS for shortest path
vector<int> bfsShortestPath(GraphList& graph, int start, int end) {
    int V = graph.getVertexCount();
    vector<int> parent(V, -1);
    vector<bool> visited(V, false);
    queue<int> q;
    
    visited[start] = true;
    q.push(start);
    
    while (!q.empty()) {
        int vertex = q.front();
        q.pop();
        
        if (vertex == end) break;
        
        for (auto& neighbor : graph.getNeighbors(vertex)) {
            if (!visited[neighbor.first]) {
                visited[neighbor.first] = true;
                parent[neighbor.first] = vertex;
                q.push(neighbor.first);
            }
        }
    }
    
    // Reconstruct path
    vector<int> path;
    int curr = end;
    while (curr != -1) {
        path.push_back(curr);
        curr = parent[curr];
    }
    
    reverse(path.begin(), path.end());
    return (path[0] == start) ? path : vector<int>(); // Return empty if no path
}
```

### Depth-First Search (DFS)
**Theory**: Explores graph by going as deep as possible before backtracking. Uses stack (LIFO) or recursion.

**Applications**: Topological sorting, cycle detection, strongly connected components

```cpp
// DFS Recursive
void dfsRecursive(GraphList& graph, int vertex, vector<bool>& visited, vector<int>& result) {
    visited[vertex] = true;
    result.push_back(vertex);
    
    for (auto& neighbor : graph.getNeighbors(vertex)) {
        if (!visited[neighbor.first]) {
            dfsRecursive(graph, neighbor.first, visited, result);
        }
    }
}

vector<int> dfsTraversal(GraphList& graph, int start) {
    int V = graph.getVertexCount();
    vector<bool> visited(V, false);
    vector<int> result;
    
    dfsRecursive(graph, start, visited, result);
    return result;
}

// DFS Iterative
vector<int> dfsIterative(GraphList& graph, int start) {
    int V = graph.getVertexCount();
    vector<bool> visited(V, false);
    vector<int> result;
    stack<int> st;
    
    st.push(start);
    
    while (!st.empty()) {
        int vertex = st.top();
        st.pop();
        
        if (!visited[vertex]) {
            visited[vertex] = true;
            result.push_back(vertex);
            
            // Add neighbors in reverse order for same order as recursive
            auto neighbors = graph.getNeighbors(vertex);
            for (auto it = neighbors.rbegin(); it != neighbors.rend(); ++it) {
                if (!visited[it->first]) {
                    st.push(it->first);
                }
            }
        }
    }
    
    return result;
}

// DFS with timestamps (discovery and finish times)
void dfsWithTimestamps(GraphList& graph, int vertex, vector<bool>& visited,
                      vector<int>& discovery, vector<int>& finish, int& time) {
    visited[vertex] = true;
    discovery[vertex] = ++time;
    
    for (auto& neighbor : graph.getNeighbors(vertex)) {
        if (!visited[neighbor.first]) {
            dfsWithTimestamps(graph, neighbor.first, visited, discovery, finish, time);
        }
    }
    
    finish[vertex] = ++time;
}
```

## 💻 3. Cycle Detection

### Cycle Detection in Undirected Graph
```cpp
// Using DFS
bool hasCycleUndirectedDFS(GraphList& graph, int vertex, vector<bool>& visited, int parent) {
    visited[vertex] = true;
    
    for (auto& neighbor : graph.getNeighbors(vertex)) {
        int adj = neighbor.first;
        
        if (!visited[adj]) {
            if (hasCycleUndirectedDFS(graph, adj, visited, vertex)) {
                return true;
            }
        } else if (adj != parent) {
            return true; // Back edge found
        }
    }
    
    return false;
}

bool detectCycleUndirected(GraphList& graph) {
    int V = graph.getVertexCount();
    vector<bool> visited(V, false);
    
    for (int i = 0; i < V; i++) {
        if (!visited[i]) {
            if (hasCycleUndirectedDFS(graph, i, visited, -1)) {
                return true;
            }
        }
    }
    
    return false;
}

// Using Union-Find
class UnionFind {
private:
    vector<int> parent, rank;
    
public:
    UnionFind(int n) {
        parent.resize(n);
        rank.resize(n, 0);
        for (int i = 0; i < n; i++) {
            parent[i] = i;
        }
    }
    
    int find(int x) {
        if (parent[x] != x) {
            parent[x] = find(parent[x]); // Path compression
        }
        return parent[x];
    }
    
    bool unite(int x, int y) {
        int rootX = find(x);
        int rootY = find(y);
        
        if (rootX == rootY) return false; // Already in same set
        
        // Union by rank
        if (rank[rootX] < rank[rootY]) {
            parent[rootX] = rootY;
        } else if (rank[rootX] > rank[rootY]) {
            parent[rootY] = rootX;
        } else {
            parent[rootY] = rootX;
            rank[rootX]++;
        }
        
        return true;
    }
};

bool detectCycleUndirectedUF(vector<pair<int, int>>& edges, int V) {
    UnionFind uf(V);
    
    for (auto& edge : edges) {
        if (!uf.unite(edge.first, edge.second)) {
            return true; // Cycle detected
        }
    }
    
    return false;
}
```

### Cycle Detection in Directed Graph
```cpp
// Using DFS with colors (White, Gray, Black)
enum Color { WHITE, GRAY, BLACK };

bool hasCycleDirectedDFS(GraphList& graph, int vertex, vector<Color>& color) {
    color[vertex] = GRAY;
    
    for (auto& neighbor : graph.getNeighbors(vertex)) {
        int adj = neighbor.first;
        
        if (color[adj] == GRAY) {
            return true; // Back edge to ancestor
        }
        
        if (color[adj] == WHITE && hasCycleDirectedDFS(graph, adj, color)) {
            return true;
        }
    }
    
    color[vertex] = BLACK;
    return false;
}

bool detectCycleDirected(GraphList& graph) {
    int V = graph.getVertexCount();
    vector<Color> color(V, WHITE);
    
    for (int i = 0; i < V; i++) {
        if (color[i] == WHITE) {
            if (hasCycleDirectedDFS(graph, i, color)) {
                return true;
            }
        }
    }
    
    return false;
}

// Using Kahn's algorithm (Topological Sort based)
bool detectCycleDirectedKahn(GraphList& graph) {
    int V = graph.getVertexCount();
    vector<int> indegree(V, 0);
    
    // Calculate indegrees
    for (int i = 0; i < V; i++) {
        for (auto& neighbor : graph.getNeighbors(i)) {
            indegree[neighbor.first]++;
        }
    }
    
    queue<int> q;
    for (int i = 0; i < V; i++) {
        if (indegree[i] == 0) {
            q.push(i);
        }
    }
    
    int processedNodes = 0;
    
    while (!q.empty()) {
        int vertex = q.front();
        q.pop();
        processedNodes++;
        
        for (auto& neighbor : graph.getNeighbors(vertex)) {
            indegree[neighbor.first]--;
            if (indegree[neighbor.first] == 0) {
                q.push(neighbor.first);
            }
        }
    }
    
    return processedNodes != V; // Cycle exists if not all nodes processed
}
```

## 💻 4. Topological Sorting

### Theory
**Topological Sort** is linear ordering of vertices in directed acyclic graph (DAG) such that for every directed edge (u,v), vertex u comes before v.

**Applications**: Task scheduling, course prerequisites, build systems

```cpp
// DFS-based Topological Sort
void topologicalSortDFS(GraphList& graph, int vertex, vector<bool>& visited, stack<int>& topoStack) {
    visited[vertex] = true;
    
    for (auto& neighbor : graph.getNeighbors(vertex)) {
        if (!visited[neighbor.first]) {
            topologicalSortDFS(graph, neighbor.first, visited, topoStack);
        }
    }
    
    topoStack.push(vertex);
}

vector<int> topologicalSort(GraphList& graph) {
    int V = graph.getVertexCount();
    vector<bool> visited(V, false);
    stack<int> topoStack;
    
    for (int i = 0; i < V; i++) {
        if (!visited[i]) {
            topologicalSortDFS(graph, i, visited, topoStack);
        }
    }
    
    vector<int> result;
    while (!topoStack.empty()) {
        result.push_back(topoStack.top());
        topoStack.pop();
    }
    
    return result;
}

// Kahn's Algorithm (BFS-based)
vector<int> topologicalSortKahn(GraphList& graph) {
    int V = graph.getVertexCount();
    vector<int> indegree(V, 0);
    vector<int> result;
    
    // Calculate indegrees
    for (int i = 0; i < V; i++) {
        for (auto& neighbor : graph.getNeighbors(i)) {
            indegree[neighbor.first]++;
        }
    }
    
    queue<int> q;
    for (int i = 0; i < V; i++) {
        if (indegree[i] == 0) {
            q.push(i);
        }
    }
    
    while (!q.empty()) {
        int vertex = q.front();
        q.pop();
        result.push_back(vertex);
        
        for (auto& neighbor : graph.getNeighbors(vertex)) {
            indegree[neighbor.first]--;
            if (indegree[neighbor.first] == 0) {
                q.push(neighbor.first);
            }
        }
    }
    
    return result;
}
```

## 💻 5. Shortest Path Algorithms

### Dijkstra's Algorithm
**Theory**: Finds shortest path from source to all other vertices in weighted graph with non-negative weights.

```cpp
vector<int> dijkstra(GraphList& graph, int src) {
    int V = graph.getVertexCount();
    vector<int> dist(V, INT_MAX);
    priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> pq;
    
    dist[src] = 0;
    pq.push({0, src});
    
    while (!pq.empty()) {
        int u = pq.top().second;
        int d = pq.top().first;
        pq.pop();
        
        if (d > dist[u]) continue; // Skip outdated entries
        
        for (auto& neighbor : graph.getNeighbors(u)) {
            int v = neighbor.first;
            int weight = neighbor.second;
            
            if (dist[u] + weight < dist[v]) {
                dist[v] = dist[u] + weight;
                pq.push({dist[v], v});
            }
        }
    }
    
    return dist;
}

// Dijkstra with path reconstruction
pair<vector<int>, vector<int>> dijkstraWithPath(GraphList& graph, int src) {
    int V = graph.getVertexCount();
    vector<int> dist(V, INT_MAX);
    vector<int> parent(V, -1);
    priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> pq;
    
    dist[src] = 0;
    pq.push({0, src});
    
    while (!pq.empty()) {
        int u = pq.top().second;
        int d = pq.top().first;
        pq.pop();
        
        if (d > dist[u]) continue;
        
        for (auto& neighbor : graph.getNeighbors(u)) {
            int v = neighbor.first;
            int weight = neighbor.second;
            
            if (dist[u] + weight < dist[v]) {
                dist[v] = dist[u] + weight;
                parent[v] = u;
                pq.push({dist[v], v});
            }
        }
    }
    
    return {dist, parent};
}

// Get path from source to destination
vector<int> getPath(vector<int>& parent, int src, int dest) {
    vector<int> path;
    int curr = dest;
    
    while (curr != -1) {
        path.push_back(curr);
        curr = parent[curr];
    }
    
    reverse(path.begin(), path.end());
    return (path[0] == src) ? path : vector<int>();
}
```

### Bellman-Ford Algorithm
**Theory**: Finds shortest path from source to all vertices, can handle negative weights, detects negative cycles.

```cpp
pair<vector<int>, bool> bellmanFord(vector<vector<pair<int, int>>>& graph, int V, int src) {
    vector<int> dist(V, INT_MAX);
    dist[src] = 0;
    
    // Relax all edges V-1 times
    for (int i = 0; i < V - 1; i++) {
        for (int u = 0; u < V; u++) {
            if (dist[u] != INT_MAX) {
                for (auto& edge : graph[u]) {
                    int v = edge.first;
                    int weight = edge.second;
                    
                    if (dist[u] + weight < dist[v]) {
                        dist[v] = dist[u] + weight;
                    }
                }
            }
        }
    }
    
    // Check for negative cycles
    for (int u = 0; u < V; u++) {
        if (dist[u] != INT_MAX) {
            for (auto& edge : graph[u]) {
                int v = edge.first;
                int weight = edge.second;
                
                if (dist[u] + weight < dist[v]) {
                    return {dist, false}; // Negative cycle detected
                }
            }
        }
    }
    
    return {dist, true}; // No negative cycle
}

// Bellman-Ford with path reconstruction
struct BellmanFordResult {
    vector<int> dist;
    vector<int> parent;
    bool hasNegativeCycle;
};

BellmanFordResult bellmanFordWithPath(vector<vector<pair<int, int>>>& graph, int V, int src) {
    vector<int> dist(V, INT_MAX);
    vector<int> parent(V, -1);
    dist[src] = 0;
    
    for (int i = 0; i < V - 1; i++) {
        for (int u = 0; u < V; u++) {
            if (dist[u] != INT_MAX) {
                for (auto& edge : graph[u]) {
                    int v = edge.first;
                    int weight = edge.second;
                    
                    if (dist[u] + weight < dist[v]) {
                        dist[v] = dist[u] + weight;
                        parent[v] = u;
                    }
                }
            }
        }
    }
    
    // Check for negative cycles
    bool hasNegCycle = false;
    for (int u = 0; u < V; u++) {
        if (dist[u] != INT_MAX) {
            for (auto& edge : graph[u]) {
                int v = edge.first;
                int weight = edge.second;
                
                if (dist[u] + weight < dist[v]) {
                    hasNegCycle = true;
                    break;
                }
            }
        }
        if (hasNegCycle) break;
    }
    
    return {dist, parent, hasNegCycle};
}
```

### Floyd-Warshall Algorithm
**Theory**: Finds shortest paths between all pairs of vertices. Uses dynamic programming.

```cpp
vector<vector<int>> floydWarshall(vector<vector<int>>& graph) {
    int V = graph.size();
    vector<vector<int>> dist = graph;
    
    // Initialize distances
    for (int i = 0; i < V; i++) {
        for (int j = 0; j < V; j++) {
            if (i == j) {
                dist[i][j] = 0;
            } else if (graph[i][j] == 0) {
                dist[i][j] = INT_MAX;
            }
        }
    }
    
    // Floyd-Warshall algorithm
    for (int k = 0; k < V; k++) {
        for (int i = 0; i < V; i++) {
            for (int j = 0; j < V; j++) {
                if (dist[i][k] != INT_MAX && dist[k][j] != INT_MAX) {
                    dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j]);
                }
            }
        }
    }
    
    return dist;
}

// Floyd-Warshall with path reconstruction
pair<vector<vector<int>>, vector<vector<int>>> floydWarshallWithPath(vector<vector<int>>& graph) {
    int V = graph.size();
    vector<vector<int>> dist = graph;
    vector<vector<int>> next(V, vector<int>(V, -1));
    
    // Initialize
    for (int i = 0; i < V; i++) {
        for (int j = 0; j < V; j++) {
            if (i == j) {
                dist[i][j] = 0;
            } else if (graph[i][j] == 0) {
                dist[i][j] = INT_MAX;
            } else {
                next[i][j] = j;
            }
        }
    }
    
    for (int k = 0; k < V; k++) {
        for (int i = 0; i < V; i++) {
            for (int j = 0; j < V; j++) {
                if (dist[i][k] != INT_MAX && dist[k][j] != INT_MAX) {
                    if (dist[i][k] + dist[k][j] < dist[i][j]) {
                        dist[i][j] = dist[i][k] + dist[k][j];
                        next[i][j] = next[i][k];
                    }
                }
            }
        }
    }
    
    return {dist, next};
}

// Reconstruct path between two vertices
vector<int> reconstructPath(vector<vector<int>>& next, int start, int end) {
    if (next[start][end] == -1) return {}; // No path
    
    vector<int> path;
    int curr = start;
    
    while (curr != end) {
        path.push_back(curr);
        curr = next[curr][end];
    }
    path.push_back(end);
    
    return path;
}
```

## 💻 6. Minimum Spanning Tree (MST)

### Theory
**MST** is a subset of edges that connects all vertices with minimum total weight, forming a tree (no cycles).

### Kruskal's Algorithm
```cpp
struct Edge {
    int u, v, weight;
    
    bool operator<(const Edge& other) const {
        return weight < other.weight;
    }
};

class UnionFind {
public:
    vector<int> parent, rank;
    
    UnionFind(int n) {
        parent.resize(n);
        rank.resize(n, 0);
        for (int i = 0; i < n; i++) {
            parent[i] = i;
        }
    }
    
    int find(int x) {
        if (parent[x] != x) {
            parent[x] = find(parent[x]);
        }
        return parent[x];
    }
    
    bool unite(int x, int y) {
        int rootX = find(x);
        int rootY = find(y);
        
        if (rootX == rootY) return false;
        
        if (rank[rootX] < rank[rootY]) {
            parent[rootX] = rootY;
        } else if (rank[rootX] > rank[rootY]) {
            parent[rootY] = rootX;
        } else {
            parent[rootY] = rootX;
            rank[rootX]++;
        }
        
        return true;
    }
};

pair<int, vector<Edge>> kruskalMST(vector<Edge>& edges, int V) {
    sort(edges.begin(), edges.end());
    
    UnionFind uf(V);
    vector<Edge> mst;
    int totalWeight = 0;
    
    for (Edge& edge : edges) {
        if (uf.unite(edge.u, edge.v)) {
            mst.push_back(edge);
            totalWeight += edge.weight;
            
            if (mst.size() == V - 1) break;
        }
    }
    
    return {totalWeight, mst};
}
```

### Prim's Algorithm
```cpp
pair<int, vector<pair<int, int>>> primMST(GraphList& graph, int start = 0) {
    int V = graph.getVertexCount();
    vector<bool> inMST(V, false);
    vector<int> key(V, INT_MAX);
    vector<int> parent(V, -1);
    
    priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> pq;
    
    key[start] = 0;
    pq.push({0, start});
    
    int totalWeight = 0;
    vector<pair<int, int>> mstEdges;
    
    while (!pq.empty()) {
        int u = pq.top().second;
        pq.pop();
        
        if (inMST[u]) continue;
        
        inMST[u] = true;
        totalWeight += key[u];
        
        if (parent[u] != -1) {
            mstEdges.push_back({parent[u], u});
        }
        
        for (auto& neighbor : graph.getNeighbors(u)) {
            int v = neighbor.first;
            int weight = neighbor.second;
            
            if (!inMST[v] && weight < key[v]) {
                key[v] = weight;
                parent[v] = u;
                pq.push({key[v], v});
            }
        }
    }
    
    return {totalWeight, mstEdges};
}
```

## 💻 7. Strongly Connected Components (SCC)

### Kosaraju's Algorithm
```cpp
void dfsFirst(GraphList& graph, int vertex, vector<bool>& visited, stack<int>& finishStack) {
    visited[vertex] = true;
    
    for (auto& neighbor : graph.getNeighbors(vertex)) {
        if (!visited[neighbor.first]) {
            dfsFirst(graph, neighbor.first, visited, finishStack);
        }
    }
    
    finishStack.push(vertex);
}

void dfsSecond(GraphList& transpose, int vertex, vector<bool>& visited, vector<int>& component) {
    visited[vertex] = true;
    component.push_back(vertex);
    
    for (auto& neighbor : transpose.getNeighbors(vertex)) {
        if (!visited[neighbor.first]) {
            dfsSecond(transpose, neighbor.first, visited, component);
        }
    }
}

GraphList getTranspose(GraphList& graph) {
    int V = graph.getVertexCount();
    GraphList transpose(V, true); // Directed graph
    
    for (int i = 0; i < V; i++) {
        for (auto& neighbor : graph.getNeighbors(i)) {
            transpose.addEdge(neighbor.first, i, neighbor.second);
        }
    }
    
    return transpose;
}

vector<vector<int>> kosarajuSCC(GraphList& graph) {
    int V = graph.getVertexCount();
    vector<bool> visited(V, false);
    stack<int> finishStack;
    
    // Step 1: Fill vertices in stack according to their finishing times
    for (int i = 0; i < V; i++) {
        if (!visited[i]) {
            dfsFirst(graph, i, visited, finishStack);
        }
    }
    
    // Step 2: Create transpose graph
    GraphList transpose = getTranspose(graph);
    
    // Step 3: Process vertices in reverse finishing order
    fill(visited.begin(), visited.end(), false);
    vector<vector<int>> sccs;
    
    while (!finishStack.empty()) {
        int vertex = finishStack.top();
        finishStack.pop();
        
        if (!visited[vertex]) {
            vector<int> component;
            dfsSecond(transpose, vertex, visited, component);
            sccs.push_back(component);
        }
    }
    
    return sccs;
}
```

## 💻 8. Bipartite Graph Detection

### Theory
**Bipartite Graph** is a graph whose vertices can be divided into two disjoint sets such that no two vertices within the same set are adjacent.

```cpp
// Using BFS
bool isBipartiteBFS(GraphList& graph, int start, vector<int>& color) {
    queue<int> q;
    color[start] = 0;
    q.push(start);
    
    while (!q.empty()) {
        int vertex = q.front();
        q.pop();
        
        for (auto& neighbor : graph.getNeighbors(vertex)) {
            int adj = neighbor.first;
            
            if (color[adj] == -1) {
                color[adj] = 1 - color[vertex];
                q.push(adj);
            } else if (color[adj] == color[vertex]) {
                return false;
            }
        }
    }
    
    return true;
}

bool isBipartite(GraphList& graph) {
    int V = graph.getVertexCount();
    vector<int> color(V, -1);
    
    for (int i = 0; i < V; i++) {
        if (color[i] == -1) {
            if (!isBipartiteBFS(graph, i, color)) {
                return false;
            }
        }
    }
    
    return true;
}

// Using DFS
bool isBipartiteDFS(GraphList& graph, int vertex, vector<int>& color, int c) {
    color[vertex] = c;
    
    for (auto& neighbor : graph.getNeighbors(vertex)) {
        int adj = neighbor.first;
        
        if (color[adj] == -1) {
            if (!isBipartiteDFS(graph, adj, color, 1 - c)) {
                return false;
            }
        } else if (color[adj] == color[vertex]) {
            return false;
        }
    }
    
    return true;
}
```

## 💻 9. Shortest Path in Grid/Maze

### Theory
Grid traversal using BFS for shortest path, handling obstacles and multiple targets.

```cpp
// Basic grid BFS
int shortestPathGrid(vector<vector<int>>& grid, pair<int, int> start, pair<int, int> end) {
    int rows = grid.size();
    int cols = grid[0].size();
    
    if (grid[start.first][start.second] == 1 || grid[end.first][end.second] == 1) {
        return -1; // Start or end is blocked
    }
    
    vector<vector<bool>> visited(rows, vector<bool>(cols, false));
    queue<pair<pair<int, int>, int>> q; // {{row, col}, distance}
    
    int dx[] = {-1, 1, 0, 0};
    int dy[] = {0, 0, -1, 1};
    
    q.push({start, 0});
    visited[start.first][start.second] = true;
    
    while (!q.empty()) {
        auto current = q.front();
        q.pop();
        
        int x = current.first.first;
        int y = current.first.second;
        int dist = current.second;
        
        if (x == end.first && y == end.second) {
            return dist;
        }
        
        for (int i = 0; i < 4; i++) {
            int nx = x + dx[i];
            int ny = y + dy[i];
            
            if (nx >= 0 && nx < rows && ny >= 0 && ny < cols && 
                !visited[nx][ny] && grid[nx][ny] == 0) {
                visited[nx][ny] = true;
                q.push({{nx, ny}, dist + 1});
            }
        }
    }
    
    return -1; // No path found
}

// Multi-source BFS (multiple starting points)
int shortestPathMultiSource(vector<vector<int>>& grid, vector<pair<int, int>>& sources, pair<int, int> target) {
    int rows = grid.size();
    int cols = grid[0].size();
    
    vector<vector<bool>> visited(rows, vector<bool>(cols, false));
    queue<pair<pair<int, int>, int>> q;
    
    // Add all sources to queue
    for (auto& source : sources) {
        q.push({source, 0});
        visited[source.first][source.second] = true;
    }
    
    int dx[] = {-1, 1, 0, 0};
    int dy[] = {0, 0, -1, 1};
    
    while (!q.empty()) {
        auto current = q.front();
        q.pop();
        
        int x = current.first.first;
        int y = current.first.second;
        int dist = current.second;
        
        if (x == target.first && y == target.second) {
            return dist;
        }
        
        for (int i = 0; i < 4; i++) {
            int nx = x + dx[i];
            int ny = y + dy[i];
            
            if (nx >= 0 && nx < rows && ny >= 0 && ny < cols && 
                !visited[nx][ny] && grid[nx][ny] == 0) {
                visited[nx][ny] = true;
                q.push({{nx, ny}, dist + 1});
            }
        }
    }
    
    return -1;
}

// A* algorithm for optimized pathfinding
struct Node {
    int x, y, g, h, f;
    
    Node(int x, int y, int g, int h) : x(x), y(y), g(g), h(h), f(g + h) {}
    
    bool operator>(const Node& other) const {
        return f > other.f;
    }
};

int manhattanDistance(int x1, int y1, int x2, int y2) {
    return abs(x1 - x2) + abs(y1 - y2);
}

int aStarPathfinding(vector<vector<int>>& grid, pair<int, int> start, pair<int, int> end) {
    int rows = grid.size();
    int cols = grid[0].size();
    
    vector<vector<bool>> visited(rows, vector<bool>(cols, false));
    priority_queue<Node, vector<Node>, greater<Node>> pq;
    
    int dx[] = {-1, 1, 0, 0};
    int dy[] = {0, 0, -1, 1};
    
    int h = manhattanDistance(start.first, start.second, end.first, end.second);
    pq.push(Node(start.first, start.second, 0, h));
    
    while (!pq.empty()) {
        Node current = pq.top();
        pq.pop();
        
        if (visited[current.x][current.y]) continue;
        visited[current.x][current.y] = true;
        
        if (current.x == end.first && current.y == end.second) {
            return current.g;
        }
        
        for (int i = 0; i < 4; i++) {
            int nx = current.x + dx[i];
            int ny = current.y + dy[i];
            
            if (nx >= 0 && nx < rows && ny >= 0 && ny < cols && 
                !visited[nx][ny] && grid[nx][ny] == 0) {
                
                int g = current.g + 1;
                int h = manhattanDistance(nx, ny, end.first, end.second);
                pq.push(Node(nx, ny, g, h));
            }
        }
    }
    
    return -1;
}
```

## 🎯 Complexity Analysis

| Algorithm | Time Complexity | Space Complexity | Use Case |
|-----------|-----------------|------------------|----------|
| BFS | O(V + E) | O(V) | Shortest path (unweighted) |
| DFS | O(V + E) | O(V) | Connectivity, cycles |
| Dijkstra | O((V + E) log V) | O(V) | Shortest path (weighted) |
| Bellman-Ford | O(VE) | O(V) | Negative weights |
| Floyd-Warshall | O(V³) | O(V²) | All pairs shortest path |
| Kruskal's MST | O(E log E) | O(V) | Minimum spanning tree |
| Prim's MST | O((V + E) log V) | O(V) | Minimum spanning tree |
| Kosaraju SCC | O(V + E) | O(V) | Strongly connected components |

## 📝 Interview Tips

1. **Understand graph representation trade-offs** - Matrix vs List
2. **Master traversal algorithms** - BFS for shortest path, DFS for connectivity
3. **Know when to use each shortest path algorithm**
4. **Practice grid/maze problems** - Very common in interviews
5. **Understand cycle detection techniques** - Different for directed/undirected

## 🎪 Common Interview Questions

**Q**: When would you use BFS vs DFS?
**A**: BFS for shortest path in unweighted graphs, level-order processing. DFS for path existence, cycle detection, topological sort.

**Q**: How do you detect cycles in directed vs undirected graphs?
**A**: Undirected: Use DFS with parent tracking or Union-Find. Directed: Use DFS with colors (White-Gray-Black) or topological sort.

**Q**: What's the difference between Dijkstra's and Bellman-Ford?
**A**: Dijkstra is faster O((V+E)logV) but can't handle negative weights. Bellman-Ford is O(VE) but handles negative weights and detects negative cycles. 