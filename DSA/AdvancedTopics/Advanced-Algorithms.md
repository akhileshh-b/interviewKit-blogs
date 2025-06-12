# Advanced Topics

## 📚 Theory - Advanced Algorithms

**Advanced algorithms** represent sophisticated techniques that solve complex computational problems efficiently. These algorithms often combine multiple data structures and algorithmic paradigms to achieve optimal performance.

### Categories Covered:
1. **String Algorithms**: Pattern matching and text processing
2. **Number Theory**: Mathematical algorithms for competitive programming
3. **Advanced Data Structures**: Specialized structures for complex queries
4. **Geometric Algorithms**: Computational geometry problems
5. **Graph Advanced**: Specialized graph algorithms

### Importance:
- **Competitive Programming**: Essential for contests like ACM-ICPC, CodeChef
- **Real-world Applications**: Search engines, bioinformatics, cryptography
- **Interview Preparation**: Top tech companies often ask advanced algorithm questions
- **Research**: Foundation for algorithm research and optimization

## 💻 1. String Algorithms

### KMP (Knuth-Morris-Pratt) Algorithm

**Theory**: Efficient string matching algorithm that avoids redundant comparisons by using failure function (LPS - Longest Proper Prefix which is also Suffix).

**Time Complexity**: O(n + m) where n = text length, m = pattern length
**Space Complexity**: O(m)

**Real-world Applications**:
- **Text editors**: Find and replace functionality
- **Search engines**: Web page content search
- **Bioinformatics**: DNA sequence matching
- **Compilers**: Lexical analysis and token recognition

```cpp
// Build failure function (LPS array)
vector<int> buildLPS(string pattern) {
    int m = pattern.length();
    vector<int> lps(m, 0);
    int len = 0; // Length of previous longest prefix suffix
    int i = 1;
    
    while (i < m) {
        if (pattern[i] == pattern[len]) {
            len++;
            lps[i] = len;
            i++;
        } else {
            if (len != 0) {
                len = lps[len - 1]; // Fallback using LPS
            } else {
                lps[i] = 0;
                i++;
            }
        }
    }
    
    return lps;
}

// KMP Search Algorithm
vector<int> KMPSearch(string text, string pattern) {
    int n = text.length();
    int m = pattern.length();
    vector<int> matches;
    
    if (m == 0) return matches;
    
    vector<int> lps = buildLPS(pattern);
    
    int i = 0; // Index for text
    int j = 0; // Index for pattern
    
    while (i < n) {
        if (pattern[j] == text[i]) {
            i++;
            j++;
        }
        
        if (j == m) {
            matches.push_back(i - j); // Found match at index i-j
            j = lps[j - 1];
        } else if (i < n && pattern[j] != text[i]) {
            if (j != 0) {
                j = lps[j - 1];
            } else {
                i++;
            }
        }
    }
    
    return matches;
}

// KMP for counting occurrences
int countOccurrences(string text, string pattern) {
    vector<int> matches = KMPSearch(text, pattern);
    return matches.size();
}

// KMP for checking if pattern exists
bool patternExists(string text, string pattern) {
    vector<int> matches = KMPSearch(text, pattern);
    return !matches.empty();
}
```

### Z-Algorithm

**Theory**: Builds Z-array where Z[i] is the length of longest substring starting from i which is also a prefix of the string.

**Applications**: Pattern matching, string comparison, finding repetitions

```cpp
vector<int> zAlgorithm(string s) {
    int n = s.length();
    vector<int> z(n);
    int l = 0, r = 0;
    
    for (int i = 1; i < n; i++) {
        if (i <= r) {
            z[i] = min(r - i + 1, z[i - l]);
        }
        
        while (i + z[i] < n && s[z[i]] == s[i + z[i]]) {
            z[i]++;
        }
        
        if (i + z[i] - 1 > r) {
            l = i;
            r = i + z[i] - 1;
        }
    }
    
    return z;
}

// Pattern matching using Z-algorithm
vector<int> zPatternSearch(string text, string pattern) {
    string combined = pattern + "$" + text;
    vector<int> z = zAlgorithm(combined);
    vector<int> matches;
    
    int patternLen = pattern.length();
    
    for (int i = patternLen + 1; i < combined.length(); i++) {
        if (z[i] == patternLen) {
            matches.push_back(i - patternLen - 1);
        }
    }
    
    return matches;
}

// Find all periods of a string
vector<int> findPeriods(string s) {
    vector<int> z = zAlgorithm(s);
    vector<int> periods;
    int n = s.length();
    
    for (int i = 1; i < n; i++) {
        if (i + z[i] == n) {
            periods.push_back(i);
        }
    }
    
    return periods;
}
```

### Advanced String Problems

```cpp
// Longest Palindromic Substring using Manacher's Algorithm
string longestPalindrome(string s) {
    if (s.empty()) return "";
    
    // Transform string: "abc" -> "^#a#b#c#$"
    string transformed = "^#";
    for (char c : s) {
        transformed += c;
        transformed += "#";
    }
    transformed += "$";
    
    int n = transformed.length();
    vector<int> P(n, 0);
    int center = 0, right = 0;
    int maxLen = 0, centerIndex = 0;
    
    for (int i = 1; i < n - 1; i++) {
        if (i < right) {
            P[i] = min(right - i, P[2 * center - i]);
        }
        
        // Try to expand around i
        while (transformed[i + P[i] + 1] == transformed[i - P[i] - 1]) {
            P[i]++;
        }
        
        // If expanded past right, adjust center and right
        if (i + P[i] > right) {
            center = i;
            right = i + P[i];
        }
        
        // Update maximum palindrome
        if (P[i] > maxLen) {
            maxLen = P[i];
            centerIndex = i;
        }
    }
    
    int start = (centerIndex - maxLen) / 2;
    return s.substr(start, maxLen);
}

// Rolling Hash for string comparison
class RollingHash {
private:
    static const int MOD = 1e9 + 7;
    static const int BASE = 31;
    
public:
    static long long computeHash(string s) {
        long long hash = 0;
        long long pow_base = 1;
        
        for (char c : s) {
            hash = (hash + (c - 'a' + 1) * pow_base) % MOD;
            pow_base = (pow_base * BASE) % MOD;
        }
        
        return hash;
    }
    
    static vector<long long> computePrefixHashes(string s) {
        int n = s.length();
        vector<long long> hash(n + 1, 0);
        vector<long long> pow_base(n + 1, 1);
        
        for (int i = 0; i < n; i++) {
            hash[i + 1] = (hash[i] + (s[i] - 'a' + 1) * pow_base[i]) % MOD;
            pow_base[i + 1] = (pow_base[i] * BASE) % MOD;
        }
        
        return hash;
    }
    
    static long long getSubstringHash(vector<long long>& prefixHash, 
                                     vector<long long>& powers, int l, int r) {
        long long result = (prefixHash[r + 1] - prefixHash[l] + MOD) % MOD;
        return (result * modInverse(powers[l], MOD)) % MOD;
    }
    
private:
    static long long modInverse(long long a, long long m) {
        return power(a, m - 2, m);
    }
    
    static long long power(long long a, long long b, long long mod) {
        long long result = 1;
        while (b > 0) {
            if (b & 1) result = (result * a) % mod;
            a = (a * a) % mod;
            b >>= 1;
        }
        return result;
    }
};
```

## 💻 2. Number Theory

### GCD and LCM

**Theory**: Greatest Common Divisor (GCD) and Least Common Multiple (LCM) are fundamental in number theory.

```cpp
// Euclidean Algorithm for GCD
int gcd(int a, int b) {
    while (b != 0) {
        int temp = b;
        b = a % b;
        a = temp;
    }
    return a;
}

// Recursive GCD
int gcdRecursive(int a, int b) {
    if (b == 0) return a;
    return gcdRecursive(b, a % b);
}

// LCM using GCD
long long lcm(int a, int b) {
    return (1LL * a * b) / gcd(a, b);
}

// GCD of array
int gcdArray(vector<int>& arr) {
    int result = arr[0];
    for (int i = 1; i < arr.size(); i++) {
        result = gcd(result, arr[i]);
        if (result == 1) break; // Early termination
    }
    return result;
}

// LCM of array
long long lcmArray(vector<int>& arr) {
    long long result = arr[0];
    for (int i = 1; i < arr.size(); i++) {
        result = lcm(result, arr[i]);
    }
    return result;
}
```

### Extended Euclidean Algorithm

**Theory**: Finds integers x, y such that ax + by = gcd(a, b)

```cpp
// Extended Euclidean Algorithm
struct ExtendedGCD {
    int gcd, x, y;
};

ExtendedGCD extendedGCD(int a, int b) {
    if (b == 0) {
        return {a, 1, 0};
    }
    
    ExtendedGCD result = extendedGCD(b, a % b);
    int x = result.y;
    int y = result.x - (a / b) * result.y;
    
    return {result.gcd, x, y};
}

// Modular Inverse using Extended Euclidean
int modularInverse(int a, int m) {
    ExtendedGCD result = extendedGCD(a, m);
    if (result.gcd != 1) {
        return -1; // Inverse doesn't exist
    }
    return (result.x % m + m) % m;
}

// Solve linear Diophantine equation ax + by = c
pair<bool, pair<int, int>> solveDiophantine(int a, int b, int c) {
    ExtendedGCD result = extendedGCD(a, b);
    
    if (c % result.gcd != 0) {
        return {false, {0, 0}}; // No solution
    }
    
    int x = result.x * (c / result.gcd);
    int y = result.y * (c / result.gcd);
    
    return {true, {x, y}};
}
```

### Sieve of Eratosthenes

**Theory**: Efficient algorithm to find all prime numbers up to a given limit.

```cpp
// Basic Sieve of Eratosthenes
vector<bool> sieveOfEratosthenes(int n) {
    vector<bool> isPrime(n + 1, true);
    isPrime[0] = isPrime[1] = false;
    
    for (int i = 2; i * i <= n; i++) {
        if (isPrime[i]) {
            for (int j = i * i; j <= n; j += i) {
                isPrime[j] = false;
            }
        }
    }
    
    return isPrime;
}

// Get list of primes up to n
vector<int> getPrimes(int n) {
    vector<bool> isPrime = sieveOfEratosthenes(n);
    vector<int> primes;
    
    for (int i = 2; i <= n; i++) {
        if (isPrime[i]) {
            primes.push_back(i);
        }
    }
    
    return primes;
}

// Segmented Sieve for large ranges
vector<int> segmentedSieve(int low, int high) {
    int limit = sqrt(high) + 1;
    vector<int> primes = getPrimes(limit);
    
    vector<bool> isPrime(high - low + 1, true);
    
    for (int prime : primes) {
        int start = max(prime * prime, (low + prime - 1) / prime * prime);
        
        for (int j = start; j <= high; j += prime) {
            isPrime[j - low] = false;
        }
    }
    
    vector<int> result;
    for (int i = low; i <= high; i++) {
        if (isPrime[i - low] && i > 1) {
            result.push_back(i);
        }
    }
    
    return result;
}

// Count primes up to n (optimized)
int countPrimes(int n) {
    if (n <= 2) return 0;
    
    vector<bool> isPrime(n, true);
    isPrime[0] = isPrime[1] = false;
    
    for (int i = 2; i * i < n; i++) {
        if (isPrime[i]) {
            for (int j = i * i; j < n; j += i) {
                isPrime[j] = false;
            }
        }
    }
    
    int count = 0;
    for (int i = 2; i < n; i++) {
        if (isPrime[i]) count++;
    }
    
    return count;
}
```

### Modular Arithmetic

**Theory**: Arithmetic operations under modulo to handle large numbers.

```cpp
const int MOD = 1e9 + 7;

// Modular exponentiation
long long power(long long base, long long exp, long long mod) {
    long long result = 1;
    base %= mod;
    
    while (exp > 0) {
        if (exp & 1) {
            result = (result * base) % mod;
        }
        base = (base * base) % mod;
        exp >>= 1;
    }
    
    return result;
}

// Modular inverse using Fermat's little theorem (when mod is prime)
long long modInverse(long long a, long long mod) {
    return power(a, mod - 2, mod);
}

// Modular factorial
vector<long long> computeFactorials(int n, long long mod) {
    vector<long long> fact(n + 1);
    fact[0] = 1;
    
    for (int i = 1; i <= n; i++) {
        fact[i] = (fact[i - 1] * i) % mod;
    }
    
    return fact;
}

// Modular inverse factorials
vector<long long> computeInverseFactorials(vector<long long>& fact, long long mod) {
    int n = fact.size() - 1;
    vector<long long> invFact(n + 1);
    invFact[n] = modInverse(fact[n], mod);
    
    for (int i = n - 1; i >= 0; i--) {
        invFact[i] = (invFact[i + 1] * (i + 1)) % mod;
    }
    
    return invFact;
}

// Modular combination (nCr)
long long nCr(int n, int r, vector<long long>& fact, vector<long long>& invFact, long long mod) {
    if (r > n || r < 0) return 0;
    
    return (fact[n] * invFact[r] % mod) * invFact[n - r] % mod;
}

// Chinese Remainder Theorem
long long chineseRemainderTheorem(vector<int>& remainders, vector<int>& moduli) {
    int n = remainders.size();
    long long result = 0;
    long long prod = 1;
    
    for (int mod : moduli) {
        prod *= mod;
    }
    
    for (int i = 0; i < n; i++) {
        long long partialProd = prod / moduli[i];
        long long inv = modInverse(partialProd, moduli[i]);
        result = (result + remainders[i] * partialProd % prod * inv) % prod;
    }
    
    return (result + prod) % prod;
}
```

### Prime Factorization

```cpp
// Prime factorization of a number
vector<pair<int, int>> primeFactorization(int n) {
    vector<pair<int, int>> factors;
    
    for (int i = 2; i * i <= n; i++) {
        int count = 0;
        while (n % i == 0) {
            n /= i;
            count++;
        }
        if (count > 0) {
            factors.push_back({i, count});
        }
    }
    
    if (n > 1) {
        factors.push_back({n, 1});
    }
    
    return factors;
}

// Count divisors using prime factorization
int countDivisors(int n) {
    auto factors = primeFactorization(n);
    int count = 1;
    
    for (auto& factor : factors) {
        count *= (factor.second + 1);
    }
    
    return count;
}

// Sum of divisors
long long sumOfDivisors(int n) {
    auto factors = primeFactorization(n);
    long long sum = 1;
    
    for (auto& factor : factors) {
        long long p = factor.first;
        int exp = factor.second;
        
        long long termSum = (power(p, exp + 1, LLONG_MAX) - 1) / (p - 1);
        sum *= termSum;
    }
    
    return sum;
}

// Euler's totient function
int eulerTotient(int n) {
    auto factors = primeFactorization(n);
    int result = n;
    
    for (auto& factor : factors) {
        int p = factor.first;
        result -= result / p;
    }
    
    return result;
}
```

## 💻 3. Advanced Data Structures

### Heavy Light Decomposition

**Theory**: Decomposes tree into heavy and light edges to answer path queries efficiently.

```cpp
class HeavyLightDecomposition {
private:
    vector<vector<int>> adj;
    vector<int> parent, depth, heavy, head, pos;
    vector<int> values;
    int currentPos;
    
    // DFS to find heavy edges
    int dfs(int v) {
        int size = 1;
        int maxChildSize = 0;
        
        for (int u : adj[v]) {
            if (u != parent[v]) {
                parent[u] = v;
                depth[u] = depth[v] + 1;
                int childSize = dfs(u);
                
                if (childSize > maxChildSize) {
                    maxChildSize = childSize;
                    heavy[v] = u;
                }
                
                size += childSize;
            }
        }
        
        return size;
    }
    
    // Decompose tree into heavy paths
    void decompose(int v, int h) {
        head[v] = h;
        pos[v] = currentPos++;
        
        if (heavy[v] != -1) {
            decompose(heavy[v], h);
        }
        
        for (int u : adj[v]) {
            if (u != parent[v] && u != heavy[v]) {
                decompose(u, u);
            }
        }
    }
    
public:
    HeavyLightDecomposition(int n) : adj(n), parent(n), depth(n), heavy(n, -1), 
                                   head(n), pos(n), values(n), currentPos(0) {}
    
    void addEdge(int u, int v) {
        adj[u].push_back(v);
        adj[v].push_back(u);
    }
    
    void build(int root) {
        parent[root] = -1;
        depth[root] = 0;
        dfs(root);
        decompose(root, root);
    }
    
    // Query maximum value on path from u to v
    int queryPath(int u, int v) {
        int result = 0;
        
        while (head[u] != head[v]) {
            if (depth[head[u]] < depth[head[v]]) {
                swap(u, v);
            }
            
            // Query segment from pos[head[u]] to pos[u]
            result = max(result, querySegment(pos[head[u]], pos[u]));
            u = parent[head[u]];
        }
        
        if (depth[u] > depth[v]) {
            swap(u, v);
        }
        
        // Query segment from pos[u] to pos[v]
        result = max(result, querySegment(pos[u], pos[v]));
        return result;
    }
    
private:
    int querySegment(int l, int r) {
        // Implement segment tree query here
        int maxVal = 0;
        for (int i = l; i <= r; i++) {
            maxVal = max(maxVal, values[i]);
        }
        return maxVal;
    }
};
```

### Mo's Algorithm

**Theory**: Offline algorithm for answering range queries by reordering queries to minimize transitions.

```cpp
struct Query {
    int l, r, idx;
    int block;
    
    bool operator<(const Query& other) const {
        if (block != other.block) {
            return block < other.block;
        }
        return (block & 1) ? l < other.l : l > other.l;
    }
};

class MosAlgorithm {
private:
    vector<int> arr;
    vector<int> freq;
    int currentAnswer;
    int blockSize;
    
    void add(int idx) {
        int val = arr[idx];
        freq[val]++;
        if (freq[val] == 1) {
            currentAnswer++;
        }
    }
    
    void remove(int idx) {
        int val = arr[idx];
        freq[val]--;
        if (freq[val] == 0) {
            currentAnswer--;
        }
    }
    
public:
    MosAlgorithm(vector<int>& input) : arr(input) {
        blockSize = sqrt(arr.size());
        freq.resize(1000001, 0);
        currentAnswer = 0;
    }
    
    vector<int> processQueries(vector<pair<int, int>>& queries) {
        int q = queries.size();
        vector<Query> mosQueries(q);
        
        for (int i = 0; i < q; i++) {
            mosQueries[i] = {queries[i].first, queries[i].second, i, queries[i].first / blockSize};
        }
        
        sort(mosQueries.begin(), mosQueries.end());
        
        vector<int> answers(q);
        int currentL = 0, currentR = -1;
        
        for (Query& query : mosQueries) {
            // Extend right
            while (currentR < query.r) {
                currentR++;
                add(currentR);
            }
            
            // Shrink right
            while (currentR > query.r) {
                remove(currentR);
                currentR--;
            }
            
            // Extend left
            while (currentL > query.l) {
                currentL--;
                add(currentL);
            }
            
            // Shrink left
            while (currentL < query.l) {
                remove(currentL);
                currentL++;
            }
            
            answers[query.idx] = currentAnswer;
        }
        
        return answers;
    }
};
```

### Centroid Decomposition

**Theory**: Decomposes tree into smaller subtrees by repeatedly finding and removing centroids.

```cpp
class CentroidDecomposition {
private:
    vector<vector<int>> adj;
    vector<bool> removed;
    vector<int> subtreeSize;
    
    int getSubtreeSize(int v, int parent) {
        subtreeSize[v] = 1;
        for (int u : adj[v]) {
            if (u != parent && !removed[u]) {
                subtreeSize[v] += getSubtreeSize(u, v);
            }
        }
        return subtreeSize[v];
    }
    
    int getCentroid(int v, int parent, int treeSize) {
        for (int u : adj[v]) {
            if (u != parent && !removed[u] && subtreeSize[u] > treeSize / 2) {
                return getCentroid(u, v, treeSize);
            }
        }
        return v;
    }
    
    void decompose(int v, int parent) {
        int treeSize = getSubtreeSize(v, -1);
        int centroid = getCentroid(v, -1, treeSize);
        
        removed[centroid] = true;
        
        // Process centroid here
        processNode(centroid);
        
        for (int u : adj[centroid]) {
            if (!removed[u]) {
                decompose(u, centroid);
            }
        }
    }
    
    void processNode(int centroid) {
        // Implement processing logic for centroid
        // This could involve answering queries, updating distances, etc.
    }
    
public:
    CentroidDecomposition(int n) : adj(n), removed(n, false), subtreeSize(n) {}
    
    void addEdge(int u, int v) {
        adj[u].push_back(v);
        adj[v].push_back(u);
    }
    
    void build() {
        decompose(0, -1);
    }
};
```

## 💻 4. Geometric Algorithms

### Convex Hull

**Theory**: Find the smallest convex polygon that contains all given points.

```cpp
struct Point {
    long long x, y;
    
    Point() : x(0), y(0) {}
    Point(long long x, long long y) : x(x), y(y) {}
    
    Point operator-(const Point& other) const {
        return Point(x - other.x, y - other.y);
    }
    
    long long cross(const Point& other) const {
        return x * other.y - y * other.x;
    }
    
    bool operator<(const Point& other) const {
        return x < other.x || (x == other.x && y < other.y);
    }
};

// Cross product of vectors OA and OB
long long crossProduct(const Point& O, const Point& A, const Point& B) {
    return (A - O).cross(B - O);
}

// Andrew's monotone chain algorithm for convex hull
vector<Point> convexHull(vector<Point> points) {
    int n = points.size();
    if (n <= 1) return points;
    
    sort(points.begin(), points.end());
    
    // Build lower hull
    vector<Point> hull;
    for (int i = 0; i < n; i++) {
        while (hull.size() >= 2 && 
               crossProduct(hull[hull.size()-2], hull[hull.size()-1], points[i]) <= 0) {
            hull.pop_back();
        }
        hull.push_back(points[i]);
    }
    
    // Build upper hull
    int lower_size = hull.size();
    for (int i = n - 2; i >= 0; i--) {
        while (hull.size() > lower_size && 
               crossProduct(hull[hull.size()-2], hull[hull.size()-1], points[i]) <= 0) {
            hull.pop_back();
        }
        hull.push_back(points[i]);
    }
    
    if (hull.size() > 1) hull.pop_back(); // Remove duplicate point
    return hull;
}

// Check if point is inside convex polygon
bool pointInConvexPolygon(const vector<Point>& polygon, const Point& point) {
    int n = polygon.size();
    bool positive = false, negative = false;
    
    for (int i = 0; i < n; i++) {
        long long cross = crossProduct(polygon[i], polygon[(i + 1) % n], point);
        if (cross > 0) positive = true;
        if (cross < 0) negative = true;
        if (positive && negative) return false;
    }
    
    return true;
}
```

### Line Intersection

```cpp
struct Line {
    Point a, b;
    
    Line(Point a, Point b) : a(a), b(b) {}
};

// Check if two line segments intersect
bool segmentsIntersect(const Line& l1, const Line& l2) {
    long long d1 = crossProduct(l2.a, l2.b, l1.a);
    long long d2 = crossProduct(l2.a, l2.b, l1.b);
    long long d3 = crossProduct(l1.a, l1.b, l2.a);
    long long d4 = crossProduct(l1.a, l1.b, l2.b);
    
    if (((d1 > 0 && d2 < 0) || (d1 < 0 && d2 > 0)) &&
        ((d3 > 0 && d4 < 0) || (d3 < 0 && d4 > 0))) {
        return true;
    }
    
    // Check for collinear cases
    if (d1 == 0 && onSegment(l2.a, l1.a, l2.b)) return true;
    if (d2 == 0 && onSegment(l2.a, l1.b, l2.b)) return true;
    if (d3 == 0 && onSegment(l1.a, l2.a, l1.b)) return true;
    if (d4 == 0 && onSegment(l1.a, l2.b, l1.b)) return true;
    
    return false;
}

bool onSegment(const Point& p, const Point& q, const Point& r) {
    return q.x <= max(p.x, r.x) && q.x >= min(p.x, r.x) &&
           q.y <= max(p.y, r.y) && q.y >= min(p.y, r.y);
}

// Find intersection point of two lines (if exists)
pair<bool, Point> lineIntersection(const Line& l1, const Line& l2) {
    long long denom = crossProduct(Point(0, 0), l1.b - l1.a, l2.b - l2.a);
    
    if (denom == 0) {
        return {false, Point()}; // Lines are parallel
    }
    
    long long t = crossProduct(l2.a - l1.a, l2.b - l2.a, Point(0, 0));
    
    Point intersection = Point(
        l1.a.x + t * (l1.b.x - l1.a.x) / denom,
        l1.a.y + t * (l1.b.y - l1.a.y) / denom
    );
    
    return {true, intersection};
}
```

## 💻 5. Advanced Graph Algorithms

### Bridges and Articulation Points

```cpp
class BridgesAndArticulationPoints {
private:
    vector<vector<int>> adj;
    vector<bool> visited;
    vector<int> disc, low, parent;
    vector<bool> articulationPoint;
    vector<pair<int, int>> bridges;
    int timer;
    
    void bridgeUtil(int u) {
        visited[u] = true;
        disc[u] = low[u] = ++timer;
        int children = 0;
        
        for (int v : adj[u]) {
            if (!visited[v]) {
                children++;
                parent[v] = u;
                bridgeUtil(v);
                
                low[u] = min(low[u], low[v]);
                
                // Check for articulation point
                if (parent[u] == -1 && children > 1) {
                    articulationPoint[u] = true;
                }
                
                if (parent[u] != -1 && low[v] >= disc[u]) {
                    articulationPoint[u] = true;
                }
                
                // Check for bridge
                if (low[v] > disc[u]) {
                    bridges.push_back({u, v});
                }
                
            } else if (v != parent[u]) {
                low[u] = min(low[u], disc[v]);
            }
        }
    }
    
public:
    BridgesAndArticulationPoints(int n) : adj(n), visited(n), disc(n), 
                                         low(n), parent(n, -1), 
                                         articulationPoint(n, false), timer(0) {}
    
    void addEdge(int u, int v) {
        adj[u].push_back(v);
        adj[v].push_back(u);
    }
    
    void findBridgesAndArticulationPoints() {
        for (int i = 0; i < adj.size(); i++) {
            if (!visited[i]) {
                bridgeUtil(i);
            }
        }
    }
    
    vector<pair<int, int>> getBridges() { return bridges; }
    vector<bool> getArticulationPoints() { return articulationPoint; }
};
```

## 🎯 Complexity Analysis

| Algorithm | Time Complexity | Space Complexity | Use Case |
|-----------|-----------------|------------------|----------|
| KMP | O(n + m) | O(m) | Pattern matching |
| Z-Algorithm | O(n) | O(n) | String preprocessing |
| Manacher's | O(n) | O(n) | Longest palindrome |
| Sieve | O(n log log n) | O(n) | Prime generation |
| Heavy-Light | O(log²n) per query | O(n) | Tree path queries |
| Mo's Algorithm | O((n + q)√n) | O(n) | Offline range queries |
| Convex Hull | O(n log n) | O(n) | Computational geometry |

## 📝 Interview Tips

1. **Understand the theory first** - Know why these algorithms work
2. **Practice implementation** - These are complex to code correctly
3. **Know when to apply** - Recognize problem patterns
4. **Master time complexities** - Essential for optimization discussions
5. **Handle edge cases** - Advanced algorithms have many corner cases

## 🎪 Common Applications

**Q**: When would you use KMP over naive string matching?
**A**: When pattern is long or text is very large. KMP's O(n+m) is much better than naive O(nm).

**Q**: What's the advantage of Heavy-Light Decomposition?
**A**: Reduces tree path queries from O(n) to O(log²n), crucial for competitive programming.

**Q**: When is Mo's Algorithm useful?
**A**: For offline range queries where online segment trees are too complex or memory-intensive.

This comprehensive guide covers the most important advanced algorithms with detailed implementations and theory needed for placement preparation and competitive programming. 