#ifndef UNION_FIND_H
#define UNION_FIND_H

class UnionFind {
private:
    static const int N = 1005;
    int pre[N];
    int rank[N];

public:
    void init(int n);
    int find(int x);
    bool connected(int x, int y);
    bool join(int x, int y);
};

#endif // UNION_FIND_H