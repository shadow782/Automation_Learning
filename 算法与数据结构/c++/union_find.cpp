#include "union_find.h"

void UnionFind::init(int n) {
    for (int i = 0; i < n; i++) {
        pre[i] = i;
        rank[i] = 1;
    }
}

int UnionFind::find(int x) {
    if (pre[x] == x) return x;
    return pre[x] = find(pre[x]);
}

bool UnionFind::connected(int x, int y) {
    return find(x) == find(y);
}

bool UnionFind::join(int x, int y) {
    x = find(x);
    y = find(y);
    if (x == y) return false;
    if (rank[x] > rank[y]) pre[y] = x;
    else {
        if (rank[x] == rank[y]) rank[y]++;
        pre[x] = y;
    }
    return true;
}