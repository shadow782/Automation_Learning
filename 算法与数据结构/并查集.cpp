#include <iostream>
using namespace std;

#include <iostream>
using namespace std;

class UnionFind {

private:
    static const int N = 1005; // 指定并查集所能包含元素的个数（由题意决定）
    int pre[N];                // 存储每个结点的前驱结点
    int rank[N];               // 树的高度

public:
    // 初始化函数，对录入的 n个结点进行初始化
    void init(int n) {
        for (int i = 0; i < n; i++) {
            pre[i] = i;        // 每个结点的上级都是自己
            rank[i] = 1;       // 每个结点构成的树的高度为 1
        }
    }

    // 查找结点 x的根结点
    int find(int x) {
        if (pre[x] == x) return x; // 递归出口：x的上级为 x本身，则 x为根结点
        return pre[x] = find(pre[x]); // 完成路径压缩，将 x的上级直接变为根结点
    }

    // 判断两个结点是否连通
    bool connected(int x, int y) {
        return find(x) == find(y); // 判断两个结点的根结点（即代表元）是否相同
    }

    // 合并两个结点所属的集合
    bool join(int x, int y) {
        x = find(x); // 寻找 x的代表元
        y = find(y); // 寻找 y的代表元
        if (x == y) return false; // 如果 x和 y的代表元一致，说明他们共属同一集合，则不需要合并，返回 false，表示合并失败；否则，执行下面的逻辑
        if (rank[x] > rank[y]) pre[y] = x; // 如果 x的高度大于 y，则令 y的上级为 x
        else { // 否则
            if (rank[x] == rank[y]) rank[y]++; // 如果 x的高度和 y的高度相同，则令 y的高度加1
            pre[x] = y; // 让 x的上级为 y
        }
        return true; // 返回 true，表示合并成功
    }
};

int main() {
    UnionFind uf;
    uf.init(5);

    // Test case 1: x and y are already in the same set
    uf.join(1, 2);
    uf.join(3, 4);
    uf.join(2, 3);
    bool result1 = uf.join(1, 4);
    // Expected output: false
    cout << result1 << endl;

    // Test case 2: x and y are in different sets
    uf.init(5);
    uf.join(1, 2);
    uf.join(3, 4);
    bool result2 = uf.join(2, 4);
    // Expected output: true
    cout << result2 << endl;

    return 0;
}