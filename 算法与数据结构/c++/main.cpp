#include <iostream>
#include "union_find.h"

int main() {
    UnionFind uf;
    uf.init(5);

    // Test case 1: x and y are already in the same set
    uf.join(1, 2);
    uf.join(3, 4);
    uf.join(2, 3);
    bool result1 = uf.join(1, 4);
    // Expected output: false
    std::cout << result1 << std::endl;

    // Test case 2: x and y are in different sets
    uf.init(5);
    uf.join(1, 2);
    uf.join(3, 4);
    bool result2 = uf.join(2, 4);
    // Expected output: true
    std::cout << result2 << std::endl;

    return 0;
}