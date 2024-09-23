#include<stdio.h>
#include<iostream>
#include<cstring>
#include<cmath>
#include<algorithm>
#include<string>

using namespace std;    // namespace 是用来解决命名冲突的，std是标准命名空间


#include <iostream>
using namespace std;

// 定义一个普通函数
void myFunction(int x) {
    cout << "Value: " << x << endl;
}

// 定义一个返回 int 指针的函数
int* getPointer(int& x) {
    return &x;
}

int main() {
    // 定义一个函数指针，指向返回类型为 void，参数为 int 的函数
    void (*funcPtr)(int);

    // 将函数的地址赋值给函数指针
    funcPtr = myFunction;

    // 通过函数指针调用函数
    funcPtr(10);


    int a = 10;

    // 调用指针函数，获取指向 a 的指针
    int* ptr = getPointer(a);

    // 输出指针指向的值
    cout << "Value: " << *ptr << endl;

    return 0;
}
