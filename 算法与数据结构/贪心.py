# 两个操作，合并两个数； 将一个数除以2

# 问最小几次操作可以将数列全转换为奇数

def e2o(num):
    cnt = 0
    while num % 2 == 0:
        num //= 2
        cnt += 1
    return cnt

def min_op(ls):
    n = len(ls)
    odd_cnt = sum([1 for x in ls if x % 2 == 1])
    if odd_cnt == 0:
        min_odd = min([e2o(x) for x in ls if x % 2 == 0])
        odd_cnt += 1
    res = (n - odd_cnt) + min_odd

    return res
print(min_op([2, 4, 6, 8, 10]))