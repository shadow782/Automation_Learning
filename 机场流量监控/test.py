import sys

def can_do(cur, nums, k):
    # 直接通过计算来判断能否成功
    total = 0
    for i in nums:
        total += min(cur, i)
    return total >= k

def main():
    # 这里可以通过输入方式获取值，但为了方便直接赋值
    # n = int(input())
    # k = int(input())
    # nums = list(map(int, input().split()))

    n = 3
    k = 6
    nums = [1, 2, 5]

    # 对数组进行排序
    nums.sort()

    # 将相邻元素的差值放入 nums 中
    for i in range(n - 1):
        nums[i] = nums[i + 1] - nums[i]
    nums[n - 1] = sys.maxsize

    # 二分查找的范围是 [1, k]
    left, right = 1, k
    ans = k

    while left <= right:
        mid = (left + right) // 2
        if can_do(mid, nums, k):
            ans = min(ans, mid)
            right = mid - 1
        else:
            left = mid + 1

    print(ans)

if __name__ == "__main__":
    main()
