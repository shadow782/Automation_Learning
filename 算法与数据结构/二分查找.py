

def bs1(ls,target):
    left, right = 0, len(ls) - 1    # 左闭右闭区间

    while left <= right:            # left > right
        mid = (left + right) // 2
        if ls[mid] < target:
            left = mid + 1        # [mid+1,right]
        else:
            right = mid - 1       # [left,mid-1] 
    return left

def bs3(ls,target):
    left, right = 0, len(ls)        # 左闭右开区间
    while left < right:             # [left,right)
        mid = (left + right) // 2
        if ls[mid] < target:        # [mid,right)
            left = mid  + 1
        else:
            right = mid
    return left

def bs4(ls,target):
    left, right = -1, len(ls)        # 左开右开区间
    while left + 1 < right:          # (left,right)
        mid = (left + right) // 2
        if ls[mid] < target:   
            left = mid               # (mid,right)
        else:
            right = mid              # (left,mid)
    return left + 1

# 区间 0-n-1
ls = [1, 2, 3, 4, 5, 5, 5, 8, 9, 10]

def find_bound():
    target = 5
    start = bs1(ls,target)
    if start == len(ls) or ls[start] != target:
        print([-1,-1])
        return
    end = bs1(ls,target+1) - 1
    print([start,end])
    # print(f'找{target}',bs1(ls,target)-1)

    # print(f'找{target}',bs3(ls,target)-1)
    # print(f'找{target}',bs4(ls,target)-1)
find_bound()