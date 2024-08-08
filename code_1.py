import sys
from collections import *
from heapq import *

# for line in sys.stdin:
#     a = line.split()
#     print(int(a[0]) + int(a[1]))
    
class solution:
    def __init__(self):
        pass

    def relocate(self,clients:list[str]):
        n = len(clients)
        h = []          # 最小堆
        for i in range(n):
            heappush(h, i + 1)
        d = Counter()   #用户是否有柜子
        p = Counter()   #用户当前编号
        ans = -1
        for u in clients:
            if d[u] == 0:
                ans = heappop(h)
                p[u] = ans
            else:
                heappush(h,p[u])
                del p[u]
            d[u] ^= 1
        print(ans)
        
    def dif(self,heartrate:list[int],activity:list[str]):
        n = len(heartrate)
        i = 0
        ans = 0
        while i < n:
            j = i 
            mx = heartrate[i]
            mn = heartrate[i]
            while j < n and activity[j] == activity[i]:
                mx = max(mx,  heartrate[j])
                mn = min(mn,  heartrate[j])
                j += 1
            ans = max(ans, mx - mn)
            i = j
        print(ans)

a = solution()
heartRate = [60, 100, 90, 80]
activityLevel = ["Low", "High", "High", "High"]
a.dif(heartRate,activityLevel)
