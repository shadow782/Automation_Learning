from collections import Counter
def lus(s:str):
    windows = Counter()
    mx = 0
    l = 0
    al,ar=0,0
    for r in range(len(s)):
        t = s[r]
        windows[t] += 1     # 右侧口
        
        while windows[t]>1:
            m = s[l]        # 左端口
            windows[m] -= 1
            l += 1
        if ar - al < r - l:
            ar,al = r,l
        mx = max(r-l+1,mx)
    return mx, s[al:ar+1]
a,b = lus("ababcdagajfgafgajfga")
print(a,b)