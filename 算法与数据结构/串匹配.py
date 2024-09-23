def next(pattern:str):
    nx = [0]*len(pattern)
    j = 0               # 前缀指针
    for i in range(1,len(pattern)):

        while j > 0 and pattern[j] != pattern[i]:
            j = nx[j-1]    
        if pattern[i]==pattern[j]:
            j += 1
        nx[i] = j
    return nx

def kmp(text,pattern):
    n = len(text)
    m = len(pattern)
    j = 0            # 模式指针
    nx = next(pattern)
    for i in range(n):
        while j>0 and text[i] != pattern[j]:
            j = nx[j-1]
        if pattern[j] == text[i]:
            j += 1
        if j==m:
            print(f"Found pattern at index {i-m+1}")
            j = nx[j-1]
    if j==0:
        print("No pattern found")


text = "ababababca"
pattern = "abababca"
kmp(text,pattern)