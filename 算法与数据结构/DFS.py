from collections import Counter

# 输入
n = int(input())
edges  = int(input())
init = list(map(int,input().split()))

# 建图
graph = [[]*n for _ in range(n)]
for _ in range(edges):
    u, v = map(int, input().split())
    graph[u][v] = 1
    graph[v][u] = 1
    graph[u][u] = 1
    graph[v][v] = 1

graph =[
    [1, 1, 0, 0, 0],
    [1, 1, 1, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 1, 1, 1],
    [0, 0, 0, 1, 1]
]
def funct(graph,init,n):

    visited = [False] * n
    def dfs(x):
        visited[x] = True
        nonlocal node_id,size
        size += 1
        for node,conn in graph[x]:  
            if conn == 0:
                continue
            if node in init:
                if node_id != -2 and node in init:
                    node_id = i if node_id == -1 else -2
            elif not visited[node]:
                dfs(node)

    cnt = Counter()
    for i,v in enumerate(visited):
        if v and i in init:
            continue
        node_id = -1
        size = 0
        dfs(i)
        if node_id >= 0:    # node_id = -1 没遇到病毒 node_id = i 遇到一个病毒 node_id = -2 遇到多个病毒
            cnt[i] += size
    
    return min((-size,node_id) for node_id,size in cnt.items())[1] if cnt else min(init)

def minimumCost(n: int, edges: List[List[int]], query: List[List[int]]) -> List[int]:
    g = [[] for _ in range(n)]
    for u,v,w in edges:
        g[u].append((v,w))
        g[v].append((u,w))

    ids = [-1]*n
    res = []

    def dfs(x:int)->int:    # x未遍历过
        and_ = -1
        ids[x] = len(res)   # 标记自己所在的连通块
        for y,w in graph[x]:
            and_ &= w
            if ids[y] == -1:
                and_ &= dfs(y)
        return and_
    
    # 
    for i in range(n):
        if ids[i] == -1:
            res.append(dfs(i))
    
    ls = []
    for f,t in query:
        if ids[f] == ids[t]:
            ls.append(res[ids[f]])
        else:
            ls.append(-1)
    return ls  

def minimumCost(n: int, edges: List[List[int]], query: List[List[int]]) -> List[int]:
    # init ziji shi ziji de die
    fa = list(range(n))
    
    and_ = [-1]*n

    # find dad
    def find(x:int) -> int:
        if fa[x] != x:
            fa[x] = find(fa[x])
        return fa[x]
    
    for x,y,w in edges:
        #join  set y to be dad
        x = find(x)
        y = find(y)
        and_[y] &= w
        if x!=y:
            and_[y] &= and_[x]
            # join
            fa[x] = y
    ls = []
    for f,t in query:
        # check if union
        if find[f] != find[t]:
            ls.append(-1)
        else:
            ls.append(and_[find[f]])