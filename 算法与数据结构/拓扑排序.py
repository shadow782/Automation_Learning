from collections import deque
# 读取输入
n = int(input())
edges = int(input())

# 构建图
graph = [[] for _ in range(n)]
for _ in range(edges):
    u,v = map(int,input().split())
    graph[u].append(v)

# 计算入度
in_deg = [0]*n

for nodes in graph:
    for no in nodes:
        in_deg[no] += 1

# 拓扑排序 
q = deque([i for i,x in enumerate(in_deg) if x == 0])
# 结果  
ans = []
while q:
    node = q.popleft()
    ans.append(node)
    for after in graph[node]:
        in_deg[after] -= 1
        if in_deg[after]==0:
            q.append(after)
print(ans)

        