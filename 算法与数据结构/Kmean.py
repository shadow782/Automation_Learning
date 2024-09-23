import numpy as np

def initialize_centroids(X, k):
	"""随机选择 k 个数据点作为初始中心点"""
	indices = np.random.choice(X.shape[0], k, replace=False)	# 随机选择 k 个索引 choice(总数, 个数, replace=False)
	return X[indices]

def assign_clusters(X, centroids):
	"""将每个数据点分配到最近的中心点"""
	distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)	# linalg.norm() 计算范数
	return np.argmin(distances, axis=1)

def update_centroids(X, labels, k):
	"""重新计算每个簇的中心点"""
	centroids = np.zeros((k, X.shape[1]))
	for i in range(k):
		points = X[labels == i]
		if points.size:
			centroids[i] = points.mean(axis=0)
	return centroids

def kmeans(X, k, max_iters=100, tol=1e-4):
	"""K-means 算法"""
	centroids = initialize_centroids(X, k)
	for _ in range(max_iters):
		labels = assign_clusters(X, centroids)
		new_centroids = update_centroids(X, labels, k)
		if np.all(np.abs(new_centroids - centroids) < tol):
			break
		centroids = new_centroids
	return centroids, labels

# 示例数据
N = 100
m = 15
k = 6
X = np.random.rand(N, m)

# 运行 K-means 算法
centroids, labels = kmeans(X, k)

print("中心点:", centroids)
print("标签:", labels)

# 可视化
import matplotlib.pyplot as plt
import matplotlib.cm as cm 
# plt.scatter(X[:, 0], X[:, 1], c=labels)
# plt.scatter(centroids[:, 0], centroids[:, 1], c='r', marker='x', s=100)
# plt.show()

# 多维数据可视化
# from mpl_toolkits.mplot3d import Axes3D
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=labels)
# ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], c='r', marker='x', s=100)
# plt.show()

# 画出每个簇的中心点 的直方图
# 每个维度 做一列， 每个样本做一行

def plot_centroids_bar_chart(centroids):
	num_clusters, num_features = centroids.shape
	fig, axes = plt.subplots(num_clusters, 1, figsize=(8, num_clusters * 2))
	colors = cm.rainbow(np.linspace(0, 1, num_features))
	
	for i in range(num_clusters):
		axes[i].bar(range(num_features), centroids[i], alpha=0.7, color=colors, edgecolor='black')
		axes[i].set_title(f'Cluster {i+1}', fontsize=5)
		# axes[i].set_xlabel('Feature Index', fontsize=5)
		axes[i].set_ylabel('Value', fontsize=5)
		
		# 添加每个簇中心点的具体值
		for j in range(num_features):
			axes[i].text(j, centroids[i, j], f'{centroids[i, j]:.2f}', ha='center', va='bottom')

	plt.tight_layout()
	plt.show()

# 调用函数绘制条形图
plot_centroids_bar_chart(centroids)




