import numpy as np
from sklearn.cluster import KMeans
class anchor:
    def __init__(self, k=3):
        self.k = k
    
    def iou(self,box, clusters):
        """
        计算单个边界框与所有聚类中心的 IoU
        """
        x = np.minimum(clusters[:, 0], box[0])
        y = np.minimum(clusters[:, 1], box[1])
        intersection = x * y

        box_area = box[0] * box[1]
        cluster_area = clusters[:, 0] * clusters[:, 1]
        union = box_area + cluster_area - intersection

        return intersection / union

    def kmeans(self,boxes, k, dist=np.median):
        """
        使用 K-means 聚类算法确定锚框尺寸
        """
        rows = boxes.shape[0]
        
        # 随机选择初始聚类中心
        clusters = boxes[np.random.choice(rows, k, replace=False)]
        
        last_clusters = np.zeros((rows,))
        while True:
            distances = np.array([1 - iou(box, clusters) for box in boxes]) # 计算所有边界框与聚类中心的 IoU
            current_clusters = np.argmin(distances, axis=1)  # 为每个边界框分配最近的聚类中心   axis=1 表示按行取最小值的索引

            if (current_clusters == last_clusters).all():   # 聚类中心不再变化 当前聚类中心即为最终结果
                break
            
            for cluster in range(k):        # 更新聚类中心
                clusters[cluster] = dist(boxes[current_clusters == cluster], axis=0)    # 取当前聚类中心的所有边界框的宽度和高度的中位数作为新的聚类中心
            
            last_clusters = current_clusters    # 记录上一次的聚类中心
        
        return clusters
    
    def generate_anchors(self,boxes):
        kmeans = KMeans(n_clusters=self.k, random_state=9)
        kmeans.fit(boxes)
        return kmeans.cluster_centers_


k2anchor = anchor(k)
# 假设 boxes 包含所有真实边界框的宽度和高度
boxes = np.array([[w, h] for w, h in zip(widths, heights)])
anchors = k2anchor.generate_anchors(boxes, k=9)
print("Anchors (width, height):")
print(anchors)

# 示例数据：假设我们有一组真实边界框的宽度和高度
boxes = np.array([
    [12, 16], [19, 36], [40, 28], [36, 62],
    [60, 80], [81, 120], [135, 169], [344, 282]
])

# 确定锚框数量 k
k = 3

# 使用 K-means 聚类确定锚框尺寸

anchors = k2anchor.kmeans(boxes, k)

print("Anchors (width, height):")
print(anchors)
