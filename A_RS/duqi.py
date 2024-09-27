# import cv2
# import numpy as np

# # 读取图像并转换为灰度图
# img = cv2.imread('abd.jpg')  # 替换为你的图像路径
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 2]  # 取 HSV 图像的 V 通道

# # 转换为浮点型
# gray = np.float32(gray)

# # 执行 Harris 角点检测
# # blockSize: 角点检测时考虑的邻域大小
# # ksize: Sobel 算子窗口大小
# # k: Harris 角点检测方程中的自由参数，通常取值在 [0.04, 0.06] 之间
# dst = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)

# # 扩大角点的结果，便于显示
# dst = cv2.dilate(dst, None)

# # 将角点标记出来（threshold 是阈值，将高于一定阈值的地方标记为角点）
# img[dst > 0.01 * dst.max()] = [0, 0, 255]

# # 显示图像
# cv2.imshow('Harris Corners', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

import cv2
import numpy as np

# 读取图像
img = cv2.imread('abd.jpg')

# 将图像转换为 HSV 颜色空间
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# 定义皮肤颜色的 HSV 范围（根据肤色调整）
lower_skin = np.array([10, 40, 60], dtype=np.uint8)  # 偏黄的肤色
upper_skin = np.array([25, 255, 255], dtype=np.uint8)

# 通过颜色范围提取腹部区域
mask = cv2.inRange(hsv, lower_skin, upper_skin)

# 使用形态学操作清理噪声（可选）
kernel = np.ones((5, 5), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

# 截取腹部区域并显示
abdomen_region = cv2.bitwise_and(img, img, mask=mask)

# 将提取出的腹部区域转换为灰度图
gray = cv2.cvtColor(abdomen_region, cv2.COLOR_BGR2GRAY)

# # 使用 Harris 角点检测
# gray = np.float32(gray)
# dst = cv2.cornerHarris(gray, 2, 3, 0.04)

# # 扩大角点检测结果
# dst = cv2.dilate(dst, None)

# 使用 SIFT 检测特征点，限制特征点数量
sift = cv2.SIFT_create(nfeatures=10)  # 只检测 10 个最显著的特征点
keypoints, descriptors = sift.detectAndCompute(gray, None)

# 获取图像中心坐标
height, width = img.shape[:2]
center_x, center_y = width // 2, height // 2

# 找到最接近中心的特征点
min_dist = float('inf')
closest_keypoint = None

for kp in keypoints:
    # 计算特征点与中心的距离
    dist = np.sqrt((kp.pt[0] - center_x) ** 2 + (kp.pt[1] - center_y) ** 2)
    
    # 更新最小距离和最近的特征点
    if dist < min_dist:
        min_dist = dist
        closest_keypoint = kp

# 绘制最接近中心的特征点
if closest_keypoint:
    img_with_keypoints = cv2.drawKeypoints(img, [closest_keypoint], None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # 显示结果
    cv2.imshow('Closest Keypoint to Center', img_with_keypoints)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No keypoints detected")

