'''
好好学习
天天向上
'''
import cv2
import numpy as np


# PID控制器类
class PIDController:
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.prev_error = 0
        self.integral = 0

    def compute(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        self.prev_error = error
        return self.Kp * error + self.Ki * self.integral + self.Kd * derivative


# 车道检测函数
def detect_lane(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=50, maxLineGap=10)
    return lines


# 初始化PID控制器
pid = PIDController(Kp=0.1, Ki=0.01, Kd=0.1)

# 视频流
cap = cv2.VideoCapture('path_to_video.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    lines = detect_lane(frame)
    if lines is not None:
        left_lane = []
        right_lane = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            slope = (y2 - y1) / (x2 - x1)
            if slope < -0.5:  # Left lane
                left_lane.append(line)
            elif slope > 0.5:  # Right lane
                right_lane.append(line)

        # 计算车道中心
        if left_lane and right_lane:
            left_lane = np.mean(left_lane, axis=0)[0]
            right_lane = np.mean(right_lane, axis=0)[0]
            left_x1, left_y1, left_x2, left_y2 = left_lane
            right_x1, right_y1, right_x2, right_y2 = right_lane
            lane_center = ((left_x1 + left_x2) / 2 + (right_x1 + right_x2) / 2) / 2
            frame_center = frame.shape[1] / 2
            error = frame_center - lane_center

            # 计算方向盘转角
            dt = 1 / 30  # 假设视频帧率为30fps
            steering_angle = pid.compute(error, dt)
            print(f"Steering Angle: {steering_angle}")

            # 可视化
            cv2.line(frame, (int(left_x1), int(left_y1)), (int(left_x2), int(left_y2)), (255, 0, 0), 5)
            cv2.line(frame, (int(right_x1), int(right_y1)), (int(right_x2), int(right_y2)), (0, 0, 255), 5)
            cv2.line(frame, (int(frame_center), frame.shape[0]), (int(lane_center), frame.shape[0] // 2), (0, 255, 0),
                     5)

    cv2.imshow('Lane Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
