# ZN的学习笔记

## 最优化控制

### 动态规划问题
- 离散时间下的最优化问题
- 无人机的起飞问题  见 Src/optimal.py 
- 无人机的定高控制问题


## 边沿计算
- 在边缘设备上运行AI应用程序，可以减少数据传输，提高数据安全性，减少云端计算压力
- 要解决两个问题 模型量化和模型部署
- 模型量化：把Pytorch模型转换成TFLite模型，可以使用ultralytics工具包; 
- 内存不够，需要调整模型的结构，可以使用pytorch的torch.quantization工具包, 参见：https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html
- 或者通过深度可分离卷积，减少模型参数，提高模型运行速度
- 或者降低数据的精度，比如把float32转换成int8
- 模型部署：把模型部署到边缘设备，可以使用NXP提供的Tensorflow Lite框架


### 配置主机环境
- 使用pip安装yolov8官方工具包，要求Python>=3.8，我使用的工具包版本是8.0.200。我的python环境打包到requirements.txt了，可以直接安装：
`pip install ultralytics`
ultralytics详细教程参见：https://docs.ultralytics.com/quickstart/#install-ultralytics
- 从ultralytics官网下载预训练好的Pytroch Yolov8n模型
`wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt`

### 移植应用程序到i.MX8MP
- 把预训练好的Pytorch Yolov8n转换并量化成NXP i.MX8MP/i.MX93支持的TFLite模型
`yolo export model=yolov8n.pt format=tflite int8=True imgsz=320`
- 把主机的[例程](https://github.com/ultralytics/ultralytics/blob/main/examples/YOLOv8-OpenCV-int8-tflite-Python/main.py "例程")，移植到i.MX8MP上，参考NXP例程：https://github.com/nxp-imx/tensorflow-imx/blob/lf-6.6.3_1.0.0/tensorflow/lite/examples/python/label_image.py 

### 在NXP云实验室硬件平台运行AI应用程序
- 把转换好的模型，测试视频，模型分类标签打包上传到云实验室的i.MX8MP平台
- 在板子上运行应用程序
`root@imx8mpevk:~# cd example/`
`root@imx8mpevk:~/example# python3 yolov8_tflite.py --model yolov8n_full_integer_quant.tflite --img test_1.mp4 -e /usr/lib/libvx_delegate.so`




