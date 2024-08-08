###配置主机环境
- 使用pip安装yolov8官方工具包，要求Python>=3.8，我使用的工具包版本是8.0.200。我的python环境打包到requirements.txt了，可以直接安装：
`pip install ultralytics`
ultralytics详细教程参见：https://docs.ultralytics.com/quickstart/#install-ultralytics
- 从ultralytics官网下载预训练好的Pytroch Yolov8n模型
`wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt`

###移植应用程序到i.MX8MP
- 把预训练好的Pytorch Yolov8n转换并量化成NXP i.MX8MP/i.MX93支持的TFLite模型
`yolo export model=yolov8n.pt format=tflite int8=True imgsz=320`
- 把主机的[例程](https://github.com/ultralytics/ultralytics/blob/main/examples/YOLOv8-OpenCV-int8-tflite-Python/main.py "例程")，移植到i.MX8MP上，参考NXP例程：https://github.com/nxp-imx/tensorflow-imx/blob/lf-6.6.3_1.0.0/tensorflow/lite/examples/python/label_image.py 

###在NXP云实验室硬件平台运行AI应用程序
- 把转换好的模型，测试视频，模型分类标签打包上传到云实验室的i.MX8MP平台
- 在板子上运行应用程序
`root@imx8mpevk:~# cd example/`
`root@imx8mpevk:~/example# python3 yolov8_tflite.py --model yolov8n_full_integer_quant.tflite --img test_1.mp4 -e /usr/lib/libvx_delegate.so`