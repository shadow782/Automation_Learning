1. 准备数据集
创建训练数据集和标注文件，标注格式为YOLO格式。

2. 安装和配置YOLO-V4
克隆Darknet仓库并编译：

git clone https://github.com/AlexeyAB/darknet.git
cd darknet
make

3. 下载YOLO-V4预训练模型权重：
bash
wget https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4.weights

4. 训练YOLO-V4模型
配置YOLO-V4：

编辑cfg/yolov4.cfg文件，根据数据集和需求进行修改。
创建包含类名的文件data/obj.names，每行一个类名。
创建包含数据集路径的文件data/obj.data。
训练模型：

bash
./darknet detector train data/obj.data cfg/yolov4.cfg yolov4.conv.137