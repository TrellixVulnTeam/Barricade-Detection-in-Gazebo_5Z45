# -Barricade-Detection-in-Gazebo-
* 自制gazebo内锥桶检测数据集，使用paddledetection开发套件，训练后导出在x86 linux中使用Paddle Inference2.0进行部署使用。
# 1.项目背景
![](https://ai-studio-static-online.cdn.bcebos.com/a8b715d97c7d4f5ea1aea317be7c8b591e267519df094bd698b0af6a72bd0fcc)
* gazebo作为一款功能强大且开源免费的仿真平台，对于疫情在家没有实物调试的大学生们来说，简直提供了完美的仿真环境去检验控制算法。刚好这段时间又开始研究智能车视觉导航，就针对其中锥桶检测整理出本项目，自制gazebo锥桶训练集，基于paddledetection完成训练最后导出模型，最终实现在gazebo仿真环境中的部署。帮助大家更好的在gazebo中检验各种算法
# 2.数据集制作
* 仿真环境的锥桶和真实世界锥桶还挺像的，但奈何没找到合适的以训练模型或者对应数据集，所以自己动手做了
* 图片是在gazebo环境中用手动方向键驱动仿真小车从各角度拍的锥桶视频，从视频中抽帧得到图片
* 标注采用的工具是开源的标注工具[lableimg](https://github.com/tzutalin/labelImg)，标记后自动生成xml文件，符合VOC数据集读取格式。
![](https://ai-studio-static-online.cdn.bcebos.com/5c0373db8b344b1b9f7a1fe468eb1e2fcf31290f71ea4ede934091a04ea419af)
* 因为考虑到任务比较简单，最终从视频流中筛选出520张数据数据集
# 3.模型训练与导出
## 3.1 划分训练集和测试集
这里按照9：1划分了训练集和测试集并生成了对应的label_list.txt，同时要注意生成txt文件要符合PaddleDetection读取格式：
Img_Path/Img_name Xml_Path/Xml_name
```
import os
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt



!mkdir barricade/ImageSets

train = open('barricade/ImageSets/train.txt', 'w')
test = open('barricade/ImageSets/test.txt', 'w')
label_list = open('barricade/label_list.txt', 'w')
xml_path = "./Annotations/"
img_path = "./JPEGImages/"
count = 0

for xml_name in os.listdir('barricade/Annotations'):
    data =img_path + xml_name[:-4] + ".jpg " + xml_path + xml_name + "\n"
    if(count%10==0):
        test.write(data)
    else:
        train.write(data)
    count += 1

label_list.write("barricade")

train.close()
test.close()
label_list.close()
```
## 3.2 配置训练参数
PaddleDetection开发套件最方便的点就是不需要自己搭建复杂的模型，不仅可以快速进行迁移学习同时对训练中的参数也可以直接在yml环境文件中进行修改，简单方便易操作
具体步骤总结为：
* 1. 点开PaddleDetection/configs目录，在其中挑选自己要用的模型
* 2. 点开挑选好的模型yml文件，进行配置的修改，其中主要注意三点：
* ---①数据集格式：PaddleDetecion支持读取的格式有VOC , COCO,wider_face和fruit四种数据集，对于初学者建议使用原本yml设置好的数据集格式使用
* ---②数据集读取目录：将TrainReader，Testreder中数据集对应的文件，dataset_dir为数据集根目录，anno_path为3.1步中我们生成的txt文件。
* ---③训练策略：主要改的几个参数是
| 名称 | 作用 | 
| -------- | -------- | 
| max_iters    | 训练iter次数     | 
| save_dir     | 保存模型参数的路径     | 
| snapshot_iter     | 经过多少iter评估一次     | 
| pretrain_weights     | 预训练参数从哪里读取     | 
| num_classes     | 检测类别数，如果是双阶段检测模型得是分类数+1     | 
| learning_rate     | 学习率以及下面的PiecewiseDecay的milestones和warmup轮次设置   | 
值得一提的是：
* 1.TestReader中也有anno_path，这个是设置推理时打出来的标签名的，是对应label_list.txt，是不是有同学在用PaddleDtection做infer的时候经常出现aeroplane这种莫名奇妙的标签，就是因为没有设置TestReader的anno_path导致的，这种情况中use_default_label也要都设成false哦
* 2.这里提到参数只是常用几个修改参数，更多yml参数作用看参考[官方解释](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.0-beta/docs/advanced_tutorials/config_doc/yolov3_mobilenet_v1.md)
* 因为本次检测任务目标种类单一、环境简单、特征明显，考虑到我们部署模型的时候还会运行其他ROS功能包，所以检测模型选择了轻量级的YOLOV3-MobilenetV1，对应的yml设置也附在了项目中
* 这里使用的是YOLOV系列，也可以使用PaddleDetection提供的脚本在自己的数据集上实现聚类得到最佳初始anchor大小，并在yml中进行修改
```
!python PaddleDetection-release-2.0-rc/tools/anchor_cluster.py -c PaddleDetection-release-2.0-rc/configs/yolov3_mobilenet_v1_voc.yml -n 9 -s 416 -i 1666
```
## 3.3 开始训练

```
!python PaddleDetection-release-2.0-rc/tools/train.py  -c PaddleDetection-release-2.0-rc/configs/yolov3_mobilenet_v1_voc.yml --eval --use_vdl=True --vdl_log_dir=vdl
```
## 3.4 导出模型
因为任务比较简单吼，训练几轮后指标都挺好的，测试集上mAP轻轻松松95+，我就直接导出了，PaddleDetection也提供导出工具
```
!python PaddleDetection-release-2.0-rc/tools/export_model.py -c PaddleDetection-release-2.0-rc/configs/yolov3_mobilenet_v1_voc.yml \
        --output_dir=inference_model \
        -o weights=output/yolov3_mobilenet_v1_voc/best_model
```
运行之后就会在设置的inference_model目录下生成我们的部署文件，一共有三个：model、params和infer_cfg.yml，有这三个文件我们就可以开始部署了。
# 4.模型部署
* 以上步骤咱们使用的是飞浆核心框架PaddlePaddle，这完成是模型的训练，大家知道训练过程中涉及反向传播是有很多参数的，而这些参数在实际使用正向推导时是没有用的，这也是3.4步导出模型的意义，把反向传播的一些参数全部去掉，只保留核心的推导参数，降低参数的，这样剩下的参数就全是对正向推导有用的了。
* 对于推理部署，飞浆提供了两种方法[PaddleLite](https://paddlelite.paddlepaddle.org.cn/introduction/tech_highlights.html)和[Paddle Inference](https://paddleinference.paddlepaddle.org.cn/product_introduction/inference_intro.html)。两种推理工具在我使用下来感觉：Paddle Lite主要针对移动端，而Paddle Inference是针对服务器端、云端和无法使用Paddle Lite的嵌入式设备，两者都提供高性能推理引擎，这次咱们部署环境是x86 Linux，所以就采用Paddle Inference实现
## 4.1 Paddle Inference 2.0
* Paddle Inference是飞浆原生推理库，原生就意味Paddle能实现的op，Paddle Inference不需要通过任何类型转换就可以实现，同时提供C、C++、Python的预测API，为了能让大家理解的更方便，这里就选择使用Python API进行推理，也可以直接在AI Studio上直接运行
## 4.2 Paddle Inference安装
* 咱们部署环境是X86的Linux可以直接参考[官网步骤](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/2.0/install/pip/windows-pip.html)安装Paddle调用Paddle Inference进行推理
* 本项目推理部署采用的Paddle Inference 2.0，请正确安装版本
## 4.3 Paddle Inference推理步骤
```
import cv2
import numpy as np
from paddle.inference import Config
from paddle.inference import create_predictor
```
### 4.3.1 配置Config
* PaddleInferen推理核心是创建推理引擎predictor，在创建前要配置Config
* Config类是所有属性配置，属性在Config实例化配置好后装载到predictor中，其方法全部是配置属性的,具体主要常用方法总结如下：
1. set_model("model","params")：读取模型和参数
2. enable_use_gpu(memory,gpu_id)：设置是否使用GPU，使用GPU分配的内存和使用的GPU的ID号
3. enable_tensorrt_engine()：开启tensorrt加速
4. enable_mkldnn()：开启MKLDNN
* 因为使用显卡跑虚拟机里的Gazebo会卡机，所以我的虚拟机环境是不加载显卡，对应推理时也是使用CPU推理，这也是为啥模型选MobilenetV1了哈哈哈
```
config = Config()
config.set_model("inference_model/yolov3_mobilenet_v1_voc/__model__","inference_model/yolov3_mobilenet_v1_voc/__params__")
config.disable_gpu()
config.enable_mkldnn()
```
### 4.3.2 创建Predictor
* 使用create_predictor(config)用于创建Predictor类
* Predictor类是推理引擎，其方法全是正向推理运行的
```
predictor = create_predictor(config)
```
### 4.3.3 配置推理引擎
* 创建推理引擎predictor之后我们要手动对其输入进行设置
```
im_shape = np.array([416, 416]).reshape((1, 2)).astype(np.int32)
img = [image,im_shape]
for i,name in enumerate(input_names):
    #定义输入的tensor
    input_tensor = predictor.get_input_handle(name)
    #确定输入tensor的大小
    input_tensor.reshape(img[i].shape)
    #对应的数据读进去
    input_tensor.copy_from_cpu(img[i].copy())

predictor.run()

output_names = predictor.get_output_names()
print(output_names)
results = []
for i, name in enumerate(output_names):
    output_tensor = predictor.get_output_handle(name)
    output_data = output_tensor.copy_to_cpu()
    print(output_data)
```
## 4.4 完整推理
* 完整程序也可在predict.py中查看
```
import cv2
import numpy as np
from paddle.inference import Config
from paddle.inference import create_predictor

def preprocess(img , Size):
    img = cv2.resize(img,(Size,Size),0,0)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    img = img / 255.0
    mean = np.array(mean)[np.newaxis, np.newaxis, :]
    std = np.array(std)[np.newaxis, np.newaxis, :]
    img -= mean
    img /= std
    img = img.astype("float32").transpose(2,0,1)
    img = img[np.newaxis,::]
    return img

def predicte(img,predictor):
    input_names = predictor.get_input_names()
    for i,name in enumerate(input_names):
        #定义输入的tensor
        input_tensor = predictor.get_input_handle(name)
        #确定输入tensor的大小
        input_tensor.reshape(img[i].shape)
        #对应的数据读进去
        input_tensor.copy_from_cpu(img[i].copy())
    #开始预测
    predictor.run()
    #开始看结果
    results =[]
    output_names = predictor.get_output_names()
    for i, name in enumerate(output_names):
        output_tensor = predictor.get_output_handle(name)
        output_data = output_tensor.copy_to_cpu()
        results.append(output_data)
    return results

if __name__ == '__main__':
    #读入的摄像头信息根据大家自己ROS结点自己读入咯，这里我就直接用摄像头读取替代了，大家到时候这里自己更换进行
    cap = cv2.VideoCapture(1)
    config = Config()
    config.set_model("inference_model/yolov3_mobilenet_v1_voc/__model__","inference_model/yolov3_mobilenet_v1_voc/__params__")
    config.disable_gpu()
    config.enable_mkldnn()
    predictor = create_predictor(config)
    im_size = 416
    im_shape = np.array([416, 416]).reshape((1, 2)).astype(np.int32)
    while(1):
        success, img = cap.read()
        if (success == False):
            break
        img = cv2.resize(img, (im_size,im_size),0, 0)
        data = preprocess(img, im_size)
        results = trash_detect(trash_detector, [data, im_shape])
        for res in results[0]:
            img = cv2.rectangle(img, (int(res[2]), int(res[3])), (int(res[4]), int(res[5])), (255, 0, 0), 2)
        cv2.imshow("img", img)
        cv2.waitKey(10)
    cap.release()

```
* 部署模型已加载到项目中
* 项目对应AIStudio链接：https://aistudio.baidu.com/aistudio/projectdetail/1608121
* 最佳训练模型下载链接：https://aistudio.baidu.com/aistudio/datasetdetail/73911
