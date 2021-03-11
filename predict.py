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