import time
import numpy as np
from PIL import Image
# from ArsenicNet3_visual_configure import MA_multi_spectral_attention_layer_2_Jet
import PIL.Image as pilimg
import matplotlib.pyplot as plt
from tensorflow.keras import Model
from tensorflow.keras.models import load_model
from SE_IBN import SeBlock, IBN, AdaptSquarePlusV1
from fcaTFFunction import MultiSpectralAttentionLayer


V2_ResNet_ml1block_3_BN_4 = {
    "save_path": r"/Storage/神经网络权重/木薯叶疾病权重和训练结果/ResNet-101_stratified/特征可视化/ml1block_3_BN_4/Jet",
    "model_name": r"/Storage/神经网络权重/木薯叶疾病权重和训练结果/ResNet-101_stratified/model.71-1.0114-1.h5",
    "layer_name": "ml1block_3_BN_4",
    "color_mode": "Jet"}

V2_ResNet_ml1block_3_addition={
    "save_path": r"/Storage/神经网络权重/木薯叶疾病权重和训练结果/ResNet-101_stratified/特征可视化/ml1block_3_addition/Jet",
    "model_name": r"/Storage/神经网络权重/木薯叶疾病权重和训练结果/ResNet-101_stratified/model.71-1.0114-1.h5",
    "layer_name": "ml1block_3_addition",
    "color_mode": "Jet"}


model_config = V2_ResNet_ml1block_3_addition
save_path = model_config["save_path"]
# save_path = r"/Storage/神经网络权重/木薯叶疾病权重和训练结果/MAIANet-3权重/判断错误图片/ml1block_3_addition"


def save_numpy2txt(array, index, path):
    full_path = path + "/" + str(index) + ".txt"
    np.savetxt(full_path, array)
    return 1


def load_trained_model():
    _custom_objects = {
        "Custom>SeBlock": SeBlock,
        "MultiSpectralAttentionLayer": MultiSpectralAttentionLayer,
        "Custom>IBN": IBN,
        "Custom>AdaptSquarePlusV1": AdaptSquarePlusV1
    }

    model_name = model_config["model_name"]
    # model_name = r"/Storage/神经网络权重/Keras_weight/model.15.h5"
    function_model = load_model(model_name, _custom_objects)
    print('model load success.')
    return function_model


def img2numpy(jpg):
    img = pilimg.open(jpg)
    resize_img = img.resize((448, 448))
    return resize_img


def statisticCostTime(test_model, img_np):
    print(time.strftime('%c', time.localtime(time.time())))
    start = time.time()

    x = test_model.predict(img_np)
    predict = np.argmax(x[0])
    print(predict)
    print(time.strftime('%c', time.localtime(time.time())))

    t = time.time() - start
    print(t)
    print(t / 100)


def get_tensor_from_model(model, layer_name, img_tensor):
    conv_layer = model.get_layer(layer_name)
    conv_numpy = Model([model.inputs], [conv_layer.output, model.output])
    conv_output, predictions = conv_numpy(img_tensor)
    return conv_output


def save_feature_as_image(index, numpy_array, path):
    # image = Image.fromarray(numpy_array.astype('uint8'))
    image = Image.fromarray(numpy_array)
    image.save(path + "/" + str(i) + ".png")


if __name__ == '__main__':
    test_model = load_trained_model()
    jpg = r'/home/sam/文档/第二篇论文/实验数据/后期验证可视化内容/CBB_231.jpg' # input image
    
    img_np = np.array(img2numpy(jpg)) # loading image
    img_np = np.expand_dims(img_np, axis=0)/255.0 # image pro-processing
    numpy_tensor = get_tensor_from_model(model=test_model,
                                         layer_name=model_config["layer_name"],
                                         img_tensor=img_np).numpy()  # 112*112*256

    for i in range(256): # 256 refer to the number of feature channels.
        numpy_array = numpy_tensor[:, :, :, i].squeeze()
        # save_feature_as_image(index=i, numpy_array=numpy_array, path=save_path)
        # plt.imshow(numpy_array, cmap=plt.cm.jet) # 喷气式飞机尾焰的颜色。
        print("index:!", i)
        save_numpy2txt(array=numpy_array, index=i, path=save_path)
        plt.axis('off')
        if model_config["color_mode"] == "gray":
            plt.imshow(numpy_array, cmap="gray")
        elif model_config["color_mode"] == "Jet":
            plt.imshow(numpy_array, cmap=plt.cm.jet)  # Jet color map.
        plt.savefig(save_path + "/" + str(i) + ".png", bbox_inches='tight',
                    pad_inches=0.02, dpi=500)
        plt.show()
        print("\n")
