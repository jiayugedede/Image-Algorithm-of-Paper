import time
import numpy as np
from PIL import Image
import PIL.Image as pilimg
import matplotlib.pyplot as plt
from tensorflow.keras import Model
from tensorflow.keras.models import load_model
from [Code.py] import [Component] # This line is just a demo to demonstrate the usage of this code.
from [Code.py] import [Component] # This line is just a demo to demonstrate the usage of this code.


Configure_Dictionary = {
    "save_path": r"[Save Path, for example: '/Storage/addition/Jet']",
    "model_name": r"[Weight path, for example: '/home/**.h5']",
    "layer_name": "[Inputing by yourself]",
    "color_mode": "Jet"}


model_config = V2_ResNet_ml1block_3_addition
save_path = model_config["save_path"]


def save_numpy2txt(array, index, path):
    full_path = path + "/" + str(index) + ".txt"
    np.savetxt(full_path, array)
    return 1


def load_trained_model():
    _custom_objects = {
        "[Component Type]": [Component],
        "[Component Type]": [Component],
        "[Component Type]": [Component],
        "[Component Type]": [Component]
    }

    model_name = model_config["model_name"]
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
    jpg = r'/home/**.jpg' # input image, please type it by yourself.
    
    img_np = np.array(img2numpy(jpg)) # loading image
    img_np = np.expand_dims(img_np, axis=0)/255.0 # image pro-processing
    numpy_tensor = get_tensor_from_model(model=test_model,
                                         layer_name=model_config["layer_name"],
                                         img_tensor=img_np).numpy()

    for i in range(256): # 256 refer to the number of feature channels.
        numpy_array = numpy_tensor[:, :, :, i].squeeze()
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
