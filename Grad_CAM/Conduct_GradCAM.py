import tensorflow as tf
import matplotlib.pyplot as plt
import os
from tensorflow.keras.models import load_model
from [Code.py] import [Component]
from fcaTFFunction import MultiSpectralAttentionLayer
from Grad_CAMutils import preprocess_image, show_imgwithheat
from gradcam import grad_cam, grad_cam_plus
plt.rcParams['figure.figsize'] = 8, 8


def load_trained_model():
    _custom_objects = {
        "[Component Type]": [Component],
        "[Component Type]": [Component],
        "[Component Type]": [Component],
        "[Component Type]": [Component]
    }

    model_name = r"/home/**.h5" # Setting it by yourself.
    function_model = load_model(model_name, _custom_objects)
    print('model load success.')
    return function_model


def get_heat_map(full_image_path, model):

    img = preprocess_image(full_image_path)

    heatmap, category_name = grad_cam(model, img,
                                      label_name=['CBB', 'CBSD', 'CGM', "CMD", "Healthy"],
                                      )
    return heatmap, category_name


model = load_trained_model()
model.summary()

image_folder = r"/home/**/input_image"
save_path = r"/home/**/Output_Grad_CAM"

name_list = os.listdir(image_folder)
for file_name in name_list:
    full_image_name = image_folder+"/"+ file_name
    heatmap, category_name = get_heat_map(full_image_name, model)
    save_name = save_path + "/" + category_name + "_" + file_name
    show_imgwithheat(full_image_name, save_name, heatmap)
