# Copyright 2020 Samson Woof

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import requests
import os
import cv2
import numpy as np
from PIL import Image
from keras.models import load_model
import keras.utils as image
import matplotlib.pyplot as plt


def _download_file_from_google_drive(id, destination):
    """Source:
        https://stackoverflow.com/questions/25010369/wget-curl-large-file-from-google-drive).
    """
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith("download_warning"):
                return value

        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768

        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk: # filter out keep-alive new chunks
                    f.write(chunk)

    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = {"id" : id}, stream = True)
    token = get_confirm_token(response)

    if token:
        params = {"id" : id, "confirm" : token}
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)



def preprocess_image(img_path, target_size=(448, 448)):
    """Preprocess the image by reshape and normalization.

    Args:
        img_path:  A string.
        target_size: A tuple, reshape to this size.
    Return:
        An image ndarray.
    """
    img = image.load_img(img_path, target_size=target_size)
    img = image.img_to_array(img)
    img /= 255.0

    return img


def show_imgwithheat(img_path, save_path, heatmap, alpha=0.8, return_array=False):
    """Show the image with heatmap.

    Args:
        img_path: string.
        heatmap:  image array, get it by calling grad_cam().
        alpha:    float, transparency of heatmap.
        return_array: bool, return a superimposed image array or not.
    Return:
        None or image array.
    """
    plt.figure(figsize=(8, 4))

    img = cv2.imread(img_path)
    try:
        img.shape
    except:
        print("Cannot read this image!")

    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = (heatmap*255).astype("uint8")
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * alpha + img
    superimposed_img = np.clip(superimposed_img, 0, 255).astype("uint8")
    superimposed_img = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)
    plt.axis('off')
    plt.tight_layout()
    plt.imshow(superimposed_img, cmap=plt.cm.jet)
    plt.savefig(fname=save_path,
                bbox_inches="tight", pad_inches=0.01, orientation="landscape", dpi=500)

    plt.show()

    if return_array:
        return superimposed_img
