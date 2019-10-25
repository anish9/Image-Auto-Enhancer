import tensorflow as tf
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img,img_to_array
import sys

file_name = sys.argv[1]

def tensor_depth_to_space(imag,block_size,names):
    x = tf.depth_to_space(imag,block_size,name=names)
    return x

def tf_subpixel_conv(tensor,block_size,filters):
    x = Conv2D(filters,3,strides=(1,1),padding="same")(tensor)
    x = Lambda(lambda x : tensor_depth_to_space(x,block_size,names="subpixel_conv"))(x)
    x  = PReLU(shared_axes=[1, 2])(x)
    return x

def Translate(img_n):
    CLIP_VALUE = 255
    NORMALIZE = 127.5
    config_mod = 2
    read = cv2.imread(img_n)
    height,width = read.shape[0],read.shape[1]
    if height % config_mod != 0:
        height = height-1
    if width  % config_mod != 0:
        width = width-1
    ima = load_img(img_n,grayscale=False,target_size=(height,width))
    image = ima/np.array(NORMALIZE)-1
    image = np.expand_dims(image,axis=0)
    prediction = Model.predict(image)
    prediction = prediction+1
    prediction = prediction*NORMALIZE
    out = np.abs(prediction).astype(np.uint16)[0]
    out = np.clip(out,0,CLIP_VALUE)
    out = cv2.cvtColor(out,cv2.COLOR_BGR2RGB)
    out = cv2.resize(out,(width,height))
    return out


Model = load_model("MODEL/v1.h5",custom_objects={"tensor_depth_to_space":tensor_depth_to_space,"tf_subpixel_conv":tf_subpixel_conv})
cv2.imwrite("OUTPUTS/out_"+str(np.random.randint(0,90,1)[0])+".jpg",Translate(file_name))