import tensorflow as tf
import cv2
from model import Model
from preprocess import *


SEED = 24

np.random.seed(SEED)
tf.random.set_seed(SEED)

"""config=tf.ConfigProto(log_device_placement=True)"""
with tf.compat.v1.Session() as sess:

    UNET = Model(sess, SEED)

    UNET.compile()

    # Loading model.
    print('Loading model...')
    load_path = 'checkpoints/2021-05-17_11_22_53/'
    UNET.load(load_path)

    videoinpath = 'v1.avi'
    videooutpath = 'v1_out.avi'
    capture = cv2.VideoCapture(videoinpath)
    height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter(videooutpath, fourcc, 20.0, (640, 360), True)
    print("Video processing")
    if capture.isOpened():
        i = 0
        while True:
            ret, img_src = capture.read()
            if not ret: break
            lab = color.rgb2lab(img_src)
            # lab_scaled = (lab + [-50., 0.5, 0.5]) / [50., 127.5, 127.5]
            img = lab[:, :, 0:1]
            size = (int(width * 0.05), int(height * 0.088889))
            shrink = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
            shrink = shrink.reshape(1, 32, 32, 1)
            pred = UNET.sample(shrink)
            img_out = np.concatenate([shrink,pred], axis=3)
            img_out = img_out.reshape(32, 32, 3)
            enlarge = cv2.resize(img_out, (0, 0), fx=20, fy=11.25, interpolation=cv2.INTER_CUBIC)
            print(i)
            cv2.imwrite("frames/{}.jpg".format(i), enlarge)
            i += 1
            writer.write(enlarge)
    else:
        print('The video failed to open！')
    writer.release()