#! /usr/bin/python import time
import rospy
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import scipy.ndimage as ndimage
from skimage.draw import circle
from skimage.feature import peak_local_max
from std_msgs.msg import Float32MultiArray
from src.models.grasp_detectors.unet import UNet
import torch
from skimage.draw import polygon
from skimage.filters import gaussian
from skimage.transform import resize
from skimage.transform import pyramid_expand
import skimage.morphology
from tqdm import tqdm
from torchvision import transforms
from matplotlib import pyplot as plt
import sys
import time

import sensor_msgs.point_cloud2 as pc2
import ctypes
import struct


bridge = CvBridge()

rospy.init_node('grasp_detection')

# Get the camera parameters
camera_info_msg = rospy.wait_for_message('/xtion/depth/camera_info', CameraInfo)
K = camera_info_msg.K  # camera matrix
fx = K[0]
cx = K[2]
fy = K[4]
cy = K[5]
P = camera_info_msg.P  # projection matrix
H = camera_info_msg.height
W = camera_info_msg.width
baseline = 75 / 1000  # 7.5 cm, measured by hand (same as kinect)

# Output publisher.
cmd_pub = rospy.Publisher('/orange/out/command', Float32MultiArray, queue_size=1)

# Load the Network.
net = UNet(1,3)
MODEL_FILE = '/home/hypatia/projects/m2ore/models_trained/unet_best2_grasp_point_jacquard.pt'
chkpt = torch.load(MODEL_FILE, map_location=torch.device('cpu'))
net.load_state_dict(chkpt['model_state_dict'], strict=False)
net = net.cpu()
# print(net)

# Execution Timing
class TimeIt:
    def __init__(self, s):
        self.s = s
        self.t0 = None
        self.t1 = None
        self.print_output = False

    def __enter__(self):
        self.t0 = time.time()

    def __exit__(self, t, value, traceback):
        self.t1 = time.time()
        if self.print_output:
            print('%s: %s' % (self.s, self.t1 - self.t0))

def image_callback(msg):  
    # print("Received image!")
    try:
        # Convert your ROS Image message to OpenCV2
        cv2_img = bridge.imgmsg_to_cv2(msg, "bgr8")
        # print cv2_img.shape

    except CvBridgeError, e:
        print(e)
    else:
        with TimeIt('Crop'):
            # Crop a square out of the middle of the image and resize it to 300*300
            crop_size = 350
            rgb_crop = cv2.resize(cv2_img[(480-crop_size)//2:(480-crop_size)//2+crop_size, (640-crop_size)//2:(640-crop_size)//2+crop_size], (320, 320))

        # Save your OpenCV2 image as a jpeg 
        cv2.imwrite('rgb_image.jpg', rgb_crop)


# src: https://answers.ros.org/question/208834/read-colours-from-a-pointcloud2-python/
def unpack_rgb(rgb_float):

    # cast float32 to int so that bitwise operations are possible
    s = struct.pack('>f' , rgb_float)
    i = struct.unpack('>l', s)[0]
    # you can get back the float value by the inverse operations
    pack = ctypes.c_uint32(i).value
    r = (pack & 0x00FF0000) >> 16
    g = (pack & 0x0000FF00) >> 8
    b = (pack & 0x000000FF)

    return r, g, b


def pcl_callback(msg):
    # read point cloud data point
    for data in pc2.read_points(msg, field_names=("x", "y", "z", "rgb"), skip_nans=True):
        x, y, z, rgb = data
        r, g, b = unpack_rgb(rgb)

        print("X: {:.4f}, Y: {:.4f}, Z: {:.4f}, R: {}, G: {}, B: {}".format(x, y, z, r, g, b))


def depth_callback(msg):
    global model
    global graph
    global prev_mp
    global ROBOT_Z
    global fx, cx, fy, cy    
    print("Received depth!")
    try:
        # old code
        #msg.encoding = "mono16"
        #depth = bridge.imgmsg_to_cv2(msg, "mono8")
        #depth_array = np.array(depth, dtype = np.dtype('f8'))
        #depth_norm = cv2.normalize(depth_array, depth_array, 0, 1, cv2.NORM_MINMAX)

        # fix property
        if msg.encoding == "16UC1":  # 16 bit grayscale
            msg.encoding = "mono16"
        elif msg.encoding == "8UC1":  # (8 bit) grayscale
            msg.encoding = "mono8"
        # Convert your ROS Image message to OpenCV2
        # Depth information is stored as uint_16 in mm!
        depth = bridge.imgmsg_to_cv2(msg)  # don't do format conversion!
        depth = depth.astype(np.float)

        pix = (msg.width/2, msg.height/2)
        print("Depth at center({}, {}): {:.2f} mm".format(pix[0], pix[1], depth[pix[1], pix[0]]))


    except CvBridgeError, e:
        print(e)
    else:
        with TimeIt('Crop'):
            # Crop a square out of the middle of the depth and resize it to 300*300
            crop_size = 350
            crop_margin_y, crop_margin_x = (H-crop_size)//2, (W-crop_size)//2
            depth_crop = cv2.resize(depth[crop_margin_y:crop_size+crop_margin_y, crop_margin_x:crop_size+crop_margin_x], (320, 320))
            # depth_scale = np.abs(depth_crop).max()
            # # print(depth_scale)
            # cv2.imshow("depth image", depth)
            # cv2.waitKey(2)

            # # Replace nan with 0 for inpainting.
            # depth_crop = depth_crop.copy()
            # depth_nan = np.isnan(depth_crop).copy()
            # depth_crop[depth_nan] = 0
            # print depth_crop.shape

        # with TimeIt('Inpaint'):
        #     # open cv inpainting does weird things at the border.
        #     depth_crop = cv2.copyMakeBorder(depth_crop, 1, 1, 1, 1, cv2.BORDER_DEFAULT)

        #     mask = (depth_crop == 0).astype(np.uint8)
        #     # Scale to keep as float, but has to be in bounds -1:1 to keep opencv happy.
        #     depth_scale = np.abs(depth_crop).max()
        #     depth_crop = depth_crop.astype(np.float32)/depth_scale  # Has to be float32, 64 not supported.

        #     depth_crop = cv2.inpaint(depth_crop, mask, 1, cv2.INPAINT_NS)

        #     # Back to original size and value range.
        #     depth_crop = depth_crop[1:-1, 1:-1]
        #     depth_crop = depth_crop * depth_scale

        with TimeIt('Calculate Depth'):
            depth_mean = np.max(depth_crop)
            # Figure out roughly the depth in mm of the part between the grippers for collision avoidance.
            depth_center = depth_crop[100:200, 100:200].flatten()
            depth_center.sort()
            depth_center = depth_center.max()
            # print depth_center

        # Save your OpenCV2 image as a jpeg 
        cv2.imwrite('camera_image.tiff', depth_crop)

        # Start model inference
        # with TimeIt('Inference'):

        # normalize depth
        depth_norm = cv2.normalize(depth, None, 0, 1, cv2.NORM_MINMAX)

        # convert to tensor and clamp
        depth_img = transforms.functional.to_tensor(depth_norm).float()
        depth_img = torch.clamp(depth_img - depth_img.mean(), -1, 1)

        # run inference
        with torch.no_grad():
           pos, cos, sin, width, graspness, bins = net(depth_img.unsqueeze(0))
        pos_g = pos * torch.sigmoid(graspness.unsqueeze(1))
        pos_f = pos_g * torch.sigmoid(bins)

        # Calculate the angle map.
        cos_out = cos.squeeze().numpy()
        sin_out = sin.squeeze().numpy()
        ang_out = np.arctan2(sin_out, cos_out)/2.0

        width_out = width.squeeze().numpy() *150  # Scaled 0-150:0-1

        temp = pos_f.squeeze()
        temp = temp.numpy()
        max_pixel = np.unravel_index(np.argmax(temp, axis=None), temp.shape)

        ang = ang_out[max_pixel[0], max_pixel[1], max_pixel[2]]
        width = width_out[max_pixel[0], max_pixel[1], max_pixel[2]]
        # Convert max_pixel back to uncropped/resized image coordinates in order to do the camera transform.
        # max_pix = ((np.array(max_pixel[1:2])* crop_size / 320.0 ) + np.array([(480 - crop_size)//2, (640 - crop_size) // 2]))
        max_pix = ((np.array(max_pixel[1:2]) / 320.0 * crop_size) + np.array([(480 - crop_size)//2, (640 - crop_size) // 2]))
        

        max_pix = np.round(max_pix).astype(np.int)
        #print(max_pix)

        point_depth = float(depth[max_pix[0], max_pix[1]])

        # These magic numbers are my camera intrinsic parameters.
        x = (max_pixel[2] - cx)/(fx) * point_depth
        y = (max_pixel[1] - cy)/(fy) * point_depth
        z = point_depth
        
        print("m2ore found grasp point! X: {:.4f}, Y: {:.4f}, Z: {:.4f}".format(x, y, z))

        if np.isnan(z):
            return



    with TimeIt('Publish'):
        # Output the best grasp pose relative to camera.
        cmd_msg = Float32MultiArray()
        cmd_msg.data = [x, y, z, ang, width, depth_center]
        cmd_pub.publish(cmd_msg)        


if __name__ == "__main__":
    # TODO: check if correct topic
    depth_sub = rospy.Subscriber('/xtion/depth/image_rect_raw', Image, depth_callback, queue_size=1)
    #depth_sub = rospy.Subscriber('/xtion/depth_registered/image_raw', Image, depth_callback, queue_size=1)

    #pcl_sub = rospy.Subscriber('/throttle_filtering_points/filtered_points', PointCloud2, pcl_callback, queue_size=1)
    rgb_sub = rospy.Subscriber('/xtion/rgb/image_raw', Image, image_callback, queue_size=1)

    while not rospy.is_shutdown():
        rospy.spin()
