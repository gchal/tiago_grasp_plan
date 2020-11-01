#! /usr/bin/python
import time
import rospy
from sensor_msgs.msg import Image, CameraInfo
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



bridge = CvBridge()

rospy.init_node('grasp_detection')

# Get the camera parameters
camera_info_msg = rospy.wait_for_message('/xtion/depth/camera_info', CameraInfo)
K = camera_info_msg.K
fx = K[0]
cx = K[2]
fy = K[4]
cy = K[5]


# Output publisher.
cmd_pub = rospy.Publisher('/orange/out/command', Float32MultiArray, queue_size=1)

# Load the Network.
net = UNet(1,3)
MODEL_FILE = PATH_TO_MODEL
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

    except CvBridgeError, e:
        print(e)
    else:
        with TimeIt('Crop'):
            # Crop a square out of the middle of the image and resize it to 320*320
            crop_size = 350
            rgb_crop = cv2.resize(cv2_img[(480-crop_size)//2:(480-crop_size)//2+crop_size, (640-crop_size)//2:(640-crop_size)//2+crop_size], (320, 320))

        # Save your OpenCV2 image as a jpeg 
        cv2.imwrite('rgb_image.jpg', rgb_crop)

def depth_callback(msg):
    global model
    global graph
    global prev_mp
    global ROBOT_Z
    global fx, cx, fy, cy    
    print("Received depth!")
    try:
        msg.encoding = "mono16"
        # Convert your ROS Image message to OpenCV2
        depth = bridge.imgmsg_to_cv2(msg, "mono8")
        # print depth.shape

    except CvBridgeError, e:
        print(e)
    else:
        with TimeIt('Crop'):
            # Crop a square out of the middle of the depth and resize it to 300*300
            crop_size = 350
            depth_crop = cv2.resize(depth[(480-crop_size)//2:(480-crop_size)//2+crop_size, (640-crop_size)//2:(640-crop_size)//2+crop_size], (320, 320))
            depth_scale = np.abs(depth_crop).mean()
            print(depth_scale)

        with TimeIt('Calculate Depth'):
            depth_mean = np.max(depth_crop)
            # Figure out roughly the depth in mm of the part between the grippers for collision avoidance.
            depth_center = depth_crop[100:200, 100:200].flatten()
            depth_center.sort()
            depth_center = depth_center.max() * 1000.0
            # print depth_center

        # Save your OpenCV2 depth image as a tiff 
        cv2.imwrite('depth_image.tiff', depth_crop)


        # Start model inference
        with TimeIt('Inference'):
        depth_img = transforms.functional.to_tensor(depth).float()
        depth_img = torch.clamp(depth_img - depth_img.mean(), -1, 1)
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
        max_pix = ((np.array(max_pixel[1:2]) / 320.0 * crop_size) + np.array([(480 - crop_size)//2, (640 - crop_size) // 2]))
        max_pix = np.round(max_pix).astype(np.int)

        point_depth = float(depth[max_pix[0], max_pix[1]])/np.max(depth_crop)

        # These magic numbers are my camera intrinsic parameters.
        x = (max_pixel[2] - cx)/(fx) * point_depth
        y = (max_pixel[1] - cy)/(fy) * point_depth
        z = point_depth
        if np.isnan(z):
            return


    with TimeIt('Publish'):
        # Output the best grasp pose relative to camera.
        cmd_msg = Float32MultiArray()
        cmd_msg.data = [x, y, z, ang, width, depth_center]
        cmd_pub.publish(cmd_msg)        

depth_sub = rospy.Subscriber('/xtion/depth/image_raw', Image, depth_callback, queue_size=1)
rgb_sub = rospy.Subscriber('/xtion/rgb/image_raw', Image, image_callback, queue_size=1)

while not rospy.is_shutdown():
    rospy.spin()
