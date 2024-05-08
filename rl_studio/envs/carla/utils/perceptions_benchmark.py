import argparse
import math
import os, sys
import shutil
import time
from pathlib import Path

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from collections import Counter

print(sys.path)
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import scipy.special
import numpy as np
import torchvision.transforms as transforms
import PIL.Image as image
from PIL import Image

from rl_studio.envs.carla.utils.DemoDataset import LoadImages, LoadStreams
from tqdm import tqdm
normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

detection_mode = 'yolop'
# detection_mode = 'lane_detector'

show_images = False
apply_mask = True

labels = {
    "/home/ruben/Desktop/RL-Studio/rl_studio/inference/input/00044446.png": [0.009375, 0.003125, 0., -0.00625, -0.009375],
    "/home/ruben/Desktop/RL-Studio/rl_studio/inference/input/00048953.png": [0.115625, 0.096875, 0.078125, 0.0625, 0.04375],
    "/home/ruben/Desktop/RL-Studio/rl_studio/inference/input/00049145.png": [0.1125, 0.0625, 0.015625, -0.015625, -0.0625],
    "/home/ruben/Desktop/RL-Studio/rl_studio/inference/input/00049753.png": [0.30625 , 0.390625, 0.475,    0.53125,  1.      ],
    "/home/ruben/Desktop/RL-Studio/rl_studio/inference/input/00049843.png": [-0.046875 , 0.040625 , 0.128125 , 0.1875  ,  0.275   ],
    "/home/ruben/Desktop/RL-Studio/rl_studio/inference/input/00050172.png": [-0.090625, -0.128125, -0.1625, -0.184375, -0.221875],
    "/home/ruben/Desktop/RL-Studio/rl_studio/inference/input/00050239.png": [-0.1, -0.1375, -0.178125, -0.20625, -0.246875],
    "/home/ruben/Desktop/RL-Studio/rl_studio/inference/input/00050338.png": [-0.096875, -0.140625, -0.18125, -0.209375, -0.25],
    "/home/ruben/Desktop/RL-Studio/rl_studio/inference/input/00055943.png": [ 0.140625  ,0.084375 , 0.028125 ,-0.0125 ,  -0.06875 ],
    "/home/ruben/Desktop/RL-Studio/rl_studio/inference/input/00056039.png": [0.028125, 0.021875, 0.01875, 0.015625, 0.0125],
    "/home/ruben/Desktop/RL-Studio/rl_studio/inference/input/00056136.png": [0.009375, 0.00625, 0.00625, 0.00625, 0.003125],
    "/home/ruben/Desktop/RL-Studio/rl_studio/inference/input/00056244.png": [0., 0., -0.003125, -0.00625, -0.00625],
    "/home/ruben/Desktop/RL-Studio/rl_studio/inference/input/00056346.png": [0.003125, -0.00625, -0.0125, -0.01875, -0.025],
    "/home/ruben/Desktop/RL-Studio/rl_studio/inference/input/00056448.png": [-0.003125, 0., 0.00625, 0.009375, 0.0125],
    "/home/ruben/Desktop/RL-Studio/rl_studio/inference/input/00056557.png": [0., 0.003125, 0.003125, 0.00625, 0.00625],
    "/home/ruben/Desktop/RL-Studio/rl_studio/inference/input/00056843.png":  [0.153125, 0.109375, 0.071875, 0.04375  ,0.003125],
    "/home/ruben/Desktop/RL-Studio/rl_studio/inference/input/00056926.png": [0.01875, 0.0125, 0.009375, 0.00625, 0.],
    "/home/ruben/Desktop/RL-Studio/rl_studio/inference/input/00057008.png": [0.015625, 0.009375, 0.00625, 0.003125, 0.],
    "/home/ruben/Desktop/RL-Studio/rl_studio/inference/input/00057326.png": [-0.009375, -0.009375, -0.0125, -0.015625, -0.015625],
    "/home/ruben/Desktop/RL-Studio/rl_studio/inference/input/00057415.png": [0.05  ,   0.04375,  0.0375,   0.034375, 0.028125],
    "/home/ruben/Desktop/RL-Studio/rl_studio/inference/input/00057517.png": [ 0.121875  ,0.078125  ,0.040625 , 0.0125 ,  -0.028125],
    "/home/ruben/Desktop/RL-Studio/rl_studio/inference/input/00057618.png": [0.14375, 0.1125, 0.084375, 0.065625, 0.0375],
    "/home/ruben/Desktop/RL-Studio/rl_studio/inference/input/00057706.png": [0.1125, 0.08125, 0.046875, 0.025, -0.009375],
    "/home/ruben/Desktop/RL-Studio/rl_studio/inference/input/00057811.png": [0.015625, 0.0125, 0.0125, 0.009375, 0.00625],
    "/home/ruben/Desktop/RL-Studio/rl_studio/inference/input/00057906.png": [0., 0.003125, 0.00625, 0.00625, 0.009375],
    "/home/ruben/Desktop/RL-Studio/rl_studio/inference/input/00058004.png": [0.021875, 0.021875, 0.01875, 0.015625, 0.0125],
    "/home/ruben/Desktop/RL-Studio/rl_studio/inference/input/00058106.png": [0.01875, 0.01875, 0.01875, 0.01875, 0.01875],
    "/home/ruben/Desktop/RL-Studio/rl_studio/inference/input/00058314.png": [0.0375, 0.021875, 0.00625, -0.003125, -0.015625],
    "/home/ruben/Desktop/RL-Studio/rl_studio/inference/input/00058409.png": [0.159375, 0.11875, 0.078125, 0.05, 0.009375],
    "/home/ruben/Desktop/RL-Studio/rl_studio/inference/input/00058510.png": [0.140625, 0.103125, 0.06875, 0.04375, 0.00625],
    "/home/ruben/Desktop/RL-Studio/rl_studio/inference/input/00058694.png": [0.0375, 0.028125, 0.01875, 0.0125, 0.003125],
    "/home/ruben/Desktop/RL-Studio/rl_studio/inference/input/00058886.png": [ 0.009375,  0.003125, -0.00625,  -0.0125  , -0.021875],
    "/home/ruben/Desktop/RL-Studio/rl_studio/inference/input/00058984.png": [0.015625, 0.0125, 0.00625, 0.003125, 0.],
    "/home/ruben/Desktop/RL-Studio/rl_studio/inference/input/00059163.png": [0.084375, 0.059375, 0.034375, 0.01875, -0.00625],
    "/home/ruben/Desktop/RL-Studio/rl_studio/inference/input/00059264.png": [0.128125, 0.096875, 0.065625, 0.046875, 0.015625],
    "/home/ruben/Desktop/RL-Studio/rl_studio/inference/input/00059463.png": [0.034375, 0.025, 0.01875, 0.015625, 0.009375],
    "/home/ruben/Desktop/RL-Studio/rl_studio/inference/input/00059558.png": [-0.003125, -0.003125, 0., 0., 0.],
    "/home/ruben/Desktop/RL-Studio/rl_studio/inference/input/00106547.png": [0.125,    0.184375 ,0.246875, 0.2875   ,1.      ],
    "/home/ruben/Desktop/RL-Studio/rl_studio/inference/input/00106767.png": [-0.05, -0.00625, 0.040625, 0.06875, 0.1125],
    "/home/ruben/Desktop/RL-Studio/rl_studio/inference/input/00048848.png":[0.265625, 0.265625, 0.2625,   0.265625, 0.2625  ],
    "/home/ruben/Desktop/RL-Studio/rl_studio/inference/input/00051465.png": [-0.009375, 0.003125, 0.015625, 0.021875,
                                                                             0.034375],
    "/home/ruben/Desktop/RL-Studio/rl_studio/inference/input/00049646.png": [0.259375, 0.259375, 0.25625,  0.253125, 0.253125],
}


def select_device(logger=None, device='', batch_size=None):
    # device = 'cpu' or '0' or '0,1,2,3'
    cpu_request = device.lower() == 'cpu'
    if device and not cpu_request:  # if device requested other than 'cpu'
        os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable
        assert torch.cuda.is_available(), 'CUDA unavailable, invalid device %s requested' % device  # check availablity

    cuda = False if cpu_request else torch.cuda.is_available()
    if cuda:
        c = 1024 ** 2  # bytes to MB
        ng = torch.cuda.device_count()
        if ng > 1 and batch_size:  # check that batch_size is compatible with device_count
            assert batch_size % ng == 0, 'batch-size %g not multiple of GPU count %g' % (batch_size, ng)
        x = [torch.cuda.get_device_properties(i) for i in range(ng)]
        s = f'Using torch {torch.__version__} '
        for i in range(0, ng):
            if i == 1:
                s = ' ' * len(s)
            if logger:
                logger.info("%sCUDA:%g (%s, %dMB)" % (s, i, x[i].name, x[i].total_memory / c))
    else:
        if logger:
            logger.info(f'Using torch {torch.__version__} CPU')

    if logger:
        logger.info('')  # skip a line
    return torch.device('cuda:0' if cuda else 'cpu')

device = select_device()
x_row = [ 240, 270, 300, 320, 350 ]
NO_DETECTED = 0

if detection_mode == 'yolop':
    from rl_studio.envs.carla.utils.yolop.YOLOP import get_net
    import torchvision.transforms as transforms

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    # INIT YOLOP
    yolop_model = get_net()
    checkpoint = torch.load("/home/ruben/Desktop/RL-Studio/rl_studio/envs/carla/utils/yolop/weights/End-to-end.pth",
                            map_location=device)
    yolop_model.load_state_dict(checkpoint['state_dict'])
    yolop_model = yolop_model.to(device)
elif detection_mode == "lane_detector":
    lane_model = torch.load('/home/ruben/Desktop/RL-Studio/rl_studio/envs/carla/utils/lane_det/fastai_torch_lane_detector_model.pth')
    lane_model.eval()


def post_process(ll_segment):
    ''''
    Lane line post-processing
    '''
    # ll_segment = morphological_process(ll_segment, kernel_size=5, func_type=cv2.MORPH_OPEN)
    # ll_segment = morphological_process(ll_segment, kernel_size=20, func_type=cv2.MORPH_CLOSE)
    # return ll_segment
    # ll_segment = morphological_process(ll_segment, kernel_size=4, func_type=cv2.MORPH_OPEN)
    # ll_segment = morphological_process(ll_segment, kernel_size=8, func_type=cv2.MORPH_CLOSE)

    # Step 1: Create a binary mask image representing the trapeze
    mask = np.zeros_like(ll_segment)
    # pts = np.array([[300, 250], [-500, 600], [800, 600], [450, 260]], np.int32)
    pts = np.array([[180, 200], [-50, 450], [630, 450], [440, 200]], np.int32)
    cv2.fillPoly(mask, [pts], (255, 255, 255))  # Fill trapeze region with white (255)
    cv2.imshow("applied_mask", mask) if show_images else None

    # Step 2: Apply the mask to the original image
    ll_segment_masked = cv2.bitwise_and(ll_segment, mask)
    ll_segment_excluding_mask = cv2.bitwise_not(mask)
    # Apply the exclusion mask to ll_segment
    ll_segment_excluded = cv2.bitwise_and(ll_segment, ll_segment_excluding_mask)
    cv2.imshow("discarded", ll_segment_excluded) if show_images else None

    return ll_segment_masked


def post_process_hough_lane_det(ll_segment):
    # Step 4: Perform Hough transform to detect lines
    # ll_segment = cv2.dilate(ll_segment, (3, 3), iterations=4)
    # ll_segment = cv2.erode(ll_segment, (3, 3), iterations=2)
    cv2.imshow("preprocess", ll_segment) if show_images else None
    # edges = cv2.Canny(ll_segment, 50, 100)

    # Reapply HoughLines on the dilated image
    lines = cv2.HoughLinesP(
        ll_segment,  # Input edge image
        1,  # Distance resolution in pixels
        np.pi / 90,  # Angle resolution in radians
        threshold=7,  # Min number of votes for valid line
        minLineLength=5,  # Min allowed length of line
        maxLineGap=60  # Max allowed gap between line for joining them
    )
    # Sort lines by their length
    # lines = sorted(lines, key=lambda x: x[0][0] * np.sin(x[0][1]), reverse=True)[:5]

    # Create a blank image to draw lines
    line_mask = np.zeros_like(ll_segment, dtype=np.uint8)  # Ensure dtype is uint8

    # Iterate over points
    for points in lines if lines is not None else []:
        # Extracted points nested in the list
        x1, y1, x2, y2 = points[0]
        # Draw the lines joing the points
        # On the original image
        cv2.line(line_mask, (x1, y1), (x2, y2), (255, 255, 255), 2)

    # Postprocess the detected lines
    # line_mask = morphological_process(line_mask, kernel_size=5, func_type=cv2.MORPH_OPEN)
    # line_mask = morphological_process(line_mask, kernel_size=5, func_type=cv2.MORPH_CLOSE)
    # kernel = np.ones((3, 3), np.uint8)  # Adjust the size as needed
    # eroded_image = cv2.erode(line_mask, kernel, iterations=1)
    cv2.imshow("hough", line_mask) if show_images else None

    return lines


def post_process_hough_yolop_v1(ll_segment):
    # Step 4: Perform Hough transform to detect lines
    ll_segment = cv2.dilate(ll_segment, (3, 3), iterations=4)
    ll_segment = cv2.erode(ll_segment, (3, 3), iterations=2)
    cv2.imshow("preprocess", ll_segment) if show_images else None
    lines = cv2.HoughLinesP(
        ll_segment,  # Input edge image
        1,  # Distance resolution in pixels
        np.pi/60,  # Angle resolution in radians
        threshold=8,  # Min number of votes for valid line
        minLineLength=8,  # Min allowed length of line
        maxLineGap=20  # Max allowed gap between line for joining them
    )

    line_mask = np.zeros_like(ll_segment, dtype=np.uint8)  # Ensure dtype is uint8

    # Draw the detected lines on the blank image
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_mask, (x1, y1), (x2, y2), (255, 255, 255), 2)  # Draw lines in white (255, 255, 255)

    # Apply dilation to the line image

    # edges = cv2.Canny(line_mask, 50, 100)

    # cv2.imshow("intermediate_hough", ll_segment) if show_images else None

    # Reapply HoughLines on the dilated image
    lines = cv2.HoughLinesP(
        ll_segment,  # Input edge image
        1,  # Distance resolution in pixels
        np.pi / 90,  # Angle resolution in radians
        threshold=35,  # Min number of votes for valid line
        minLineLength=15,  # Min allowed length of line
        maxLineGap=20  # Max allowed gap between line for joining them
    )
    # Sort lines by their length
    # lines = sorted(lines, key=lambda x: x[0][0] * np.sin(x[0][1]), reverse=True)[:5]

    # Create a blank image to draw lines
    line_mask = np.zeros_like(ll_segment, dtype=np.uint8)  # Ensure dtype is uint8

    # Iterate over points
    for points in lines if lines is not None else []:
        # Extracted points nested in the list
        x1, y1, x2, y2 = points[0]
        # Draw the lines joing the points
        # On the original image
        cv2.line(line_mask, (x1, y1), (x2, y2), (255, 255, 255), 2)

    # Postprocess the detected lines
    # line_mask = morphological_process(line_mask, kernel_size=5, func_type=cv2.MORPH_OPEN)
    # line_mask = morphological_process(line_mask, kernel_size=5, func_type=cv2.MORPH_CLOSE)
    # kernel = np.ones((3, 3), np.uint8)  # Adjust the size as needed
    # eroded_image = cv2.erode(line_mask, kernel, iterations=1)
    cv2.imshow("hough", line_mask)  if show_images else None

    return lines

def post_process_hough_yolop(ll_segment):
    # Step 4: Perform Hough transform to detect lines
    cv2.imshow("preprocess", ll_segment) if show_images else None

    # Reapply HoughLines on the dilated image
    lines = cv2.HoughLinesP(
        ll_segment,  # Input edge image
        1,  # Distance resolution in pixels
        np.pi / 90,  # Angle resolution in radians
        threshold=40,  # Min number of votes for valid line
        minLineLength=10,  # Min allowed length of line
        maxLineGap=30  # Max allowed gap between line for joining them
    )
    # Sort lines by their length
    # lines = sorted(lines, key=lambda x: x[0][0] * np.sin(x[0][1]), reverse=True)[:5]

    # Create a blank image to draw lines
    line_mask = np.zeros_like(ll_segment, dtype=np.uint8)  # Ensure dtype is uint8

    # Iterate over points
    for points in lines if lines is not None else []:
        # Extracted points nested in the list
        x1, y1, x2, y2 = points[0]
        # Draw the lines joing the points
        # On the original image
        cv2.line(line_mask, (x1, y1), (x2, y2), (255, 255, 255), 2)

    # Postprocess the detected lines
    # line_mask = morphological_process(line_mask, kernel_size=5, func_type=cv2.MORPH_OPEN)
    # line_mask = morphological_process(line_mask, kernel_size=5, func_type=cv2.MORPH_CLOSE)
    # kernel = np.ones((3, 3), np.uint8)  # Adjust the size as needed
    # eroded_image = cv2.erode(line_mask, kernel, iterations=1)
    cv2.imshow("hough", line_mask)  if show_images else None

    return lines


def post_process_hough_programmatic(ll_segment):
    # Step 4: Perform Hough transform to detect lines
    lines = cv2.HoughLinesP(
        ll_segment,  # Input edge image
        1,  # Distance resolution in pixels
        np.pi/60,  # Angle resolution in radians
        threshold=20,  # Min number of votes for valid line
        minLineLength=10,  # Min allowed length of line
        maxLineGap=50  # Max allowed gap between line for joining them
    )

    line_mask = np.zeros_like(ll_segment, dtype=np.uint8)  # Ensure dtype is uint8

    # Draw the detected lines on the blank image
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_mask, (x1, y1), (x2, y2), (255, 255, 255), 2)  # Draw lines in white (255, 255, 255)

    # Apply dilation to the line image

    edges = cv2.Canny(line_mask, 50, 100)

    cv2.imshow("intermediate_hough", edges)  if show_images else None

    # Reapply HoughLines on the dilated image
    lines = cv2.HoughLinesP(
        edges,  # Input edge image
        1,  # Distance resolution in pixels
        np.pi / 60,  # Angle resolution in radians
        threshold=20,  # Min number of votes for valid line
        minLineLength=13,  # Min allowed length of line
        maxLineGap=50  # Max allowed gap between line for joining them
    )
    # Sort lines by their length
    # lines = sorted(lines, key=lambda x: x[0][0] * np.sin(x[0][1]), reverse=True)[:5]

    # Create a blank image to draw lines
    line_mask = np.zeros_like(ll_segment, dtype=np.uint8)  # Ensure dtype is uint8

    # Iterate over points
    for points in lines if lines is not None else []:
        # Extracted points nested in the list
        x1, y1, x2, y2 = points[0]
        # Draw the lines joing the points
        # On the original image
        cv2.line(line_mask, (x1, y1), (x2, y2), (255, 255, 255), 2)

    # Postprocess the detected lines
    # line_mask = morphological_process(line_mask, kernel_size=5, func_type=cv2.MORPH_OPEN)
    # line_mask = morphological_process(line_mask, kernel_size=5, func_type=cv2.MORPH_CLOSE)
    # kernel = np.ones((3, 3), np.uint8)  # Adjust the size as needed
    # eroded_image = cv2.erode(line_mask, kernel, iterations=1)
    cv2.imshow("hough", line_mask) if show_images else None

    return lines

def extend_lines(lines, image_height):
    extended_lines = []
    for line in lines if lines is not None else []:
        x1, y1, x2, y2 = line[0]
        # Calculate slope and intercept
        if x2 - x1 != 0:
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1
            # Calculate new endpoints to extend the line
            x1_extended = int(x1 - 2 * (x2 - x1))  # Extend 2 times the original length
            y1_extended = int(slope * x1_extended + intercept)
            x2_extended = int(x2 + 2 * (x2 - x1))  # Extend 2 times the original length
            y2_extended = int(slope * x2_extended + intercept)
            # Ensure the extended points are within the image bounds
            x1_extended = max(0, min(x1_extended, image_height - 1))
            y1_extended = max(0, min(y1_extended, image_height - 1))
            x2_extended = max(0, min(x2_extended, image_height - 1))
            y2_extended = max(0, min(y2_extended, image_height - 1))
            # Append the extended line to the list
            extended_lines.append([(x1_extended, y1_extended, x2_extended, y2_extended)])
    return extended_lines


def detect_yolop(raw_image):
    # Get names and colors
    names = yolop_model.module.names if hasattr(yolop_model, 'module') else yolop_model.names

    # Run inference
    img = transform(raw_image).to(device)
    img = img.float()  # uint8 to fp16/32
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    det_out, da_seg_out, ll_seg_out = yolop_model(img)

    ll_seg_mask = torch.nn.functional.interpolate(ll_seg_out, scale_factor=int(1), mode='bicubic')
    _, ll_seg_mask = torch.max(ll_seg_mask, 1)
    ll_seg_mask = ll_seg_mask.int().squeeze().cpu().numpy()

    return ll_seg_mask


def merge_and_extend_lines(lines, ll_segment):
    # Merge parallel lines
    merged_lines = []
    for line in lines if lines is not None else []:
        x1, y1, x2, y2 = line[0]
        angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi  # Compute the angle of the line

        # Check if there is a similar line in the merged lines
        found = False
        for merged_line in merged_lines:
            angle_diff = abs(merged_line['angle'] - angle)
            if angle_diff < 20 and abs(angle) > 25:  # Adjust this threshold based on your requirement
                # Merge the lines by averaging their coordinates
                merged_line['x1'] = (merged_line['x1'] + x1) // 2
                merged_line['y1'] = (merged_line['y1'] + y1) // 2
                merged_line['x2'] = (merged_line['x2'] + x2) // 2
                merged_line['y2'] = (merged_line['y2'] + y2) // 2
                found = True
                break

        if not found and abs(angle) > 25:
            merged_lines.append({'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'angle': angle})

    # Draw the merged lines on the original image
    merged_image = np.zeros_like(ll_segment, dtype=np.uint8)  # Ensure dtype is uint8
    #if len(merged_lines) < 2 or len(merged_lines) > 2:
    #    print("ii")
    for line in merged_lines:
        x1, y1, x2, y2 = line['x1'], line['y1'], line['x2'], line['y2']
        cv2.line(merged_image, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Display the original image with merged lines
    cv2.imshow('Merged Lines', merged_image)  if show_images else None

    line_mask = np.zeros_like(ll_segment, dtype=np.uint8)  # Ensure dtype is uint8

    # Step 5: Perform linear regression on detected lines
    # Iterate over detected lines
    for line in merged_lines if lines is not None else []:
        # Extract endpoints of the line
        x1, y1, x2, y2 = line['x1'], line['y1'], line['x2'], line['y2']

        # Fit a line to the detected points
        vx, vy, x0, y0 = cv2.fitLine(np.array([[x1, y1], [x2, y2]], dtype=np.float32), cv2.DIST_L2, 0, 0.01, 0.01)

        # Calculate the slope and intercept of the line
        slope = vy / vx

        # Extend the line if needed (e.g., to cover the entire image width)
        extended_y1 = ll_segment.shape[0] - 1  # Bottom of the image
        extended_x1 = x0 + (extended_y1 - y0) / slope
        extended_y2 = 0  # Upper part of the image
        extended_x2 = x0 + (extended_y2 - y0) / slope

        if extended_x1 > 2147483647 or extended_x2 > 2147483647 or extended_y1 > 2147483647 or extended_y2 > 2147483647:
            cv2.line(line_mask, (int(x0), 0), (int(x0), ll_segment.shape[0] - 1), (255, 0, 0), 2)
            continue
        # Draw the extended line on the image
        cv2.line(line_mask, (int(extended_x1), extended_y1), (int(extended_x2), extended_y2), (255, 0, 0), 2)
    return line_mask

def detect_lane_detector(raw_image):
    image_tensor = raw_image.transpose(2, 0, 1).astype('float32') / 255
    x_tensor = torch.from_numpy(image_tensor).to("cuda").unsqueeze(0)
    model_output = torch.softmax(lane_model.forward(x_tensor), dim=1).cpu().numpy()
    return model_output

def lane_detection_overlay(image, left_mask, right_mask):
    res = np.copy(image)
    # We show only points with probability higher than 0.5
    res[left_mask > 0.5, :] = [255,0,0]
    res[right_mask > 0.5,:] = [0, 0, 255]
    return res

def detect_lines(raw_image, detection_mode):
    if detection_mode == 'programmatic':
        gray = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
        # mask_white = cv2.inRange(gray, 200, 255)
        # mask_image = cv2.bitWiseAnd(gray, mask_white)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        ll_segment = cv2.Canny(blur, 50, 100)
        cv2.imshow("raw", ll_segment) if show_images else None
        processed = post_process(ll_segment)
        lines = post_process_hough_programmatic(processed if apply_mask else ll_segment)
    elif detection_mode == 'yolop':
        with torch.no_grad():
            ll_segment = (detect_yolop(raw_image) * 255).astype(np.uint8)
        cv2.imshow("raw", ll_segment) if show_images else None
        processed = post_process(ll_segment)
        lines = post_process_hough_yolop(processed if apply_mask else ll_segment)
    else:
        with torch.no_grad():
            ll_segment, left_mask, right_mask = detect_lane_detector(raw_image)[0]
        ll_segment = np.zeros_like(raw_image)
        ll_segment = lane_detection_overlay(ll_segment, left_mask, right_mask)
        cv2.imshow("raw", ll_segment) if show_images else None
        # Extract blue and red channels
        blue_channel = ll_segment[:, :, 0]  # Blue channel
        red_channel = ll_segment[:, :, 2]  # Red channel
        # Combine blue and red channels into a grayscale image
        ll_segment = 0.5 * blue_channel + 0.5 * red_channel
        ll_segment = cv2.convertScaleAbs(ll_segment)
        # Display the grayscale image
        processed = post_process(ll_segment)
        lines = post_process_hough_lane_det(processed if apply_mask else ll_segment)

    detected_lines = merge_and_extend_lines(lines, ll_segment)

    # line_mask = morphological_process(line_mask, kernel_size=15, func_type=cv2.MORPH_CLOSE)
    # line_mask = morphological_process(line_mask, kernel_size=5, func_type=cv2.MORPH_OPEN)
    # line_mask = cv2.dilate(line_mask, (15, 15), iterations=15)
    # line_mask = cv2.erode(line_mask, (5, 5), iterations=20)

    # TODO (Ruben) It is quite hardcoded and unrobust. Fix this to enable all lines and more than
    # 1 lane detection and cameras in other positions
    boundary_y = detected_lines.shape[1] * 2 // 6
    # Copy the lower part of the source image into the target image
    detected_lines[:boundary_y, :] = 0
    detected_lines = (detected_lines // 255).astype(np.uint8)  # Keep the lower one-third of the image

    return detected_lines

def choose_lane(distance_to_center_normalized, center_points):
    last_row = len(x_row) - 1
    closest_lane_index = min(enumerate(distance_to_center_normalized[last_row]), key=lambda x: abs(x[1]))[0]
    distances = [array[closest_lane_index] if len(array) > closest_lane_index else min(array) for array in distance_to_center_normalized]
    centers = [array[closest_lane_index] if len(array) > closest_lane_index else min(array) for array in center_points]
    return distances, centers

def choose_lane_v1(distance_to_center_normalized, center_points):
    close_lane_indexes = [min(enumerate(inner_array), key=lambda x: abs(x[1]))[0] for inner_array in
                          distance_to_center_normalized]
    distances = [array[index] for array, index in zip(distance_to_center_normalized, close_lane_indexes)]
    centers = [array[index] for array, index in zip(center_points, close_lane_indexes)]
    return distances, centers


def find_lane_center(mask):
    # Find the indices of 1s in the array
    mask_array = np.array(mask)
    indices = np.where(mask_array > 0.8)[0]

    # If there are no 1s or only one set of 1s, return None
    if len(indices) < 2:
        # TODO (Ruben) For the moment we will be dealing with no detection as a fixed number
        return [NO_DETECTED]

    # Find the indices where consecutive 1s change to 0
    diff_indices = np.where(np.diff(indices) > 1)[0]
    # If there is only one set of 1s, return None
    if len(diff_indices) == 0:
        return [NO_DETECTED]

    interested_line_borders = np.array([], dtype=np.int8)
    # print(indices)
    for index in diff_indices:
        interested_line_borders = np.append(interested_line_borders, indices[index])
        interested_line_borders = np.append(interested_line_borders, int(indices[index+1]))

    midpoints = calculate_midpoints(interested_line_borders)
    # print(midpoints)
    return midpoints


def calculate_midpoints(input_array):
    midpoints = []
    for i in range(0, len(input_array) - 1, 2):
        midpoint = (input_array[i] + input_array[i + 1]) // 2
        midpoints.append(midpoint)
    return midpoints


def detect_missing_points(lines):
    num_points = len(lines)
    max_line_points = max(len(line) for line in lines)
    missing_line_count = sum(1 for line in lines if len(line) < max_line_points)

    return missing_line_count > 0 and missing_line_count <= num_points // 2


def interpolate_missing_points(input_lists, x_row):
    # Find the index of the list with the maximum length
    max_length_index = max(range(len(input_lists)), key=lambda i: len(input_lists[i]))

    # Determine the length of the complete lists
    complete_length = len(input_lists[max_length_index])

    # Initialize a list to store the inferred list
    inferred_list = []

    # Iterate over each index in x_row
    for i, x_value in enumerate(x_row):
        # If the current index is in the list with incomplete points
        if len(input_lists[i]) < complete_length:
            interpolated_list = []
            for points_i in range(complete_length):
                # TODO calculates interpolated point of missing line and then build the interpolated_list
                # Since it is not trivial, we just discard this point from the moment
                interpolated_y = NO_DETECTED
                interpolated_list.append(interpolated_y)
            inferred_list.append(interpolated_list)
        else:
            # If the current list is complete, simply append the corresponding y value
            inferred_list.append(input_lists[i])

    return inferred_list

def calculate_center_v1(mask):
    width = mask.shape[1]
    center_image = width / 2
    lines = [mask[x_row[i], :] for i, _ in enumerate(x_row)]
    center_lane_indexes = [
        find_lane_center(lines[x]) for x, _ in enumerate(lines)
    ]

    # this part consists of checking the number of lines detected in all rows
    # then discarding the rows (set to 1) in which more or less centers are detected
    center_lane_indexes = discard_not_confident_centers(center_lane_indexes)

    center_lane_distances = [
        [center_image - x for x in inner_array] for inner_array in center_lane_indexes
    ]

    # Calculate the average position of the lane lines
    ## normalized distance
    distance_to_center_normalized = [
        np.array(x) / (width - center_image) for x in center_lane_distances
    ]
    return center_lane_indexes, distance_to_center_normalized


def calculate_center(mask):
    width = mask.shape[1]
    center_image = width / 2
    lines = [mask[x_row[i], :] for i, _ in enumerate(x_row)]
    center_lane_indexes = [
        find_lane_center(lines[x]) for x, _ in enumerate(lines)
    ]

    if detect_missing_points(center_lane_indexes):
        center_lane_indexes = interpolate_missing_points(center_lane_indexes, x_row)
    # this part consists of checking the number of lines detected in all rows
    # then discarding the rows (set to 1) in which more or less centers are detected
    center_lane_indexes = discard_not_confident_centers(center_lane_indexes)

    center_lane_distances = [
        [center_image - x for x in inner_array] for inner_array in center_lane_indexes
    ]

    # Calculate the average position of the lane lines
    ## normalized distance
    distance_to_center_normalized = [
        np.array(x) / (width - center_image) for x in center_lane_distances
    ]
    return center_lane_indexes, distance_to_center_normalized


def discard_not_confident_centers(center_lane_indexes):
    # Count the occurrences of each list size leaving out of the equation the non-detected
    size_counter = Counter(len(inner_list) for inner_list in center_lane_indexes if NO_DETECTED not in inner_list)
    # Check if size_counter is empty, which mean no centers found
    if not size_counter:
        return center_lane_indexes
    # Find the most frequent size
    # most_frequent_size = max(size_counter, key=size_counter.get)

    # Iterate over inner lists and set elements to 1 if the size doesn't match majority
    result = []
    for inner_list in center_lane_indexes:
        # if len(inner_list) != most_frequent_size:
        if len(inner_list) < 1: # If we don't see the 2 lanes, we discard the row
            inner_list = [NO_DETECTED] * len(inner_list)  # Set all elements to 1
        result.append(inner_list)

    return result

def get_ll_seg_image(dists, ll_segment, suffix="",  name='ll_seg'):
    ll_segment_int8 = (ll_segment * 255).astype(np.uint8)
    ll_segment_all = [np.copy(ll_segment_int8),np.copy(ll_segment_int8),np.copy(ll_segment_int8)]

    # draw the midpoint used as right center lane
    for index, dist in zip(x_row, dists):
        # Set the value at the specified index and distance to 1
        add_midpoints(ll_segment_all[0], index, dist)

    # draw a line for the selected perception points
    for index in x_row:
        for i in range(630):
            ll_segment_all[0][index][i] = 255
    ll_segment_stacked = np.stack(ll_segment_all, axis=-1)
    # We now show the segmentation and center lane postprocessing
    cv2.imshow(name + suffix, ll_segment_stacked) if show_images else None
    cv2.waitKey(1)  # 1 millisecond
    return ll_segment_stacked

def add_midpoints(ll_segment, index, dist):
    # Set the value at the specified index and distance to 1
    draw_dash(index, dist, ll_segment)
    draw_dash(index + 2, dist, ll_segment)
    draw_dash(index + 1, dist, ll_segment)
    draw_dash(index - 1, dist, ll_segment)
    draw_dash(index - 2, dist, ll_segment)

def draw_dash(index, dist, ll_segment):
    ll_segment[index, dist - 1] = 255  # <-- here is the real calculated center
    ll_segment[index, dist - 3] = 255
    ll_segment[index, dist - 2] = 255
    ll_segment[index, dist - 4] = 255
    ll_segment[index, dist - 5] = 255
    ll_segment[index, dist - 6] = 255

def detect(opt):

    if not os.path.exists(opt.save_dir):
        os.makedirs(opt.save_dir)  # make new dir

    # Set Dataloader
    if opt.source.isnumeric():
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(opt.source, img_size=opt.img_size)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(opt.source, img_size=opt.img_size)
        bs = 1  # batch_size


    # Run inference
    t0 = time.time()

    save_path_bad_dir = str(opt.save_dir + "_bad")
    save_path_good_dir = str(opt.save_dir + "_good")
    save_path_unknown_dir = str(opt.save_dir + "_unknown")
    save_path_expected_bad_dir = str(opt.save_dir + "_expected_bad")
    save_path_bad_raw_dir = str(opt.save_dir + "_bad_raw")

    for filespath in [save_path_bad_dir, save_path_good_dir, save_path_unknown_dir, save_path_expected_bad_dir, save_path_bad_raw_dir]:
        if os.path.exists(filespath):
            for file_name in os.listdir(filespath):
                file_path = os.path.join(filespath, file_name)
                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                except Exception as e:
                    print(f"Error deleting file {filespath}: {e}")

    detected = 0
    not_labeled = 0
    not_labeled_not_detected = 0
    labeled = len(labels)
    all_images = 0
    for i, (path, img, img_det, vid_cap,shapes) in enumerate(dataset):
        all_images += 1

        save_path_bad = str(save_path_bad_dir + '/' + Path(path).name)
        save_path_good = str(save_path_good_dir + '/' + Path(path).name)
        save_path_unknown = str(save_path_unknown_dir + '/' + Path(path).name)
        save_path_expected_bad = str(save_path_expected_bad_dir + '/' + Path(path).name)
        save_path_bad_raw = str(save_path_bad_raw_dir + '/' + Path(path).name)

        # numpy_image = img.cpu().numpy()
       # img = np.squeeze(numpy_image, axis=0)


        height = img.shape[0]
        width = img.shape[1]

        # Calculate the new height to maintain the aspect ratio
        new_height = int((640 / width) * height)

        resized_img = Image.fromarray(img).resize((640, new_height))

        # Convert back to numpy array if needed
        # For example, if you want to return a numpy array:
        resized_img_np = np.array(resized_img)

        ll_seg_out = detect_lines(resized_img_np, detection_mode)
        (
            center_lanes,
            distance_to_center_normalized,
        ) = calculate_center_v1(ll_seg_out)
        right_lane_normalized_distances, right_center_lane = choose_lane_v1(distance_to_center_normalized, center_lanes)

        ll_segment_stacked = get_ll_seg_image(right_center_lane, ll_seg_out)
        centers = np.array(right_lane_normalized_distances)
        print(centers)

        if dataset.mode == 'images':
            # Resize detected_lines to match the dimensions of image
            detected_lines_resized = cv2.resize(ll_segment_stacked, (img.shape[1], img.shape[0]))
            # Define the transparency level (alpha) for the overlay
            alpha = 0.5  # You can adjust this value to change the transparency
            # Overlay the detected lines on top of the RGB image
            overlay = cv2.addWeighted(img, 1 - alpha, detected_lines_resized, alpha, 0)
            cv2.imshow("perception", overlay) if show_images else None
            # cv2.waitKey(10000)
            #cv2.waitKey(0)  # 1 millisecond
            # if not 1 in centers:
            #     print('"'+path+'"' + ": " + str(centers) + ",")
            #     cv2.imwrite(save_path_good + "_good", overlay)
            # TODO Ha funcionado con yolop lo de abajo
            if labels.get(path) is None:
                not_labeled_not_detected += 1
                cv2.imwrite(save_path_expected_bad, overlay)
            elif wasDetected(labels.get(path), centers.tolist()):
                detected += 1
                cv2.imwrite(save_path_good, overlay)
            elif labels.get(path) is not None:
                cv2.imwrite(save_path_bad, overlay)
                cv2.imwrite(save_path_bad_raw, img)
            elif not 1 in centers.tolist():
                not_labeled += 1
                cv2.imwrite(save_path_unknown, overlay)
            # if path not in labels:
            #     not_labeled += 1
            #     cv2.imwrite(save_path_bad, overlay)


        else:
            cv2.imshow('image', ll_seg_out)
            cv2.waitKey(1)  # 1 millisecond

    cv2.waitKey(10000) if show_images else None
    print('Results saved to %s' % Path(opt.save_dir+"xxx"))
    print('Done. (%.3fs)' % (time.time() - t0))
    print(f"expected good ->{detected} images of {labeled} labeled images were detected = {(detected/labeled) * 100}%.")
    print(f"not expected bad -> {labeled - detected} images of {labeled} labeled images were not detected = {((labeled - detected)/labeled) * 100}%.")
    print(f"total good -> {detected} images of {all_images} were detected = {(detected/all_images) * 100}%.")
    print(f"not expected good -> {not_labeled} images of {all_images - labeled} not labeled images were detected = {(not_labeled/(all_images - labeled)) * 100}%.")

    # print(f"{not_labeled} images of {all_images} were not labeled = {(not_labeled/all_images) * 100}%.")


def wasDetected(detected: list, labels: list):
    for i in range(len(detected)):
        if abs(detected[i] - labels[i]) > 0.045:
            return False
    return True

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='weights/End-to-end.pth', help='model.pth path(s)')
    parser.add_argument('--source', type=str, default='/home/ruben/Desktop/RL-Studio/rl_studio/inference/images', help='source')  # file/folder   ex:inference/images
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--save-dir', type=str, default='/home/ruben/Desktop/RL-Studio/rl_studio/inference/output', help='directory to save results')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    opt = parser.parse_args()
    with torch.no_grad():
        detect(opt)
