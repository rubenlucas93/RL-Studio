import argparse
import math
import os, sys
import shutil
import time
from pathlib import Path
from sklearn.preprocessing import PolynomialFeatures
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from statsmodels.distributions.empirical_distribution import ECDF
import pickle

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from collections import Counter

print(sys.path)
import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

from rl_studio.envs.carla.utils.DemoDataset import LoadImages, LoadStreams
normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

# detection_mode = 'yolop'
# detection_mode = 'lane_detector'
# detection_mode = 'programmatic'

show_images = False
apply_mask = True

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
images_high = 380
upper_limit = 200
x_row = [ upper_limit + 10, 270, 300, 320, images_high - 10 ]
NO_DETECTED = 0
THRESHOLDS_PERC = [0.1, 0.3, 0.5, 0.7, 0.9]
PERFECT_THRESHOLD = 0.9

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


lane_model = torch.load('/home/ruben/Desktop/RL-Studio/rl_studio/envs/carla/utils/lane_det/fastai_torch_lane_detector_model.pth')
lane_model.eval()

lane_model_v3 = torch.load('/home/ruben/Desktop/RL-Studio/rl_studio/envs/carla/utils/lane_det/best_model_torch.pth')
lane_model_v3.eval()


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


def post_process_hough_lane_det_v1(ll_segment):
    # Step 4: Perform Hough transform to detect lines
    #ll_segment = cv2.dilate(ll_segment, (3, 3), iterations=4)
    #ll_segment = cv2.erode(ll_segment, (3, 3), iterations=2)
    cv2.imshow("preprocess", ll_segment) if show_images else None
    #edges = cv2.Canny(ll_segment, 50, 100)

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

# TODO It is not feasible for online. Since it is calling predict thrice. Optimize
def post_process_hough_lane_det(ll_segment, side):
    # Step 4: Perform Hough transform to detect lines
    # ll_segment = cv2.dilate(ll_segment, (3, 3), iterations=4)
    # ll_segment = cv2.erode(ll_segment, (3, 3), iterations=4)
    cv2.imshow("preprocess" + side, ll_segment) if show_images else None
    # edges = cv2.Canny(ll_segment, 50, 100)
    # Extract coordinates of non-zero points
    nonzero_points = np.argwhere(ll_segment == 255)
    if len(nonzero_points) == 0:
        return None

    # Extract x and y coordinates
    x = nonzero_points[:, 1].reshape(-1, 1)  # Reshape for scikit-learn input
    y = nonzero_points[:, 0]

    # Fit linear regression model
    # model = LinearRegression()

    # Create a pipeline that first transforms the input features and then fits the linear regression model
    scaler = preprocessing.StandardScaler()
    degree = 3 # Quadratic regression
    model = make_pipeline(PolynomialFeatures(degree), scaler, LinearRegression())

    model.fit(x, y)

    # # Predict y values based on x
    # y_pred = model.predict(x)

    # degree = 4
    # # Fit coefficients
    # coeffs = np.polyfit(x.flatten(), y, 4)
    # # Generate polynome function f(x)
    # f = np.poly1d(coeffs)

    line_mask = np.zeros_like(ll_segment, dtype=np.uint8)  # Ensure dtype is uint8

    sp_predictions = []
    # Draw the linear regression line
    # for i in range(len(x)):
    #     cv2.circle(line_mask, (x[i][0], int(y_pred[i])), 2, (255, 0, 0), -1)
    #     sp_predictions.append(y_pred[i])
    for i in range(len(x)):
        cv2.circle(line_mask, (i, int(model.predict([[i]]))), 2, (255, 0, 0), -1)
        sp_predictions.append([i, model.predict([[i]])])
        
    cv2.imshow("result" + side, line_mask) if show_images else None

    # Find the minimum and maximum x coordinates
    min_x = np.min(x)
    max_x = np.max(x)

    # Find the corresponding predicted y-values for the minimum and maximum x coordinates
    y1 = int(model.predict([[min_x]]))
    y2 = int(model.predict([[max_x]]))

    # Define the line segment
    line_segment = (min_x, y1, max_x, y2)

    return line_segment, sp_predictions


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


# TODO All "merged" part is not needed for lane_detector
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
    for line in merged_lines:
        x1, y1, x2, y2 = line['x1'], line['y1'], line['x2'], line['y2']
        cv2.line(merged_image, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Display the merged lines
    cv2.imshow('Merged Lines', merged_image) if show_images else None

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

def detect_lane_detector_v3(raw_image):
    image_tensor = raw_image.transpose(2, 0, 1).astype('float32') / 255
    x_tensor = torch.from_numpy(image_tensor).to("cuda").unsqueeze(0)
    model_output = torch.softmax(lane_model_v3.forward(x_tensor), dim=1).cpu().numpy()
    return model_output
def lane_detection_overlay(image, left_mask, right_mask):
    res = np.copy(image)
    # We show only points with probability higher than 0.5
    res[left_mask > 0.5, :] = [255,0,0]
    res[right_mask > 0.5,:] = [0, 0, 255]
    return res

def detect_lines(raw_image, detection_mode, processing_mode):
    if detection_mode == 'programmatic':
        gray = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
        # mask_white = cv2.inRange(gray, 200, 255)
        # mask_image = cv2.bitWiseAnd(gray, mask_white)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        ll_segment = cv2.Canny(blur, 50, 100)
        cv2.imshow("raw", ll_segment) if show_images else None
        processed = post_process(ll_segment)
        if processing_mode == 'none':
            lines = processed
        else:
            lines = post_process_hough_programmatic(processed if apply_mask else ll_segment)
    elif detection_mode == 'yolop':
        with torch.no_grad():
            ll_segment = (detect_yolop(raw_image) * 255).astype(np.uint8)
        cv2.imshow("raw", ll_segment) if show_images else None
        if processing_mode == 'none':
            lines = ll_segment
        else:
            processed = post_process(ll_segment)
            lines = post_process_hough_yolop(processed if apply_mask else ll_segment)
    # TODO refactor to not duplicate code in following branches
    elif detection_mode == "lane_det_v3":
        with torch.no_grad():
            ll_segment, left_mask, right_mask = detect_lane_detector_v3(raw_image)[0]
        ll_segment = np.zeros_like(raw_image)
        ll_segment = lane_detection_overlay(ll_segment, left_mask, right_mask)
        cv2.imshow("raw", ll_segment) if show_images else None
        # Extract blue and red channels
        if processing_mode == 'none':
            lines = ll_segment
        else:
            # Display the grayscale image
            ll_segment = post_process(ll_segment)
            blue_channel = ll_segment[:, :, 0]  # Blue channel
            red_channel = ll_segment[:, :, 2]  # Red channel

            lines = []
            left_line, sp_left = post_process_hough_lane_det(blue_channel, "left")
            if left_line is not None:
                lines.append([left_line])
            right_line, sp_right = post_process_hough_lane_det(red_channel, "right")
            if right_line is not None:
                lines.append([right_line])
            ll_segment = 0.5 * blue_channel + 0.5 * red_channel
            ll_segment = cv2.convertScaleAbs(ll_segment)
    else:
        with torch.no_grad():
            ll_segment, left_mask, right_mask = detect_lane_detector(raw_image)[0]
        ll_segment = np.zeros_like(raw_image)
        ll_segment = lane_detection_overlay(ll_segment, left_mask, right_mask)
        cv2.imshow("raw", ll_segment) if show_images else None
        # Extract blue and red channels
        if processing_mode == 'none':
            lines = ll_segment
        else:
            # Display the grayscale image
            ll_segment = post_process(ll_segment)
            blue_channel = ll_segment[:, :, 0]  # Blue channel
            red_channel = ll_segment[:, :, 2]  # Red channel

            lines = []
            left_line = post_process_hough_lane_det(blue_channel)
            if left_line is not None:
                lines.append([left_line])
            right_line = post_process_hough_lane_det(red_channel)
            if right_line is not None:
                lines.append([right_line])
            ll_segment = 0.5 * blue_channel + 0.5 * red_channel
            ll_segment = cv2.convertScaleAbs(ll_segment)

    if processing_mode == 'none':
        detected_lines = ll_segment
        # detected_lines = (detected_lines // 255).astype(np.uint8)  # Keep the lower one-third of the image
    else:
        detected_lines = merge_and_extend_lines(lines, ll_segment)
        # line_mask = morphological_process(line_mask, kernel_size=15, func_type=cv2.MORPH_CLOSE)
        # line_mask = morphological_process(line_mask, kernel_size=5, func_type=cv2.MORPH_OPEN)
        # line_mask = cv2.dilate(line_mask, (15, 15), iterations=15)
        # line_mask = cv2.erode(line_mask, (5, 5), iterations=20)

        # TODO (Ruben) It is quite hardcoded and unrobust. Fix this to enable all lines and more than
        # 1 lane detection and cameras in other positions
        boundary_y = detected_lines.shape[1] * 1 // 3
        # Copy the lower part of the source image into the target image
        detected_lines[:boundary_y, :] = 0
        detected_lines = (detected_lines // 255).astype(np.uint8)  # Keep the lower one-third of the image

    return detected_lines, sp_left, sp_right

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
    for index in diff_indices:
        interested_line_borders = np.append(interested_line_borders, indices[index])
        interested_line_borders = np.append(interested_line_borders, int(indices[index+1]))

    midpoints = calculate_midpoints(interested_line_borders)
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


def calculate_lines_percent(mask):
    width = mask.shape[1]
    center_image = width / 2

    line_size = images_high - upper_limit
    line_thresholds = np.array(mask[upper_limit: images_high]) > 0.8

    # Initialize counters
    positive_count = 0
    negative_count = 0

    # Iterate through each set of indices
    for thresholds in line_thresholds:
        indices = np.where(thresholds)[0]

        # Check if any index is positive
        if any(index - center_image > 0 for index in indices):
            positive_count += 1
        # Check if any index is negative
        if any(index - center_image < 0 for index in indices):
            negative_count += 1

    left_lane_perc = negative_count / line_size
    right_lane_perc = positive_count / line_size

    return left_lane_perc, right_lane_perc

def calculate_center_v1(mask, sp_left, sp_right):
    width = mask.shape[1]
    center_image = width / 2
    # lines = [mask[x_row[i], :] for i, _ in enumerate(x_row)]
    # center_lane_indexes = [
    #     find_lane_center(lines[x]) for x, _ in enumerate(lines)
    # ]
    center_lane_indexes = []
    for _, i in enumerate(x_row):
        center_lane_indexes.append([int(sp_left[i][1])])
        # center_lane_indexes.append([width - int(sp_left[i][1] + ((sp_right[i][1] - sp_left[i][1]) / 2))])

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
        if len(inner_list) < 1 or inner_list[0] < 0 or inner_list[0]>600: # If we don't see the 2 lanes, we discard the row
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


def calculate_and_plot_lines_counts_above(save_results_benchmark_path, percs, yolop_left_perc, yolop_right_perc,
                                                lane_detector_left_perc, lane_detector_right_perc,
                                                lane_detector_v3_left_perc, lane_detector_v3_right_perc,
                                                programmatic_left_perc, programmatic_right_perc, processing_mode):

    for perc in percs:
        yolop_counts = calculate_lines_counts_above(perc, yolop_left_perc, yolop_right_perc)
        lane_det_counts = calculate_lines_counts_above(perc, lane_detector_left_perc,
                                                       lane_detector_right_perc)
        lane_det_v3_counts = calculate_lines_counts_above(perc, lane_detector_v3_left_perc,
                                                          lane_detector_v3_right_perc)
        prog_counts = calculate_lines_counts_above(perc, programmatic_left_perc, programmatic_right_perc)

        subplots = 4 if processing_mode != "none" else 3

        fig2, axs2 = plt.subplots(1, subplots, figsize=(18, 6))

        axs2[0].bar(['Just left', 'Just right', 'both', 'none'], yolop_counts, color=['blue', 'green'])
        axs2[0].set_title('YOLOP')
        axs2[0].set_ylabel(f'percentage of images above {perc * 100}% detected')
        axs2[0].set_ylim(0, 100)

        axs2[2].bar(['Just left', 'Just right', 'both', 'none'], prog_counts, color=['blue', 'green'])
        axs2[2].set_title('Programmatic')
        axs2[2].set_ylabel(f'percentage of images above {perc * 100}% detected')
        axs2[2].set_ylim(0, 100)

        axs2[1].bar(['Just left', 'Just right', 'both', 'none'], lane_det_v3_counts, color=['blue', 'green'])
        axs2[1].set_title('MobileV3Small')
        axs2[1].set_ylabel(f'percentage of images above {perc * 100}% detected')
        axs2[1].set_ylim(0, 100)

        # if processing_mode != "none":
        #     axs2[3].bar(['Just left', 'Just right', 'both', 'none'], prog_counts, color=['blue', 'green'])
        #     axs2[3].set_title('Programmatic')
        #     axs2[3].set_ylabel(f'percentage of images above {perc * 100}% detected')
        #     axs2[3].set_ylim(0, 100)

        plt.savefig(save_results_benchmark_path / f'plot_{perc * 100}_above.png')
        plt.close()
    pass


def calculate_lines_counts_above(threshold, left_perc, right_perc):
    return [
        (np.sum((np.array(left_perc) >= threshold) & (np.array(right_perc) < threshold)) / len(left_perc)) * 100,
        (np.sum((np.array(left_perc) < threshold) & (np.array(right_perc) >= threshold)) / len(left_perc)) * 100,
        (np.sum((np.array(left_perc) >= threshold) & (np.array(right_perc) >= threshold)) / len(left_perc)) * 100,
        (np.sum((np.array(left_perc) < threshold) & (np.array(right_perc) < threshold)) / len(left_perc)) * 100,
    ]



def perform_all_benchmarking(dataset, processing_mode):
    # Create the save directory if it doesn't exist
    save_results_benchmark_dir = str(opt.save_dir + "/benchmark/" + processing_mode + "/")
    save_results_benchmark_path = Path(save_results_benchmark_dir)
    save_results_benchmark_path.mkdir(parents=True, exist_ok=True)

    if os.path.exists(save_results_benchmark_dir):
        for file_name in os.listdir(save_results_benchmark_dir):
            file_path = os.path.join(save_results_benchmark_dir, file_name)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
            except Exception as e:
                print(f"Error deleting file {save_results_benchmark_dir}: {e}")

    # # Benchmarking data (these would be populated by your benchmark_one function)
    # yolop_times, yolop_errors, yolop_left_perc, yolop_right_perc, percentage_yolop = benchmark_one(dataset, "yolop",
    #                                                                                                processing_mode)
    lane_det_v3_times, lane_detector_v3_errors, lane_detector_v3_left_perc, lane_detector_v3_right_perc, percentage_v3_lane = benchmark_one(
        dataset, "lane_det_v3", processing_mode)
    # lane_det_times, lane_detector_errors, lane_detector_left_perc, lane_detector_right_perc, percentage_lane = benchmark_one(
    #     dataset, "lane_detector", processing_mode)
    #
    # programmatic_times, programmatic_errors, programmatic_left_perc, programmatic_right_perc, percentage_programmatic = benchmark_one(
    #     dataset, "programmatic", processing_mode)

    # # Plot 1: Average errors and calculation time
    # labels = ['YOLOP', 'MobileV3Small', 'Thresholding']
    # percentages = [percentage_yolop, percentage_v3_lane, percentage_programmatic]
    # colors = ['blue', 'green', 'black']
    #
    # # if processing_mode == "none":
    # #     labels.pop(3)
    # #     percentages.pop(3)
    # #     colors.pop(3)
    #
    # plt.figure(figsize=(10, 10))
    # plt.subplot(1, 1, 1)
    # plt.bar(labels, percentages, color=colors)
    # plt.title('Percentage of Detected Images')
    # plt.ylabel('Percentage of images in which images are perfectly detected')
    # plt.ylim(0, 100)
    #
    # plt.savefig(save_results_benchmark_path / 'plot0.png')
    # plt.close()
    #
    # times = [np.mean(yolop_times), np.mean(lane_det_v3_times), np.mean(programmatic_times)]
    # colors = ['blue', 'green', 'black']
    #
    # # if processing_mode == "none":
    # #     times.pop(3)
    # #     colors.pop(3)
    #
    # plt.figure(figsize=(10, 10))
    # plt.subplot(1, 1, 1)
    # plt.bar(labels, times, color=colors)
    # plt.title('Average Times')
    # plt.ylabel('Time (s)')
    #
    # plt.savefig(save_results_benchmark_path / 'plottimes.png')
    # plt.close()
    #
    # calculate_and_plot_lines_counts_above(save_results_benchmark_path, THRESHOLDS_PERC, yolop_left_perc, yolop_right_perc,
    #                                             lane_detector_left_perc, lane_detector_right_perc,
    #                                             lane_detector_v3_left_perc, lane_detector_v3_right_perc,
    #                                             programmatic_left_perc, programmatic_right_perc, processing_mode)
    #
    # # Plot 3: Average errors
    # fig3, ax3 = plt.subplots(1, 1, figsize=(10, 6))
    # averages = np.array([np.mean(yolop_errors), np.mean(lane_detector_v3_errors),
    #             np.mean(programmatic_errors)])
    # averages = averages * 320
    # colors = ['blue', 'green', 'black']
    #
    # # if processing_mode == "none":
    # #     averages = averages[:-1]
    # #     colors.pop(3)
    #
    # ax3.bar(labels, averages, color=colors)
    # ax3.set_xlabel('Perception Mode')
    # ax3.set_ylabel('Average Error in pixels')
    # ax3.set_title('Average error when both lines detected')
    # plt.savefig(save_results_benchmark_path / 'ploterrolane.png')
    # plt.close()
    #
    # # Plot 4: ECDF of errors
    # fig4, ax4 = plt.subplots(1, 1, figsize=(20, 10))
    #
    # yolop_errors = np.array(yolop_errors)
    # pix_yolop_errors = yolop_errors * 320
    # lane_detector_errors = np.array(lane_detector_errors)
    # pix_lane_detector_errors = lane_detector_errors * 320
    # lane_detector_v3_errors = np.array(lane_detector_v3_errors)
    # pix_lane_detector_v3_errors = lane_detector_v3_errors * 320
    # programmatic_errors = np.array(programmatic_errors)
    # pix_programmatic_errors = programmatic_errors * 320
    #
    # ecdf_yolop = ECDF(pix_yolop_errors)
    # ecdf_lane_detector = ECDF(pix_lane_detector_errors)
    # ecdf_lane_detector_v3 = ECDF(pix_lane_detector_v3_errors)
    # ecdf_programmatic = ECDF(pix_programmatic_errors)
    # ax4.plot(ecdf_yolop.x, ecdf_yolop.y, label='YOLOP')
    # ax4.plot(ecdf_lane_detector.x, ecdf_lane_detector.y, label='Lane Detector')
    # ax4.plot(ecdf_lane_detector_v3.x, ecdf_lane_detector_v3.y, label='Lane Detector V3')
    #
    # if processing_mode != "none":
    #     ax4.plot(ecdf_programmatic.x, ecdf_programmatic.y, label='Programmatic')
    #
    # ax4.set_xlabel('Error')
    # ax4.set_ylabel('ECDF')
    # ax4.set_title('ECDF Comparison of Errors')
    # ax4.legend()
    # ax4.grid(True)
    #
    # plt.savefig(save_results_benchmark_path / 'plotecdf.png')
    # plt.close()
    #
    # subplots = 4 if processing_mode != "none" else 3
    #
    # # Plot 2: Number of images with left and right lanes detected
    # fig5, axs5 = plt.subplots(1, subplots, figsize=(18, 6))
    # yolop_avg_lines = [
    #     (np.mean((np.array(yolop_left_perc))) * 100),
    #     (np.mean((np.array(yolop_right_perc))) * 100)
    # ]
    # lane_det_avg_lines = [
    #     (np.mean((np.array(lane_detector_left_perc))) * 100),
    #     (np.mean((np.array(lane_detector_right_perc))) * 100)
    # ]
    # lane_detector_v3_avg_lines = [
    #     (np.mean((np.array(lane_detector_v3_left_perc))) * 100),
    #     (np.mean((np.array(lane_detector_v3_right_perc))) * 100)
    # ]
    # programmatic_avg_lines = [
    #     (np.mean((np.array(programmatic_left_perc))) * 100),
    #     (np.mean((np.array(programmatic_right_perc))) * 100)
    # ]
    #
    # axs5[0].bar(['left lane', 'right lane'], yolop_avg_lines, color=['blue', 'green'])
    # axs5[0].set_title('YOLOP')
    # axs5[0].set_ylabel('lane percentage detected')
    # axs5[0].set_ylim(0, 100)
    #
    # axs5[1].bar(['left lane', 'right lane'], lane_det_avg_lines, color=['blue', 'green'])
    # axs5[1].set_title('Lane Detector')
    # axs5[1].set_ylabel('lane percentage detected')
    # axs5[1].set_ylim(0, 100)
    #
    # axs5[2].bar(['left lane', 'right lane'], lane_detector_v3_avg_lines, color=['blue', 'green'])
    # axs5[2].set_title('Lane Detector V3')
    # axs5[2].set_ylabel('lane percentage detected')
    # axs5[2].set_ylim(0, 100)
    #
    # if processing_mode != "none":
    #     axs5[3].bar(['left lane', 'right lane'], programmatic_avg_lines, color=['blue', 'green'])
    #     axs5[3].set_title('Programmatic')
    #     axs5[3].set_ylabel('lane percentage detected')
    #     axs5[3].set_ylim(0, 100)
    #
    # plt.savefig(save_results_benchmark_path / 'plotlanepercentage.png')
    # plt.close()
    #
    #
    #
    # print(f"All plots saved to {save_results_benchmark_path}")


def detect(opt):
    # Set Dataloader
    if opt.source.isnumeric():
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(opt.source, img_size=opt.img_size)
    else:
        dataset = LoadImages(opt.source, img_size=opt.img_size)

    # perform_all_benchmarking(dataset, "none")
    perform_all_benchmarking(dataset, "postprocess")

def benchmark_one(dataset, detection_mode, processing_mode):
    print("running benchmarking on " + detection_mode)
    # Run inference
    t0 = time.time()

    save_path_bad_dir = str(opt.save_dir + detection_mode + "/" + processing_mode + "/bad")
    save_path_good_dir = str(opt.save_dir + detection_mode + "/" + processing_mode +  "/good")
    save_path_bad_raw_dir = str(opt.save_dir + detection_mode + "/" + processing_mode +  "/bad_raw")
    save_path_out_raw_dir = str(opt.save_dir + detection_mode + "/" + processing_mode + "/good_raw")
    save_results_metrics_dir = str(opt.save_dir + detection_mode + "/" + processing_mode + "/metrics")

    # TODO avoid this duplicated code
    if not os.path.exists(save_path_bad_dir):
        os.makedirs(save_path_bad_dir)

    if not os.path.exists(save_path_good_dir):
        os.makedirs(save_path_good_dir)

    if not os.path.exists(save_path_bad_raw_dir):
        os.makedirs(save_path_bad_raw_dir)

    if not os.path.exists(save_path_out_raw_dir):
        os.makedirs(save_path_out_raw_dir)

    if not os.path.exists(save_results_metrics_dir):
        os.makedirs(save_results_metrics_dir)

    directories = [save_path_bad_dir, save_path_good_dir, save_path_bad_raw_dir, save_path_out_raw_dir,
                   save_results_metrics_dir]
    for directory_path in directories:
        if os.path.exists(directory_path):
            for file_name in os.listdir(directory_path):
                file_path = os.path.join(directory_path, file_name)
                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                except Exception as e:
                    print(f"Error deleting file {directory_path}: {e}")

    detected = 0
    all_images = 0
    all_avg_errors_when_detected = []
    all_left_perc = []
    all_right_perc = []
    times = []
    for i, (path, img, img_det, vid_cap, shapes) in enumerate(dataset):
        all_images += 1

        save_path_bad = str(save_path_bad_dir + '/' + Path(path).name)
        save_path_good = str(save_path_good_dir + '/' + Path(path).name)
        save_path_bad_raw = str(save_path_bad_raw_dir + '/' + Path(path).name)
        save_path_out_raw = str(save_path_out_raw_dir + '/' + Path(path).name)

        height = img.shape[0]
        width = img.shape[1]

        # Calculate the new height to maintain the aspect ratio
        new_height = int((640 / width) * height)

        resized_img = Image.fromarray(img).resize((640, new_height))

        # Convert back to numpy array if needed
        # For example, if you want to return a numpy array:
        resized_img_np = np.array(resized_img)
        start = time.time()

        ll_seg_out, sp_left, sp_right = detect_lines(resized_img_np, detection_mode, processing_mode)

        processing_time = time.time() - start
        times.append(processing_time)

        ll_seg_out_raw = ll_seg_out
        # TODO (Ruben) use this raw output to measure percentage of line detected

        (
            center_lanes,
            distance_to_center_normalized,
        ) = calculate_center_v1(ll_seg_out, sp_left, sp_right)
        right_lane_normalized_distances, right_center_lane = choose_lane_v1(distance_to_center_normalized, center_lanes)

        ll_segment_stacked = get_ll_seg_image(right_center_lane, ll_seg_out)
        centers = np.array(right_lane_normalized_distances)
        # print(centers)

        left_percent, right_percent = calculate_lines_percent(ll_seg_out)
        all_left_perc.append(left_percent)
        all_right_perc.append(right_percent)

        if dataset.mode == 'images':
            # Resize detected_lines to match the dimensions of image
            if processing_mode != "none":
                detected_lines_resized = cv2.resize(ll_segment_stacked, (img.shape[1], img.shape[0]))
                # Define the transparency level (alpha) for the overlay
                alpha = 0.5  # You can adjust this value to change the transparency
                # Overlay the detected lines on top of the RGB image
                overlay = cv2.addWeighted(img, 1 - alpha, detected_lines_resized, alpha, 0)
                cv2.imshow("perception", overlay) if show_images else None

            total_error = 0
            detected_points = 0
            for x in right_lane_normalized_distances:
                if abs(x) != 1:
                    detected_points += 1
                    total_error += abs(x)

            if detected_points > 0:
                average_abs = total_error / detected_points
                all_avg_errors_when_detected.append(average_abs)

            if processing_mode == "none":
                if left_percent > PERFECT_THRESHOLD and right_percent > PERFECT_THRESHOLD:
                    detected += 1
                    cv2.imwrite(save_path_good, ll_seg_out_raw)
                    cv2.imwrite(save_path_out_raw, img)
                else:
                    cv2.imwrite(save_path_bad, ll_seg_out_raw)
                    cv2.imwrite(save_path_bad_raw, img)
            else:
                if wasDetected(centers.tolist()):
                    detected += 1
                    cv2.imwrite(save_path_good, overlay)
                    cv2.imwrite(save_path_out_raw, img)
                else:
                    cv2.imwrite(save_path_bad, overlay)
                    cv2.imwrite(save_path_bad_raw, img)

        else:
            cv2.imshow('image', ll_seg_out)
            cv2.waitKey(1)  # 1 millisecond

        cv2.waitKey(0) if show_images else None

    cv2.waitKey(10000) if show_images else None
    print('Done. (%.3fs)' % (time.time() - t0))
    print(f"total good -> {detected} images of {all_images} were detected = {(detected/all_images) * 100}%.")

    file_name = processing_mode + "_errors.pkl"
    save_results_metrics_errors_file = save_results_metrics_dir + "/" + file_name
    with open(save_results_metrics_errors_file, 'wb') as file:
        pickle.dump(all_avg_errors_when_detected, file)

    file_name = processing_mode + "_left.pkl"
    save_results_metrics_left_file = save_results_metrics_dir + "/" + file_name
    with open(save_results_metrics_left_file, 'wb') as file:
        pickle.dump(all_left_perc, file)

    file_name = processing_mode + "_right.pkl"
    save_results_metrics_right_file = save_results_metrics_dir + "/" + file_name
    with open(save_results_metrics_right_file, 'wb') as file:
        pickle.dump(all_right_perc, file)

    return times, all_avg_errors_when_detected, all_left_perc, all_right_perc, (detected/all_images) * 100


def wasDetected(labels: list):
    for i in range(len(labels)):
        if abs(labels[i]) > 0.1:
            return False
    return True

def anyDetected(labels: list):
    for i in range(len(labels)):
        if abs(labels[i]) < 0.8:
            return True
    return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='weights/End-to-end.pth', help='model.pth path(s)')
    parser.add_argument('--source', type=str, default='/home/ruben/Desktop/RL-Studio/rl_studio/inference/images', help='source')  # file/folder   ex:inference/images
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--save-dir', type=str, default='/home/ruben/Desktop/RL-Studio/rl_studio/inference/output/', help='directory to save results')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    opt = parser.parse_args()
    with torch.no_grad():
        detect(opt)
