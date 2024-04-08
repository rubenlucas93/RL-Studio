import cv2
import numpy as np
import torch
import os


class LaneDetector:
    def __init__(self, model_path: str):
        torch.cuda.empty_cache()
        self.__model: torch.nn.Module = torch.load(model_path)
        self.__model.eval()

        self.__threshold = 0.05
        # self.__threshold = 0.4
        self.__num_points = 0
        # self.__num_points = 50

    def detect(self, img_array: np.array) -> tuple:
        with torch.no_grad():
            image_tensor = img_array.transpose(2, 0, 1).astype('float32') / 255
            x_tensor = torch.from_numpy(image_tensor).to("cuda").unsqueeze(0)
            back, left, right = torch.softmax(self.__model.forward(x_tensor), dim=1).cpu().numpy()[0]

        res, left_mask, right_mask = self.__lane_detection_overlay(img_array, left, right)

        return res, left_mask, right_mask

    def __lane_detection_overlay(self, image: np.ndarray,
                                 left_mask: np.ndarray, right_mask: np.ndarray) -> tuple:
        """
        @type left_mask: object
        """
        res = np.copy(image)

        if ((left_mask[left_mask > self.__threshold].shape[0] >= self.__num_points)
                and (right_mask[right_mask > self.__threshold].shape[0] >= self.__num_points)):
            left_mask = self.__image_polyfit(left_mask)
            right_mask = self.__image_polyfit(right_mask)

            # We show only points with probability higher than 0.1
            res[left_mask > self.__threshold, :] = [255, 0, 0]
            res[right_mask > self.__threshold, :] = [0, 0, 255]
        else:
            left_mask = np.zeros_like(left_mask)
            right_mask = np.zeros_like(right_mask)

        return res, left_mask, right_mask

    def __image_polyfit(self, image: np.ndarray) -> np.ndarray:
        img = np.copy(image)
        img[image > self.__threshold] = 255

        indices = np.where(img == 255)

        grade = 1
        coefficients = np.polyfit(indices[0], indices[1], grade)

        x = np.linspace(0, img.shape[1], num=2500)
        y = np.polyval(coefficients, x)
        points = np.column_stack((x, y)).astype(int)

        valid_points = []

        for point in points:
            if (0 < point[1] < 1023) and (0 < point[0] < 509):
                valid_points.append(point)

        valid_points = np.array(valid_points)
        polyfitted = np.zeros_like(img)
        polyfitted[tuple(valid_points.T)] = 255

        return polyfitted


