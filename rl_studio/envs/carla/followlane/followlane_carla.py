import os
import weakref
from collections import Counter
import math
import time
import carla
import random
import cv2
import torch
from numpy import random
import numpy as np
from rl_studio.envs.carla.followlane.followlane_env import FollowLaneEnv
from rl_studio.envs.carla.followlane.settings import FollowLaneCarlaConfig
from rl_studio.envs.carla.followlane.utils import AutoCarlaUtils
from PIL import Image
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression


from rl_studio.envs.carla.utils.bounding_boxes import BasicSynchronousClient
from rl_studio.envs.carla.utils.manual_control import CameraManager
from rl_studio.envs.carla.utils.visualize_multiple_sensors import (
    DisplayManager,
    SensorManager,
    CustomTimer,
)
import pygame

NO_DETECTED = 0


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


def draw_dash(index, dist, ll_segment):
    ll_segment[index, dist - 1] = 255  # <-- here is the real calculated center
    ll_segment[index, dist - 3] = 255
    ll_segment[index, dist - 2] = 255
    ll_segment[index, dist - 4] = 255
    ll_segment[index, dist - 5] = 255
    ll_segment[index, dist - 6] = 255

def calculate_midpoints(input_array):
    midpoints = []
    for i in range(0, len(input_array) - 1, 2):
        midpoint = (input_array[i] + input_array[i + 1]) // 2
        midpoints.append(midpoint)
    return midpoints


def add_midpoints(ll_segment, index, dist):
    # Set the value at the specified index and distance to 1
    draw_dash(index, dist, ll_segment)
    draw_dash(index + 2, dist, ll_segment)
    draw_dash(index + 1, dist, ll_segment)
    draw_dash(index - 1, dist, ll_segment)
    draw_dash(index - 2, dist, ll_segment)


def connect_dashed_lines(ll_seg_mask):
    # TODO
    return ll_seg_mask

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


def choose_lane(distance_to_center_normalized, center_points):
    close_lane_indexes = [min(enumerate(inner_array), key=lambda x: abs(x[1]))[0] for inner_array in
                          distance_to_center_normalized]
    distances = [array[index] for array, index in zip(distance_to_center_normalized, close_lane_indexes)]
    centers = [array[index] for array, index in zip(center_points, close_lane_indexes)]
    return distances, centers


def wasDetected(center_lanes):
    for i in range(len(center_lanes)):
        if abs(center_lanes[i]) > 0.1:
            return False
    return True


def getTransformFromPoints(points):
    return carla.Transform(
            carla.Location(
                x=points[0],
                y=points[1],
                z=points[2],
            ),
            carla.Rotation(
                pitch=points[3],
                yaw=points[4],
                roll=points[5],
            ),
        )


class FollowLaneStaticWeatherNoTraffic(FollowLaneEnv):
    def __init__(self, **config):

        self.show_images = False
        self.show_all_points = False
        self.debug_waypoints = config.get("debug_waypoints")

        ###### init class variables
        FollowLaneCarlaConfig.__init__(self, **config)
        self.sync_mode = config["sync"]
        self.reset_threshold = config["reset_threshold"] if self.sync_mode else 1
        self.spawn_points = config.get("spawn_points")
        self.detection_mode = config.get("detection_mode")
        if self.detection_mode == 'yolop':
            from rl_studio.envs.carla.utils.yolop.YOLOP import get_net
            import torchvision.transforms as transforms
            normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )

            self.transform = transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])

            self.steps_stopped = 0
            # INIT YOLOP
            self.yolop_model = get_net()
            self.device = select_device()
            checkpoint = torch.load("envs/carla/utils/yolop/weights/End-to-end.pth",
                                    map_location=self.device)
            self.yolop_model.load_state_dict(checkpoint['state_dict'])
            self.yolop_model = self.yolop_model.to(self.device)
        elif self.detection_mode == "lane_detector_v2":
            self.lane_model = torch.load('envs/carla/utils/lane_det/fastai_torch_lane_detector_model.pth')
            self.lane_model.eval()
        elif self.detection_mode == "lane_detector":
            self.lane_model = torch.load(
                '/home/ruben/Desktop/RL-Studio/rl_studio/envs/carla/utils/lane_det/best_model_torch.pth')
            self.lane_model.eval()
        # self.display_manager = None
        # self.vehicle = None
        # self.actor_list = []
        self.timer = CustomTimer()
        self.step_count = 0

        self.client = carla.Client(
            config["carla_server"],
            config["carla_client"],
        )
        self.client.set_timeout(10.0)
        print(f"\n maps in carla 0.9.13: {self.client.get_available_maps()}\n")

        self.world = self.client.load_world(config["town"])
        self.original_settings = self.world.get_settings()
        self.traffic_manager = self.client.get_trafficmanager(config["manager_port"])
        settings = self.world.get_settings()
        self.forced_freq = config.get("async_forced_delta_seconds")
        if self.sync_mode:
            settings.max_substep_delta_time = 0.02
            settings.fixed_delta_seconds = config.get("fixed_delta_seconds")
            settings.synchronous_mode = True
            self.traffic_manager.set_synchronous_mode(True)
        else:
            self.traffic_manager.set_synchronous_mode(False)
        self.world.apply_settings(settings)
        current_settings = self.world.get_settings()
        print(f"Current World Settings: {current_settings}")
        # self.camera = None
        # self.vehicle = None
        # self.display = None
        # self.image = None

        ## -- display manager
        self.display_manager = DisplayManager(
            grid_size=[2, 3],
            window_size=[1500, 800],
        )

        self.car = None

        self.perfect_distance_pixels = None
        self.perfect_distance_normalized = None


    def setup_car_fix_pose(self, init):
        car_bp = self.world.get_blueprint_library().filter("vehicle.*")[0]
        location = carla.Transform(
            carla.Location(
                x=init.transform.location.x,
                y=init.transform.location.y,
                z=init.transform.location.z,
            ),
            carla.Rotation(
                pitch=init.transform.rotation.pitch,
                yaw=init.transform.rotation.yaw,
                roll=init.transform.rotation.roll,
            ),
        )

        self.car = self.world.spawn_actor(car_bp, location)
        while self.car is None:
            self.car = self.world.spawn_actor(car_bp, location)

        self.actor_list.append(self.car)
        spectator = self.world.get_spectator()
        spectator_location = carla.Transform(
            location.location + carla.Location(z=100),
            carla.Rotation(-90, location.rotation.yaw, 0))
        spectator.set_transform(spectator_location)

        time.sleep(1)

    def reset(self):

        self.steps_stopped = 0
        self.collision_hist = []
        self.actor_list = []
        self.previous_time = 0
        self.set_init_pose()
        if self.sync_mode:
            self.world.tick()
        else:
            self.world.wait_for_tick()

        time.sleep(1)
        self.episode_start = time.time()
        self.car.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))

        # AutoCarlaUtils.show_image("image", self.front_camera_1_5.front_camera, 1)
        # AutoCarlaUtils.show_image("bird_view", self.birds_eye_camera.front_camera, 1)

        raw_image = self.get_resized_image(self.front_camera_1_5.front_camera)

        ll_segment_post_process = self.detect_lines(raw_image)
        (
            center_lanes,
            distance_to_center_normalized,
        ) = self.calculate_center(ll_segment_post_process)
        right_lane_normalized_distances, right_center_lane = choose_lane(distance_to_center_normalized, center_lanes)

        self.show_ll_seg_image(right_center_lane, ll_segment_post_process) if self.sync_mode and self.show_images else None

        state_size = len(distance_to_center_normalized)
        # right_lane_normalized_distances = [1,1,1,1,1,1,1,1,1,1]
        # state_size = 12
        time.sleep(1)
        right_lane_normalized_distances.append(0)
        right_lane_normalized_distances.append(0)

        return np.array(right_lane_normalized_distances), state_size

    ####################################################
    ####################################################

    def find_lane_center(self, mask):
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

    def calculate_center(self, mask):
        width = mask.shape[1]
        center_image = width / 2
        ## get total lines in every line point
        lines = [mask[self.x_row[i], :] for i, _ in enumerate(self.x_row)]
        # ## As we drive in the right lane, we get from right to left
        # lines_inversed = [list(reversed(lines[x])) for x, _ in enumerate(lines)]
        ## get the distance from the right lane to center
        center_lane_indexes = [
            self.find_lane_center(lines[x]) for x, _ in enumerate(lines)
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

    def calculate_states(self, mask):
        width = mask.shape[1]
        center_image = width / 2
        ## get total lines in every line point
        lines = [mask[self.x_row[i], :] for i, _ in enumerate(self.x_row)]
        ## As we drive in right lane, we get from right to left
        lines_inversed = [list(reversed(lines[x])) for x, _ in enumerate(lines)]
        ## get the distance from right lane to center
        inv_index_right = [
            np.argmax(lines_inversed[x]) for x, _ in enumerate(lines_inversed)
        ]
        index_right = [
            width - inv_index_right[x] for x, _ in enumerate(inv_index_right)
        ]
        distance_to_center = [
            width - inv_index_right[x] - center_image
            for x, _ in enumerate(inv_index_right)
        ]
        ## normalized distances
        distance_to_center_normalized = [
            abs(float((center_image - index_right[i]) / center_image))
            for i, _ in enumerate(index_right)
        ]
        # pixels_in_state = mask.shape[1] / self.num_regions
        # states = [int(value / pixels_in_state) for _, value in enumerate(index_right)]
        states = distance_to_center_normalized

        return states, distance_to_center, distance_to_center_normalized

    def preprocess_image(self, red_mask):
        ## first, we cut the upper image
        img_sliced = self.slice_image(red_mask)
        ## -- convert to GRAY
        gray_mask = cv2.cvtColor(img_sliced, cv2.COLOR_BGR2GRAY)
        ## --  aplicamos mascara para convertir a BLANCOS Y NEGROS
        _, white_mask = cv2.threshold(gray_mask, 10, 255, cv2.THRESH_BINARY)

        return white_mask

    def draw_waypoints(self, spawn_points, init, target, lane_id, life_time):
        filtered_waypoints = []
        # i = init
        # for waypoint in spawn_points[init + 1: target + 2]:
        i = 0
        for waypoint in spawn_points:
            filtered_waypoints.append(waypoint)
            string = f"[{waypoint.road_id},{waypoint.lane_id},{i}]"
            # if waypoint.lane_id == lane_id:
            if i != target:
                self.world.debug.draw_string(
                    waypoint.transform.location,
                    f"X - {string}",
                    draw_shadow=False,
                    color=carla.Color(r=0, g=255, b=0),
                    life_time=life_time,
                    persistent_lines=True,
                )
            else:
                self.world.debug.draw_string(
                    waypoint.transform.location,
                    f"X - {string}",
                    draw_shadow=False,
                    color=carla.Color(r=255, g=0, b=0),
                    life_time=life_time,
                    persistent_lines=True,
                )
            i += 1

        return filtered_waypoints

    def get_target_waypoint(self, target_waypoint, life_time):
        """
        draw target point
        """
        self.world.debug.draw_string(
            target_waypoint.transform.location,
            "O",
            draw_shadow=False,
            color=carla.Color(r=255, g=0, b=0),
            life_time=life_time,
            persistent_lines=True,
        )

    def setup_car_random_pose(self):
        car_bp = self.world.get_blueprint_library().filter("vehicle.*")[0]
        if self.spawn_points is not None:
            spawn_point_index = random.randint(0, len(self.spawn_points)-1)
            spawn_point = self.spawn_points[spawn_point_index]
            if random.random() > 0.5:
                location = getTransformFromPoints(spawn_point)
            else:
                location = random.choice(self.world.get_map().get_spawn_points())
            self.car = self.world.spawn_actor(car_bp, location)
            while self.car is None:
                self.car = self.world.spawn_actor(car_bp, location)
        else:
            location = random.choice(self.world.get_map().get_spawn_points())
            self.car = self.world.try_spawn_actor(car_bp, location)
            while self.car is None:
                self.car = self.world.try_spawn_actor(car_bp, location)
        self.actor_list.append(self.car)
        # initial_velocity = carla.Vector3D(x=random.randint(0, 15), y=0, z=0)  # 5 m/s in the x direction
        # self.car.set_target_velocity(initial_velocity)
        time.sleep(1)

    def setup_col_sensor(self):
        colsensor = self.world.get_blueprint_library().find("sensor.other.collision")
        transform = carla.Transform(carla.Location(x=2.5, z=0.7))
        self.colsensor = self.world.spawn_actor(
            colsensor, transform, attach_to=self.car
        )
        self.actor_list.append(self.colsensor)
        self.colsensor.listen(lambda event: self.collision_data(event))

    def collision_data(self, event):
        self.collision_hist.append(event)

    def destroy_all_actors(self):
        for actor in self.actor_list[::-1]:
            # for actor in self.actor_list:
            actor.destroy()
        # print(f"\nin self.destroy_all_actors(), actor : {actor}\n")

        # self.actor_list = []
        # .client.apply_batch(
        #    [carla.command.DestroyActor(x) for x in self.actor_list[::-1]]
        # )

    ################################################################################
    def step(self, action):
        # print(f"=============== STEP ===================")

        # action = [0, 0]
        ### -------- send action
        params = self.control(action)

        now = time.time()
        elapsed_time = now - self.previous_time

        if self.sync_mode:
            self.world.tick()
        else:
            if elapsed_time < self.forced_freq:
                wait_time = self.forced_freq - elapsed_time
                time.sleep(wait_time)
            self.world.wait_for_tick()

        now = time.time()
        params["fps"] = 1 / (now - self.previous_time)
        self.previous_time = now

        ## -- states
        # mask = self.preprocess_image(
        #    self.front_camera_1_5_red_mask.front_camera_red_mask
        # )
        raw_image = self.get_resized_image(self.front_camera_1_5.front_camera) # TODO Think it is not aligned with BM

        ll_segment = self.detect_lines(raw_image)
        (
            center_lanes,
            distance_to_center_normalized,
        ) = self.calculate_center(ll_segment)
        # We get the first of all calculated "center lanes" assuming it will be the right lane
        #right_lane_normalized_distances = [inner_array[-1] for inner_array in distance_to_center_normalized]
        #right_center_lane = [[inner_array[-1]] for inner_array in center_lanes]
        right_lane_normalized_distances, right_center_lane = choose_lane(distance_to_center_normalized, center_lanes)

        self.show_ll_seg_image(right_center_lane, ll_segment)

        if self.debug_waypoints:
            average_abs = sum(abs(x) for x in right_lane_normalized_distances) / len(distance_to_center_normalized)
            if average_abs > 0.8:
                color = carla.Color(r=255, g=0, b=0)
            else:
                green_value = max(int((1 - average_abs * 2 ) * 255), 0 )
                color = carla.Color(r=0, g=green_value, b=0)

            self.world.debug.draw_string(
                self.car.get_transform().location,
                "X",
                draw_shadow=False,
                color=color,
                life_time=10000000,
                persistent_lines=True,
            )
            if self.step_count % 100 == 0:
                print(self.car.get_transform())

        # self.show_ll_seg_image(center_lanes, ll_segment_post_process, name="ll_seg_all")

        # print(f"states:{states}\n")
        # AutoCarlaUtils.show_image_with_centrals(
        #    "image",
        #    self.front_camera_1_5.front_camera[ll_segment.shape[0] :],
        #    1,
        #    distance_center,
        #    distance_to_center_normalized,
        #    self.x_row,
        # )

        ## ------ calculate distance error and states
        # print(f"{self.perfect_distance_normalized =}"
        # right_lane_normalized goes between 1 and -1
        distance_error = [abs(x) for x in right_lane_normalized_distances]
        ## -------- Rewards
        reward, done, crash = self.rewards_easy(distance_error, params)
        self.step_count += 1
        if done:
            self.step_count == 0

        params["bad_perception"], _ = self.has_bad_perception(right_lane_normalized_distances, threshold=0.999)
        params["crash"] = crash

        right_lane_normalized_distances.append(params["velocity"]/5)
        right_lane_normalized_distances.append(params["steering_angle"])

        return np.array(right_lane_normalized_distances), reward, done, params

    def control(self, action):

        # if action[0] >= 0:
        #     self.car.apply_control(carla.VehicleControl(throttle=float(action[0]), steer=float(action[1])))
        # else:
        #     self.car.apply_control(carla.VehicleControl(throttle=0.0, brake=float(abs(action[0])), steer=float(action[1])))

        v = self.car.get_velocity()

        brake = 0.0
        if action[2] > 0.5 and v.x ** 2 + v.y ** 2 + v.z ** 2 > self.punish_ineffective_vel:
            brake = 0.2
        self.car.apply_control(carla.VehicleControl(throttle=float(action[0]), brake=brake, steer=float(action[1])))

        params = {}

        v = self.car.get_velocity()
        params["velocity"] = math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)

        w = self.car.get_angular_velocity()
        params["angular_velocity"] = w

        w_angle = self.car.get_control().steer
        params["steering_angle"] = w_angle

        return params

    def rewards_followlane_dist_v_angle(self, error, params):
        # rewards = []
        # for i,_ in enumerate(error):
        #    if (error[i] < 0.2):
        #        rewards.append(10)
        #    elif (0.2 <= error[i] < 0.4):
        #        rewards.append(2)
        #    elif (0.4 <= error[i] < 0.9):
        #        rewards.append(1)
        #    else:
        #        rewards.append(0)
        rewards = [0.1 / error[i] for i, _ in enumerate(error)]
        function_reward = sum(rewards) / len(rewards)
        function_reward += math.log10(params["velocity"])
        function_reward -= 1 / (math.exp(params["steering_angle"]))

        return function_reward

    def rewards_easy(self, distance_error, params):
        done = self.end_if_conditions(distance_error, threshold=self.reset_threshold,
                                                       min_conf_states=len(distance_error)//2)
        params["d_reward"] = 0
        params["v_reward"] = 0
        params["v_eff_reward"] = 0
        params["reward"] = 0
        if done:
            crash = True
            return -2, done, crash

        crash = False
        done, states_above_threshold = self.has_bad_perception(distance_error, self.reset_threshold, len(distance_error)//2)

        if done:
            return 0, done, crash

        if params["velocity"] < self.punish_ineffective_vel:
            self.steps_stopped += 1
            if self.steps_stopped > 100:
                done = True
            return 0, done, crash

        self.steps_stopped = 0
        d_rewards = []
        for _, error in enumerate(distance_error):
            # d_rewards.append(1 - error)
            d_rewards.append(math.pow(max(0.6 - error, 0)/0.6, 3))

        # TODO ignore non detected centers
        d_reward = sum(d_rewards) / len(d_rewards)
        # d_reward = math.pow(d_reward, 9)
        params["d_reward"] = d_reward

        # reward Max = 1 here
        punish = 0
        punish += self.punish_zig_zag_value * abs(params["steering_angle"])

        v_reward = params["velocity"]/20
        v_eff_reward = v_reward * math.pow(d_reward, 5)
        params["v_reward"] = v_reward
        params["v_eff_reward"] = v_eff_reward

        beta = self.beta
        # TODO Ver que valores toma la velocity para compensarlo mejor
        function_reward = beta * d_reward + (1-beta) * v_eff_reward
        if function_reward > punish: # to avoid negative rewards
            function_reward -= punish
        else:
            function_reward = 0
        params["reward"] = function_reward

        return function_reward, done, crash

    def rewards_followlane_center_v_w(self):
        """esta sin terminar"""
        center = 0
        done = False
        if 0.65 >= center > 0.25:
            reward = 10
        elif (0.9 > center > 0.65) or (0.25 >= center > 0):
            reward = 2
        elif 0 >= center > -0.9:
            reward = 1
        else:
            reward = -100
            done = True

        return reward, done

    def slice_image(self, red_mask):
        height = red_mask.shape[0]
        image_middle_line = (height) // 2
        img_sliced = red_mask[image_middle_line:]
        return img_sliced.copy()

    def get_resized_image(self, sensor_data, new_width=640):
        # Assuming sensor_data is the image obtained from the sensor
        # Convert sensor_data to a numpy array or PIL Image if needed
        # For example, if sensor_data is a numpy array:
        # sensor_data = Image.fromarray(sensor_data)
        sensor_data = np.array(sensor_data, copy=True)

        # Get the current width and height
        height = sensor_data.shape[0]
        width = sensor_data.shape[1]

        # Calculate the new height to maintain the aspect ratio
        new_height = int((new_width / width) * height)

        resized_img = Image.fromarray(sensor_data).resize((new_width, new_height))

        # Convert back to numpy array if needed
        # For example, if you want to return a numpy array:
        resized_img_np = np.array(resized_img)

        return resized_img_np

    def detect_lines(self, raw_image):
        if self.detection_mode == 'programmatic':
            gray = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
            # mask_white = cv2.inRange(gray, 200, 255)
            # mask_image = cv2.bitWiseAnd(gray, mask_white)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            ll_segment = cv2.Canny(blur, 50, 100)
            cv2.imshow("raw", ll_segment) if self.sync_mode and self.show_images else None
            processed = self.post_process(ll_segment)
            lines = self.post_process_hough_programmatic(processed)
        elif self.detection_mode == 'yolop':
            with torch.no_grad():
                ll_segment = (self.detect_yolop(raw_image) * 255).astype(np.uint8)
            cv2.imshow("raw", ll_segment) if self.sync_mode and self.show_images else None
            # processed = self.post_process(ll_segment)
            lines = self.post_process_hough_yolop(ll_segment)
        else:
            with torch.no_grad():
                ll_segment, left_mask, right_mask = self.detect_lane_detector(raw_image)[0]
            ll_segment = np.zeros_like(raw_image)
            ll_segment = self.lane_detection_overlay(ll_segment, left_mask, right_mask)
            cv2.imshow("raw", ll_segment) if self.sync_mode and self.show_images else None
            # Extract blue and red channels
            blue_channel = ll_segment[:, :, 0]  # Blue channel
            red_channel = ll_segment[:, :, 2]  # Red channel

            lines = []
            left_line = self.post_process_hough_lane_det(blue_channel)
            if left_line is not None:
                lines.append([left_line])
            right_line = self.post_process_hough_lane_det(red_channel)
            if right_line is not None:
                lines.append([right_line])
            ll_segment = 0.5 * blue_channel + 0.5 * red_channel
            ll_segment = cv2.convertScaleAbs(ll_segment)
        detected_lines = self.merge_and_extend_lines(lines, ll_segment)

        # line_mask = morphological_process(line_mask, kernel_size=15, func_type=cv2.MORPH_CLOSE)
        # line_mask = morphological_process(line_mask, kernel_size=5, func_type=cv2.MORPH_OPEN)
        # line_mask = cv2.dilate(line_mask, (15, 15), iterations=15)
        # line_mask = cv2.erode(line_mask, (5, 5), iterations=20)

        # TODO (Ruben) It is quite hardcoded and unrobust. Fix this to enable all lines and more than
        # 1 lane detection and cameras in other positions
        boundary_y = ll_segment.shape[1] * 2 // 5
        # Copy the lower part of the source image into the target image
        ll_segment[boundary_y:, :] = detected_lines[boundary_y:, :]
        ll_segment = (ll_segment // 255).astype(np.uint8) # Keep the lower one-third of the image

        return ll_segment

    def detect_yolop(self, raw_image):
        # Get names and colors
        names = self.yolop_model.module.names if hasattr(self.yolop_model, 'module') else self.yolop_model.names

        # Run inference
        img = self.transform(raw_image).to(self.device)
        img = img.float()  # uint8 to fp16/32
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        # Inference
        det_out, da_seg_out, ll_seg_out = self.yolop_model(img)

        ll_seg_mask = torch.nn.functional.interpolate(ll_seg_out, scale_factor=int(1), mode='bicubic')
        _, ll_seg_mask = torch.max(ll_seg_mask, 1)
        ll_seg_mask = ll_seg_mask.int().squeeze().cpu().numpy()

        return ll_seg_mask

    def show_ll_seg_image(self,dists, ll_segment, suffix="",  name='ll_seg'):
        ll_segment_int8 = (ll_segment * 255).astype(np.uint8)
        ll_segment_all = [np.copy(ll_segment_int8),np.copy(ll_segment_int8),np.copy(ll_segment_int8)]

        # draw the midpoint used as right center lane
        for index, dist in zip(self.x_row, dists):
            # Set the value at the specified index and distance to 1
            add_midpoints(ll_segment_all[0], index, dist)

        # draw a line for the selected perception points
        for index in self.x_row:
            for i in range(630):
                ll_segment_all[0][index][i] = 255

        ll_segment_stacked = np.stack(ll_segment_all, axis=-1)
        # We now show the segmentation and center lane postprocessing
        cv2.imshow(name + suffix, ll_segment_stacked)
        cv2.waitKey(1)  # 1 millisecond

    def post_process(self, ll_segment):
        ''''
        Lane line post-processing
        '''
        #ll_segment = morphological_process(ll_segment, kernel_size=5, func_type=cv2.MORPH_OPEN)
        #ll_segment = morphological_process(ll_segment, kernel_size=20, func_type=cv2.MORPH_CLOSE)
        #return ll_segment
        # ll_segment = morphological_process(ll_segment, kernel_size=4, func_type=cv2.MORPH_OPEN)
        # ll_segment = morphological_process(ll_segment, kernel_size=8, func_type=cv2.MORPH_CLOSE)

        # Step 1: Create a binary mask image representing the trapeze
        mask = np.zeros_like(ll_segment)
        # pts = np.array([[300, 250], [-500, 600], [800, 600], [450, 260]], np.int32)
        pts = np.array([[280, 200], [-50, 400], [630, 400], [440, 200]], np.int32)
        cv2.fillPoly(mask, [pts], (255, 255, 255))  # Fill trapeze region with white (255)
        cv2.imshow("applied_mask", mask) if self.sync_mode and self.show_images else None

        # Step 2: Apply the mask to the original image
        ll_segment_masked = cv2.bitwise_and(ll_segment, mask)
        ll_segment_excluding_mask = cv2.bitwise_not(mask)
        # Apply the exclusion mask to ll_segment
        ll_segment_excluded = cv2.bitwise_and(ll_segment, ll_segment_excluding_mask)
        cv2.imshow("discarded", ll_segment_excluded) if self.sync_mode and self.show_images else None

        return ll_segment_masked


    def post_process_hough_lane_det(self, ll_segment):
        # ll_segment = cv2.dilate(ll_segment, (3, 3), iterations=4)
        # ll_segment = cv2.erode(ll_segment, (3, 3), iterations=2)
        cv2.imshow("preprocess", ll_segment) if self.show_images else None
        # edges = cv2.Canny(ll_segment, 50, 100)
        # Extract coordinates of non-zero points
        nonzero_points = np.argwhere(ll_segment == 255)
        if len(nonzero_points) == 0:
            return None

        # Extract x and y coordinates
        x = nonzero_points[:, 1].reshape(-1, 1)  # Reshape for scikit-learn input
        y = nonzero_points[:, 0]

        # Fit linear regression model
        model = LinearRegression()
        model.fit(x, y)

        # Predict y values based on x
        y_pred = model.predict(x)

        line_mask = np.zeros_like(ll_segment, dtype=np.uint8)  # Ensure dtype is uint8

        # Draw the linear regression line
        for i in range(len(x)):
            cv2.circle(line_mask, (x[i][0], int(y_pred[i])), 2, (255, 0, 0), -1)

        cv2.imshow("result", line_mask) if self.show_images else None

        # Find the minimum and maximum x coordinates
        min_x = np.min(x)
        max_x = np.max(x)

        # Find the corresponding predicted y-values for the minimum and maximum x coordinates
        y1 = int(model.predict([[min_x]]))
        y2 = int(model.predict([[max_x]]))

        # Define the line segment
        line_segment = (min_x, y1, max_x, y2)

        return line_segment


    def post_process_hough_yolop(self, ll_segment):
        # Step 4: Perform Hough transform to detect lines
        ll_segment = cv2.dilate(ll_segment, (3, 3), iterations=4)
        ll_segment = cv2.erode(ll_segment, (3, 3), iterations=2)
        cv2.imshow("preprocess", ll_segment) if self.sync_mode and self.show_images else None
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

        edges = cv2.Canny(line_mask, 50, 100)

        cv2.imshow("intermediate_hough", edges) if self.sync_mode and self.show_images else None

        # Reapply HoughLines on the dilated image
        lines = cv2.HoughLinesP(
            edges,  # Input edge image
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
        cv2.imshow("hough", line_mask) if self.sync_mode and self.show_images else None

        return lines


    def post_process_hough_programmatic(self, ll_segment):
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

        cv2.imshow("intermediate_hough", edges) if self.sync_mode and self.show_images else None

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
        cv2.imshow("hough", line_mask) if self.sync_mode and self.show_images else None

        return lines

    def extend_lines(self, lines, image_height):
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

    def end_if_conditions(self, distances_error, threshold=0.3, min_conf_states=3):
        if len(self.collision_hist) > 0:  # te has chocado, baby
            return True

        return False

    def set_init_pose(self):
        ## ---  Car
        waypoints_town = self.world.get_map().generate_waypoints(5.0)
        init_waypoint = waypoints_town[self.waypoints_init]

        if self.alternate_pose:
            self.setup_car_random_pose()
        elif self.waypoints_init is not None:
            if self.show_all_points:
                self.draw_waypoints(
                   waypoints_town,
                   self.waypoints_init,
                   self.waypoints_target,
                   self.waypoints_lane_id,
                   2000,
                )
            self.setup_car_fix_pose(init_waypoint)
        else:  # TODO: hacer en el caso que se quiera poner el target con .next()
            waypoints_lane = init_waypoint.next_until_lane_end(1000)
            waypoints_next = init_waypoint.next(1000)
            print(f"{init_waypoint.transform.location.x = }")
            print(f"{init_waypoint.transform.location.y = }")
            print(f"{init_waypoint.lane_id = }")
            print(f"{init_waypoint.road_id = }")
            print(f"{len(waypoints_lane) = }")
            print(f"{len(waypoints_next) = }")
            w_road = []
            w_lane = []
            for x in waypoints_next:
                w_road.append(x.road_id)
                w_lane.append(x.lane_id)

            counter_lanes = Counter(w_lane)
            counter_road = Counter(w_road)
            print(f"{counter_lanes = }")
            print(f"{counter_road = }")

            self.setup_car_fix_pose(init_waypoint)


        ## --- Sensor collision
        self.setup_col_sensor()

        # Create a camera sensor blueprint
        camera_bp = self.world.get_blueprint_library().find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", "800")
        camera_bp.set_attribute("image_size_y", "600")
        camera_bp.set_attribute("fov", "90")

        self.front_camera_1_5 = SensorManager(
            self.world,
            self.display_manager,
            "RGBCamera",
            carla.Transform(carla.Location(x=2, z=1.5), carla.Rotation(yaw=+00)),
            self.car,
            {},
            display_pos=[0, 0],
        )

        self.birds_eye_camera = SensorManager(
            self.world,
            self.display_manager,
            "BIRD_VIEW",
            carla.Transform(carla.Location(x=0, y=0, z=60), carla.Rotation(pitch=-90)),
            self.car,
            {},
            display_pos=[1, 0],
        )

        # self.front_camera_1_5_segmentated = SensorManager(
        #     self.world,
        #     self.display_manager,
        #     "SemanticCamera",
        #     carla.Transform(carla.Location(x=2, z=1.5), carla.Rotation(yaw=+00)),
        #     self.car,
        #     {},
        #     display_pos=[0, 1],
        # )

        # self.front_camera_1_5_red_mask = SensorManager(
        #     self.world,
        #     self.display_manager,
        #     "RedMask",
        #     carla.Transform(carla.Location(x=2, z=1.5), carla.Rotation(yaw=+0)),
        #     self.car,
        #     {},
        #     display_pos=[0, 2],
        # )

    def merge_and_extend_lines(self, lines, ll_segment):
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
        cv2.imshow('Merged Lines', merged_image) if self.sync_mode and self.show_images else None

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

    def has_bad_perception(self, distances_error, threshold=0.3, min_conf_states=5):
        done = False
        states_above_threshold = sum(1 for state_value in distances_error if state_value > threshold)

        if states_above_threshold is None:
            states_above_threshold = 0

        if (states_above_threshold > len(distances_error) - min_conf_states):  # salimos porque no detecta linea a la derecha
            done = True
        return done, states_above_threshold

    def detect_lane_detector(self, raw_image):
        image_tensor = raw_image.transpose(2, 0, 1).astype('float32') / 255
        x_tensor = torch.from_numpy(image_tensor).to("cuda").unsqueeze(0)
        model_output = torch.softmax(self.lane_model.forward(x_tensor), dim=1).cpu().numpy()
        return model_output


    def lane_detection_overlay(self, image, left_mask, right_mask):
        res = np.copy(image)
        # We show only points with probability higher than 0.5
        res[left_mask > 0.5, :] = [255,0,0]
        res[right_mask > 0.5,:] = [0, 0, 255]
        return res


