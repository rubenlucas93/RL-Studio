import math

import numpy as np


class F1GazeboRewards:
    @staticmethod
    def rewards_followlane_centerline(center, rewards):
        """
        works perfectly
        rewards in function of center of Line
        """
        done = False
        if 0.65 >= center > 0.25:
            reward = rewards["from_10"]
        elif (0.9 > center > 0.65) or (0.25 >= center > 0):
            reward = rewards["from_02"]
        elif 0 >= center > -0.9:
            reward = rewards["from_01"]
        else:
            reward = rewards["penal"]
            done = True

        return reward, done

    def rewards_followlane_v_centerline_step(self, vel_cmd, center, step, rewards):
        """
        rewards in function of velocity, angular v and center
        """

        done = False
        if 0.65 >= center > 0.25:
            reward = (rewards["from_10"] + vel_cmd.linear.x) - math.log(step)
        elif (0.9 > center > 0.65) or (0.25 >= center > 0):
            reward = (rewards["from_02"] + vel_cmd.linear.x) - math.log(step)
        elif 0 >= center > -0.9:
            # reward = (self.rewards["from_01"] + vel_cmd.linear.x) - math.log(step)
            reward = -math.log(step)
        else:
            reward = rewards["penal"]
            done = True

        return reward, done

    def rewards_followlane_v_w_centerline(
        self, vel_cmd, center, rewards, beta_1, beta_0
    ):
        """
        v and w are linear dependents, plus center to the eq.
        """

        w_target = beta_0 - (beta_1 * abs(vel_cmd.linear.x))
        w_error = abs(w_target - abs(vel_cmd.angular.z))
        done = False

        if abs(center) > 0.9 or center < 0:
            done = True
            reward = rewards["penal"]
        elif center >= 0:
            reward = (
                (1 / math.exp(w_error)) + (1 / math.exp(center)) + 2
            )  # add a constant to favor right lane
            # else:
            #    reward = (1 / math.exp(w_error)) + (math.exp(center))

        return reward, done

    def calculate_reward(self, error: float) -> float:
        d = np.true_divide(error, self.center_image)
        reward = np.round(np.exp(-d), 4)
        return reward

    def rewards_followline_center(self, center, rewards):
        """
        original for Following Line
        """
        done = False
        if center > 0.9:
            done = True
            reward = rewards["penal"]
        elif 0 <= center <= 0.2:
            reward = rewards["from_10"]
        elif 0.2 < center <= 0.4:
            reward = rewards["from_02"]
        else:
            reward = rewards["from_01"]

        return reward, done

    def rewards_all_points_followline_velocity_center(self, v, w, state, rewards):
        """
        original for Following Line
        """

        # we reward proximity to the line
        p_reward1, done1 = self.reward_proximity(state[4], rewards)
        p_reward2, done2 = self.reward_proximity(state[3], rewards)
        p_reward3, done3 = self.reward_proximity(state[2], rewards)
        p_reward4, done4 = self.reward_proximity(state[1], rewards)
        p_reward5, done5 = self.reward_proximity(state[0], rewards)
        p_reward = p_reward1 + 0.8*p_reward2 + 0.6*p_reward3 + 0.4*p_reward4 + 0.2*p_reward5
        done = done1 and done2 and done3 and done4 and done5

        # we reward higher velocities as long as the car keeps stick to the line
        v_reward = abs(v) * p_reward

        return p_reward + v_reward, done
    def rewards_followline_velocity_center(self, v, state, range_v):
        """
        original for Following Line
        """
        # we reward proximity to the line
        p_reward1, done1 = self.reward_proximity(state[-1])
        p_reward2, done2 = self.reward_proximity(state[4])
        p_reward = (p_reward1 + p_reward2)/2
        done = done1 and done2

        # we reward higher velocities as long as the car keeps stick to the line
        # v_reward = self.normalize_range(v, range_v[0], range_v[1])
        # v_reward = self.sigmoid_function(range_v[0], range_v[1], v, 5)
        v_reward = self.linear_function(-0.07, 0.07, v)
        #reward shaping to ease training with speed:
        if abs(state[-1]) > 0.5 and abs(state[4]) > 0.5:
            beta = 1
        else:
        # elif abs(state[4]) <= 0.3 and abs(state[3]) <= 0.3:
        #     beta = 0.7
        # elif abs(state[4]) <= 0.5 and abs(state[3]) <= 0.5:
        #     beta = 0.8
        # else:
        #     beta = 0.9
        # if abs(state[4]) <= 0.15:
        #     beta = 0.7
        # elif abs(state[4]) <= 0.4:
        #     beta = 0.8
        # else:
        #     beta = 0.9
            beta = 0.8
        reward = (beta * p_reward) + ((1 - beta) * (p_reward * v_reward))
        return reward, done

    def normalize_range(self, num, a, b):
        return (num - a) / (b - a)

    def reward_proximity(self, state):
        # sigmoid_pos = self.sigmoid_function(0, 1, state)
        if abs(state) > 0.7:
            return 0, True
        else:
            # return 1-self.sigmoid_function(0, 1, abs(state), 5), False
            return self.linear_function(1, -1.4, abs(state)), False

    def sigmoid_function(self, start, end, x, slope=10):
        slope = slope / (end - start)
        sigmoid = 1 / (1 + np.exp(-slope * (x - ((start + end) / 2))))
        return sigmoid

    def linear_function(self, cross_x, slope, x):
        return cross_x + (slope * x)

    def rewards_followline_v_w_centerline(
        self, vel_cmd, center, rewards, beta_1, beta_0
    ):
        """
        Applies a linear regression between v and w
        Supposing there is a lineal relationship V and W. So, formula w = B_0 + x*v.

        Data for Formula1:
        Max W = 5 r/s we take max abs value. Correctly it is w left or right
        Max V = 100 m/s
        Min V = 20 m/s
        B_0 = B_1 * Max V
        B_1 = (W Max / (V Max - V Min))

        w target = B_0 - B_1 * v
        error = w_actual - w_target
        reward = 1/exp(reward + center))) where Max value = 1

        Args:
                linear and angular velocity
                center

        Returns: reward
        """

        # print_messages(
        #    "in reward_v_w_center_linear()",
        #    beta1=self.beta_1,
        #    beta0=self.beta_0,
        # )

        w_target = beta_0 - (beta_1 * abs(vel_cmd.linear.x))
        w_error = abs(w_target - abs(vel_cmd.angular.z))
        done = False

        if abs(center) > 0.9:
            done = True
            reward = rewards["penal"]
        elif center > 0:
            reward = (
                (1 / math.exp(w_error)) + (1 / math.exp(center)) + 2
            )  # add a constant to favor right lane
        else:
            reward = (1 / math.exp(w_error)) + (math.exp(center))

        return reward, done
