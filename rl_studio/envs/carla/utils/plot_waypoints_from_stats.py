#!/usr/bin/env python
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
import pickle
from collections import Counter
import glob
import os
import sys

try:
    sys.path.append(
        glob.glob(
            "../carla/dist/carla-*%d.%d-%s.egg"
            % (
                sys.version_info.major,
                sys.version_info.minor,
                "win-amd64" if os.name == "nt" else "linux-x86_64",
            )
        )[0]
    )
except IndexError:
    pass

import carla

import argparse


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--host",
        metavar="H",
        default="127.0.0.1",
        help="IP of the host server (default: 127.0.0.1)",
    )
    argparser.add_argument(
        "-p",
        "--port",
        metavar="P",
        default=2000,
        type=int,
        help="TCP port to listen to (default: 2000)",
    )
    argparser.add_argument(
        "-t",
        "--town",
        metavar="T",
        default="Town04",
        help="load Town to work with",
    )
    argparser.add_argument(
        "-o",
        "--only_road",
        default=False,
        metavar="R",
        help="only show roads",
    )
    args = argparser.parse_args()

    # Approximate distance between the waypoints
    WAYPOINT_DISTANCE = 5.0  # in meters

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(2.0)

        # world = client.get_world()
        world = client.load_world(args.town)
        carla_map = world.get_map()

        import matplotlib.pyplot as plt

        # plt.subplot(211)
        # Invert the y axis since we follow UE4 coordinates
        plt.gca().invert_yaxis()
        plt.margins(x=0.7, y=0)

        file = open(
            "/home/ruben/Desktop/RL-Studio/rl_studio/logs/retraining/follow_lane_carla_ddpg_auto_carla_baselines/TensorBoard/DDPG_Actor_conv2d32x64_Critic_conv2d32x64-20240907-162944/loc_stats.pkl"
            , "rb")
        location_stats = pickle.load(file)

        for location_stats_episode in location_stats:
            location_actions = location_stats_episode["actions"]
            location_rewards = location_stats_episode["rewards"]
            location_next_states = location_stats_episode["next_states"]

            # First POC is printing just velocities
            for key, value in location_actions.items():
                x, y, z = key
                location = carla.Location(
                    x=x,
                    y=y,
                    z=z,
                )

                acceleration = value[0]
                # For remember ->
                # - green 0.9
                # - yellow 0.7
                # - orange 0.5
                # - red
                # ---- Darker -> more acceleration = Darker color
                if acceleration > 0.9:
                    # Very high acceleration - Dark green
                    color = carla.Color(r=0, g=128, b=0)
                elif acceleration > 0.7:
                    # High acceleration - Dark yellow
                    red_value = int(255 * (1 - (acceleration - 0.7) / 0.2))
                    color = carla.Color(r=red_value, g=128, b=0)
                elif acceleration > 0.5:
                    # Moderate acceleration - Dark orange
                    red_value = int(255 * (1 - (acceleration - 0.5) / 0.2))
                    color = carla.Color(r=red_value, g=64, b=0)
                else:
                    # Low acceleration - Dark red
                    red_value = int(255 * (1 - acceleration / 0.5))
                    color = carla.Color(r=red_value, g=0, b=0)

                world.debug.draw_string(
                    location,
                    "X",
                    draw_shadow=False,
                    color=color,
                    life_time=10000000,
                    persistent_lines=True,
                )

    finally:
        pass


if __name__ == "__main__":
    try:
        main()
    finally:
        print("Done.")
