import carla

def get_fps(world):
    frame_times = world.debug_tick_data.frame_times
    avg_frame_time = sum(frame_times) / len(frame_times)
    fps = 1 / avg_frame_time
    return fps

def main():# Connect to the Carla server
    client = carla.Client('localhost', 4001)
    client.set_timeout(5.0)

    # Get the world and the spectator
    world = client.get_world()
    spectator = world.get_spectator()

    # rotation = carla.Rotation(pitch=90.999451, yaw=161.219055, roll=-179.999800)
    # lotation = carla.Location(x=2.514739, y=51.087231, z=42.113667)
    # new_spectator_transform = carla.Transform(lotation, rotation)
    # spectator.set_transform(new_spectator_transform)

    # Retrieve the transform of the spectator
    spectator_transform = spectator.get_transform()

    # Extract the rotation from the transform
    spectator_rotation = spectator_transform.rotation
    spectator_location = spectator_transform.location

    print(f"Spectator Rotation: {spectator_location}")
    print(f"Spectator Rotation: {spectator_rotation}")

    # Query and print FPS
    fps = get_fps(world)
    print(f'Current FPS: {fps}')

if __name__ == '__main__':
    main()
