import ast


def extract_reward_function(filename, function_name):
    with open(filename, 'r') as file:
        tree = ast.parse(file.read())

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            return ast.get_source_segment(open(filename).read(), node)

    return None


if __name__ == '__main__':
    filename = '/home/ruben/Desktop/RL-Studio/rl_studio/envs/carla/followlane/followlane_carla_sb.py'
    function_name = 'rewards_easy'

    reward_function_code = extract_reward_function(filename, function_name)
    if reward_function_code:
        print(f"Reward function '{function_name}' extracted:\n{reward_function_code}")
    else:
        print(f"Function '{function_name}' not found in {filename}.")
