import airsim
import json
from pathlib import Path
import time

DATA_PATH = Path('F:/data/blocks/flight_path.json')
CONTINUE = False


def write_pose(data, d_pose, c_pose):
    data['flight_path'].append({
        'drone_orientation': {
            'w': d_pose.orientation.w_val,
            'x': d_pose.orientation.x_val,
            'y': d_pose.orientation.y_val,
            'z': d_pose.orientation.z_val
        },
        'drone_position': {
            'x': d_pose.position.x_val,
            'y': d_pose.position.y_val,
            'z': d_pose.position.z_val
        },
        'camera_orientation': {
            'w': c_pose.orientation.w_val,
            'x': c_pose.orientation.x_val,
            'y': c_pose.orientation.y_val,
            'z': c_pose.orientation.z_val
        },
        'camera_position': {
            'x': c_pose.position.x_val,
            'y': c_pose.position.y_val,
            'z': c_pose.position.z_val
        }
    })
    data['count'] = data['count'] + 1
    return data


def init_json(cont):
    if cont:
        with open(DATA_PATH) as file:
            data = json.load(file)
    else:
        data = {
            'count': 0,
            'flight_path': []
        }
    return data


def map_flight_path(client, data, max_iter=2400):
    for i in range(max_iter):
        write_pose(data, client.simGetVehiclePose(), client.simGetCameraInfo('0').pose)
        time.sleep(0.05)
        print(i)
    return


def main():
    client = airsim.MultirotorClient()
    client.confirmConnection()
    print('Starting in 3')
    time.sleep(1)
    print('2')
    time.sleep(1)
    print('1')
    time.sleep(1)
    print('0')
    data = init_json(CONTINUE)
    map_flight_path(client, data)
    with open(DATA_PATH, 'w') as file:
        json.dump(data, file)
    print('Done')


if __name__ == '__main__':
    main()
