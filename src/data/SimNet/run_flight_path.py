import airsim
import json
from pathlib import Path
import time

DATA_PATH = Path('F:/data/blocks/flight_path.json')


def follow_path(client, data):
    client.simPause(True)
    for p in data['flight_path']:
        pos = airsim.Vector3r(p['drone_position']['x'], p['drone_position']['y'], p['drone_position']['z'])
        q = p['drone_orientation']
        rot = airsim.Quaternionr(q['w'], q['x'], q['y'], q['z'])
        pose = airsim.Pose(pos, rot)
        client.simSetVehiclePose(pose, False)
        time.sleep(0.005)


def main():
    client = airsim.MultirotorClient()
    client.confirmConnection()
    with open(DATA_PATH) as file:
        data = json.load(file)
    follow_path(client, data)

    print('Done')


if __name__ == '__main__':
    main()