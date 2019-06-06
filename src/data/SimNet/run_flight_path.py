import airsim
import json
from pathlib import Path
import time
from scipy.spatial.transform import Rotation
import numpy as np
import cv2


DATA_PATH = Path('F:/data/blocks/')
FLIGHT_NUMBER = 1

LABEL_MAP = [([0, 0, 0], 0), ([255, 0, 0], 1), ([255, 255, 0], 2), ([0, 0, 255], 3)]

# TODO: Create file hierarchy first?


def follow_path(client, data, path):
    client.simPause(True)
    n = 0
    for p in data['flight_path']:
        n += 1
        pos = airsim.Vector3r(p['drone_position']['x'], p['drone_position']['y'], p['drone_position']['z'])
        q = p['drone_orientation']
        rot = airsim.Quaternionr(q['x'], q['y'], q['z'], q['w'])
        pose = airsim.Pose(pos, rot)
        client.simSetVehiclePose(pose, False)
        r = np.random.rand()
        if r > 0.3:
            dp = path / 'train/'
        elif r > 0.1:
            dp = path / 'val/'
        else:
            dp = path / 'test'
        process_position(client, dp, n)
        time.sleep(0.1)


def process_position(client, path, n):
    responses = client.simGetImages([
        airsim.ImageRequest('0', airsim.ImageType.DepthPerspective, True, False),
        airsim.ImageRequest('0', airsim.ImageType.Scene, False, False),
        airsim.ImageRequest('0', airsim.ImageType.Test, False, False)
    ])
    depth = responses[0]
    scene = responses[1]
    gt = responses[2]
    pose = client.simGetCameraInfo('0').pose
    save_scene_image(scene, path, n)
    save_depth_image(depth, path, n)
    save_label_image(gt, path, n)
    save_camera_extrinsic(pose, path, n)


def save_depth_image(depth, path, n):
    img = airsim.get_pfm_array(depth)
    img = img / 100  # To meters

    # (16 bit png image)
    dv = (img / 120) * 65535
    dv = np.round(dv)
    dv = dv.astype('uint16')
    path = path / ('depth/' + str(n) + '.png')
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), dv)


def save_scene_image(scene, path, n):
    img1d = np.fromstring(scene.image_data_uint8, dtype=np.uint8)
    # reshape array to 4 channel image array H X W X 4
    img_rgb = img1d.reshape(scene.height, scene.width, 4)  # Test
    path = path / ('scene/' + str(n) + '.png')
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))  # Saving rgb image


def save_label_image(labels, path, n):
    img1d = np.fromstring(labels.image_data_uint8, dtype=np.uint8)
    # reshape array to 4 channel image array H X W X 4
    img_rgb = img1d.reshape(labels.height, labels.width, 4)
    image = img_rgb[:, :, :3]  # Without alpha

    vpath = path / ('color_label/' + str(n) + '.png')
    vpath.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(vpath), image)

    label_image = np.zeros((image.shape[:2]), dtype=np.uint8)
    for m in LABEL_MAP:
        label_image[(image == m[0]).all(axis=2)] = m[1]

    path = path / ('label/' + str(n) + '.png')
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), label_image)


def save_camera_extrinsic(pose, path, n):
    rot = Rotation.from_quat([
        pose.orientation.x_val,
        pose.orientation.y_val,
        pose.orientation.z_val,
        pose.orientation.w_val])
    pos = np.array([
        [pose.position.x_val],
        [pose.position.y_val],
        [pose.position.z_val]])
    rot = rot.as_dcm()
    extrinsic = np.append(rot, pos, axis=1)
    path = path / ('extrinsic/' + str(n))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, extrinsic)


def main():
    client = airsim.MultirotorClient()
    client.confirmConnection()
    f_path = DATA_PATH / ('flight_path_' + str(FLIGHT_NUMBER) + '.json')
    with open(f_path) as file:
        data = json.load(file)
    follow_path(client, data, DATA_PATH)

    print('Done')


if __name__ == '__main__':
    main()
