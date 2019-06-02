import airsim
import numpy as np
from src.utils import data_path
import time
import cv2
import json

p = data_path() / 'blocks/'

# connect to the AirSim simulator
client = airsim.MultirotorClient()
client.confirmConnection()
#client.enableApiControl(True)
#client.armDisarm(True)

# Async methods returns Future. Call join() to wait for task to complete.
#client.takeoffAsync().join()
#client.moveToPositionAsync(-10, 10, -10, 5).join()

time.sleep(3)
start = 102
increment = 898
end = start + increment

if start > 0:  # Continue
    n_path = p / 'data_.txt'
    with open(n_path) as file:
        data = json.load(file)
else:
    data = {'camera': {'w': 1368, 'h': 912, 'fov': 77}, 'scenes': []}


for i in range(start, end):
    client.simPause(True)
    responses = client.simGetImages([
        airsim.ImageRequest('0', airsim.ImageType.DepthPerspective, True, False),
        airsim.ImageRequest('0', airsim.ImageType.Scene, False, False),
        airsim.ImageRequest('0', airsim.ImageType.Test, False, False)
    ])
    depth = responses[0]
    scene = responses[1]
    gt = responses[2]
    pose = client.simGetVehiclePose()
    client.simPause(False)

    img = airsim.get_pfm_array(depth)
    img = img / 100  # To meters

    # Visualized depth (8 bits depth png
    dv = img/120 * 255
    dv = np.round(dv)
    n_path = p / ('depth_v_' + str(i) + '.png')
    cv2.imwrite(str(n_path), dv)

    # Float depth
    n_path = p / ('depth_f32_' + str(i) + '.npz')
    np.savez_compressed(n_path, data=img)  # NB: Probably flipped

    # Scene png
    img1d = np.fromstring(scene.image_data_uint8, dtype=np.uint8)
    # reshape array to 4 channel image array H X W X 4
    img_rgb = img1d.reshape(scene.height, scene.width, 4)
    # original image is fliped vertically
    img_rgb = np.flipud(img_rgb)
    n_path = p / ('scene_' + str(i) + '.png')
    airsim.write_png(p.joinpath(n_path), img_rgb)

    # GT png
    img1d = np.fromstring(gt.image_data_uint8, dtype=np.uint8)
    # reshape array to 4 channel image array H X W X 4
    img_rgb = img1d.reshape(gt.height, gt.width, 4)
    # original image is fliped vertically
    img_rgb = np.flipud(img_rgb)
    n_path = p / ('gt_' + str(i) + '.png')
    airsim.write_png(p.joinpath(n_path), img_rgb)

    # Camera
    d = {'scene': i, 'orientation': {
        'w': pose.orientation.w_val,
        'x': pose.orientation.x_val,
        'y': pose.orientation.y_val,
        'z': pose.orientation.z_val
    }, 'position': {
        'x': pose.position.x_val,
        'y': pose.position.y_val,
        'z': pose.position.z_val
    }}
    data['scenes'].append(d)
    #time.sleep(0.05)

n_path = p / 'data_.txt'
with open(n_path, 'w') as file:
    json.dump(data, file)


