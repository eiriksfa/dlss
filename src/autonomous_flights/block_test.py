import airsim
import time
import numpy as np
import imageio
from pathlib import Path
import src.semantic_segmentation.ESimNetDepth.deployed as segm

model = segm.load_model()
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)

#client.armDisarm(True)

#landed = client.getMultirotorState().landed_state
#if landed == airsim.LandedState.Landed:
 #   print("taking off...")
client.takeoffAsync().join()
#else:
#    client.hoverAsync().join()

# AirSim uses NED coordinates so negative axis is up.
# z of -7 is 7 meters above the original launch point.
z = -7

# see https://github.com/Microsoft/AirSim/wiki/moveOnPath-demo

path = [airsim.Vector3r(10.64, 29.91, -39.02),
        airsim.Vector3r(-16.5, 80.34, -54.95),
        airsim.Vector3r(-56.7, -5.29, -37.02),
        airsim.Vector3r(-8.78, -98.94, -29.82),
        airsim.Vector3r(-46.15, -46.74, -54.82),
        airsim.Vector3r(-65.9, 49.9, -35.87),
        airsim.Vector3r(-58.4, -20.54, -30)
        ]

# this method is async and we are not waiting for the result since we are passing timeout_sec=0.
result = client.moveOnPathAsync(path,
                                12, 120,
                                airsim.DrivetrainType.ForwardOnly, airsim.YawMode(False, 0), 20, 1).join()

print('finished, landing')

responses = client.simGetImages([
    airsim.ImageRequest('bottom_center', airsim.ImageType.DepthPerspective, True, False),
    airsim.ImageRequest('bottom_center', airsim.ImageType.Scene, False, False),
    airsim.ImageRequest('front_center', airsim.ImageType.DepthPerspective, True, False),
    airsim.ImageRequest('front_center', airsim.ImageType.Scene, False, False)
])

for i in range(2):
    depth = responses[2 * i]
    scene = responses[1 + (2 * i)]

    camera = 'front' if i == 1 else 'bottom'

    img = airsim.get_pfm_array(depth)
    img = img / 100  # To meters

    # (16 bit png image)
    dv = (img / 120) * 65535
    dv = np.round(dv)
    dv = dv.astype('uint16')
    path = Path('Z:/data/blocks/depth_' + camera + '.png')
    imageio.imwrite(path, dv)

    img1d = np.fromstring(scene.image_data_uint8, dtype=np.uint8)
    # reshape array to 4 channel image array H X W X 4
    scene = img1d.reshape(scene.height, scene.width, 4)  # Test
    scene = scene[:, :, :3]
    path = Path('Z:/data/blocks/scene_' + camera + '.png')
    imageio.imwrite(path, scene)

    seg_result = segm.predict(scene, dv, model)
    rp = Path('Z:/data/blocks/result_' + camera + '.png')
    imageio.imwrite(rp, seg_result)

client.moveToPositionAsync(-59.4, -23.54, -10, 5)
client.landAsync().join()

