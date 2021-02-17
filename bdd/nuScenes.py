from nuscenceskit.pythonsdk.nuscenes import nuscenes
import numpy as np
import Image as img

nusc = nuscenes.NuScenes(version='v1.0-mini', dataroot='data', verbose=True)

nusc.list_scenes()

print('Courgette')
my_scene = nusc.scene[0]
print(my_scene)
first_sample_token = my_scene['first_sample_token']
my_sample = nusc.get('sample', first_sample_token)
print(my_sample)

nusc.list_sample(my_sample['token'])

sensor = 'CAM_FRONT'
cam_front_data = nusc.get('sample_data', my_sample['data'][sensor])
print(cam_front_data)


nusc.render_sample_data(cam_front_data['token'])


def Nuscenes_image_to_array()






