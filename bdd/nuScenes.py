from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import BoxVisibility
import matplotlib.pyplot as plt
nusc = NuScenes(version='v1.0-mini', dataroot='/mnt/c/Users/robin/Documents/projets/nuscene', verbose=True)


nusc.list_scenes()
my_scene = nusc.scene[0]
first_sample_token = my_scene['first_sample_token']
my_sample = nusc.get('sample', first_sample_token)
sensor = 'CAM_FRONT'
cam_front_data = nusc.get('sample_data', my_sample['data'][sensor])
"""On pourrait potentiellement récupérer les infos dans variables"""
variable  = nusc.get_sample_data(cam_front_data['token'], box_vis_level=BoxVisibility.ANY)
nusc.render_sample_data(cam_front_data['token'],verbose=True)
plt.show()

