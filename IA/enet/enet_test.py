from enet.models.enet_unpooling.model import build,plot_model,transfer_weights
import os

def main():
    nc = 81
    dw = 256
    dh = 256
    dir_path = os.path.dirname(os.path.realpath(__file__))
    target_path = os.path.join(dir_path, 'model.png')

    autoencoder, model_name = build(nc=nc, w=dw, h=dh)
    plot_model(autoencoder, to_file=target_path, show_shapes=True)
    transfer_weights(model=autoencoder)
    return autoencoder

model = main()
from nuscenes.nuscenes import NuScenes
# From https://www.nuscenes.org/nuscenes?tutorial=nuscenes
nusc = NuScenes(version='v1.0-mini', dataroot='C:/Users/robin/Documents/projets/nuscene', verbose=True)
my_scene = nusc.sample[0]["data"]['CAM_FRONT']
print(len(nusc.sample))
print(my_scene)
nusc.render_sample_data(my_scene)
# first_sample_token = my_scene['first_sample_token']
# my_sample = nusc.get('sample', first_sample_token)
# sensor = 'CAM_FRONT'
# cam_front_data = nusc.get('sample_data', my_sample['data'][sensor])
# nusc.render_sample_data(cam_front_data['token'])