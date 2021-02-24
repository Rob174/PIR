from nuscenceskit.pythonsdk.nuscenes import nuscenes
from PIL import Image
from numpy import asarray
import h5py

nusc = nuscenes.NuScenes(version='v1.0-mini', dataroot='data', verbose=True)

def Nuscenes_import_image_to_array(dirname):
    im = Image.open(dirname)
    im_array=asarray(im)
    return im_array

def Add_image_to_h5py(dirname):
    mon_fichier = h5py.File('./mon_fichier.hdf5', 'a')
    nuscenes_group = mon_fichier.create_group('nuscenes')
    ma_matrice = Nuscenes_import_image_to_array(dirname)
    mon_dataset = nuscenes_group.create_dataset(name='demo_dataset', data=ma_matrice, dtype="i8")
    mon_fichier.close()


Add_image_to_h5py("C:/Users/emmab/Pictures/Australie/Australie MOI/Australie/20160206_124831.jpg")












