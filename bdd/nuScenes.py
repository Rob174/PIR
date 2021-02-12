
from nuscenceskit.pythonsdk.nuscenes import nuscenes
print("COucou")
nusc = nuscenes.NuScenes(version='v1.0-mini', dataroot='data', verbose=True)

print("COucou")
nusc.list_scenes()


