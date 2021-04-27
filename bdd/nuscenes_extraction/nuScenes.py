from nuscenceskit.pythonsdk.nuscenes import nuscenes
import json

def extract_image_annotation():
    with open('./data/v1.0-mini/image_annotations.json') as json_file:
        y=json.loads(json_file.read())
        file=[]
        for p in y:
            filename = p['filename']
            category = p['category_name']
            bbox = p['bbox_corners']
            find = False
            for i in file:
                if i["imageName"] == filename:
                    find = True
                    find_category=False
                    for j in i['categories'].keys():
                        if j == category:
                            i['categories'][j].append(bbox)
                            find_category=True
                    if not find_category:
                        i['categories'][category] = [bbox]

            if find==False:
                file.append({"imageName": filename, "categories": {category:[bbox]}})
        #print(file)
        with open("extracted_data_nusceneImage.json",'w') as fout:
           json.dump(file,fout)

def extract_lidar_annotation():
    with open('./data/v1.0-mini/sample_annotation.json') as sample_annotation_file, open('./lidar/v1.0-mini/lidarseg.json') as lidar_json, open('./data/v1.0-mini/sample_data.json') as sample_data_file,  open('./data/v1.0-mini/instance.json') as instance_file,open('./lidar/v1.0-mini/category.json') as category_file:
        lidarseg_file = json.loads(lidar_json.read())
        category_file = json.loads(category_file.read())
        sample_data_file = json.loads(sample_data_file.read())
        instance_file = json.loads(instance_file.read())
        sample_annotation_file = json.loads(sample_annotation_file.read())


        file=[]
        for lidar in lidarseg_file:
            filename=lidar['filename']
            sample_data_token=lidar['sample_data_token']
            for sample_data in sample_data_file:
                if(sample_data['token']==sample_data_token):
                    sample_token=sample_data['sample_token']
                    break

            for sample_annotation in sample_annotation_file:
                if(sample_token==sample_annotation['sample_token']):
                    instance_token = sample_annotation['instance_token']
                    for instance in instance_file:
                        if(instance_token==instance['token']):
                            category_token=instance['category_token']
                            for category in category_file:
                                if (category_token==category['token']):
                                    category_name=category['name']
                    nb_points_lidar=sample_annotation['num_lidar_pts']
                    size = sample_annotation['size']
                    rotation = sample_annotation['rotation']
                    translation=sample_annotation['translation']
                    find=False
                    for i in file:
                        if i["lidarName"] == filename:
                            find = True
                            find_category = False
                            for j in i['categories'].keys():
                                if j == category_name:
                                    i['categories'][j].append({'size':size,'rotation':rotation,'translation':translation,'nb_pts':nb_points_lidar})
                                    find_category = True
                            if not find_category:
                                i['categories'][category_name] = [{'size':size,'rotation':rotation,'translation':translation,'nb_pts':nb_points_lidar}]
                    if find == False:
                        file.append({"lidarName": filename, "categories": {category_name:[{'size':size,'rotation':rotation,'translation':translation,'nb_pts':nb_points_lidar}]}})
    with open("extracted_data_nusceneLidar.json", 'w') as fout:
        json.dump(file, fout)


def get_scene(imageName):
    sceneName=imageName.partition("+")[0]
    return sceneName



def extract_scene_annotations():
    with open("extracted_data_nusceneImage.json") as jsonfile:
        input = json.loads(jsonfile.read())
        file = []
        for image in input:
            filename = image['imageName']
            scenename = get_scene(filename)
            annotation = list(image['categories'])
            find_scene=False
            for i in file:
                if (scenename==i['scene']):
                    i['images'].append(filename)
                    find_annotation = False
                    for j in annotation:
                        for k in i['categories']:
                            if (k==j):
                                find_annotation=True
                        if (find_annotation==False):
                            i['categories'].append(j)
            if (find_scene==False):
                file.append({"scene": scenename, "images": [filename],"categories":annotation})
        with open("extracted_scene_annotation.json", 'w') as fout:
            json.dump(file, fout)


extract_scene_annotations()
















