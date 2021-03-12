from nuscenceskit.pythonsdk.nuscenes import nuscenes
import json
with open('nuscenes_extraction/data/v1.0-mini/image_annotations.json') as json_file:
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
    with open("extracted_data.json",'w') as fout:
       json.dump(file,fout)















