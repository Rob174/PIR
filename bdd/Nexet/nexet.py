import json
import csv

def extract_image_annotation():
    with open('train_boxes.csv') as cvs_file:
        y=csv.reader(cvs_file)
        file=[]
        for row in y:
            filename=row[0]
            category=row[5]
            coordonnees=[float(row[1]),float(row[2]),float(row[3]),float(row[4])]
            find=False
            for i in file:
                if i["imageName"] == filename:
                    find = True
                    find_category=False
                    for j in i['categories'].keys():
                        if j == category:
                            i['categories'][j].append(coordonnees)
                            find_category=True
                    if not find_category:
                        i['categories'][category] = [coordonnees]

            if find==False:
                file.append({"imageName": filename, "categories": {category:[coordonnees]}})
            print(file)
        with open("extracted_data_nexetImage.json", 'w') as fout:
            json.dump(file, fout)

extract_image_annotation()













