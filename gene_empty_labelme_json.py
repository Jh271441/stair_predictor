import os
import json
import cv2

dirs = os.listdir("stair_datasets_3")
for dir in dirs:
    files = os.listdir(os.path.join("stair_datasets_3", dir))
    for file in files:
        img = cv2.imread(os.path.join("stair_datasets_3", dir, file))
        imagePath = f"../stair_datasets_3/{dir}/{file}"
        imageData = None
        imageHeight = img.shape[0]
        imageWidth = img.shape[1]
        labelme_json = {
            "version": "5.5.0",
            "flags": {},
            "shapes": [],
            "imagePath": imagePath,
            "imageData": imageData,
            "imageHeight": imageHeight,
            "imageWidth": imageWidth
        }
        with open(f"labelme_json_3/e_{dir[0:5]}_{file[:-4]}.json", "w") as f:
            json.dump(labelme_json, f)
        print(f"Generate e_{dir[0:5]}_{file[:-4]}.json successfully!")