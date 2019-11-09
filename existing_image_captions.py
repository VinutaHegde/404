import tensorflow as tf
import scipy
import numpy as np
import json
import importlib.util
import sys
sys.path.append('/Users/wesam/PycharmProjects/DeepLearningProject/venv/bin/vist_api')
import vist
import os.path

vist_dir = '/Users/wesam/PycharmProjects/DeepLearningProject/venv/annotations'
images_dir = '/Users/wesam/PycharmProjects/DeepLearningProject/venv/images'
sis = vist.Story_in_Sequence(images_dir,vist_dir)


#check if images exist in the downloaded splits
existing_imgs_albums=[]
for img in sis.images:
    path=images_dir+'/train/'+img['id']+'.jpg'
    if os.path.isfile(path):
        if img['album_id'] not in existing_imgs_albums:
            existing_imgs_albums.append(img['album_id'])

album_id = existing_imgs_albums[1]
story_ids = sis.Albums[album_id]['story_ids']
story_id = story_ids[1]
sis.show_story(story_id)
print(sis.Stories[story_id]['img_ids'])
