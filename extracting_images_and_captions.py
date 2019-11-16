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


#creates a json:
# { "album_id": {
#               "story_id": [
#                   ["image_id" , "some caption...." ]
#                   ["image_id" , "some caption...." ]
#                   ["image_id" , "some caption...." ]
#                   ["image_id" , "some caption...." ]
#                   ["image_id" , "some caption...." ]
#               ]
#               , "story_id": [
#                    ["image_id" , "some caption...." ]
#                    ["image_id" , "some caption...." ]
#                    ["image_id" , "some caption...." ]
#                    ["image_id" , "some caption...." ]
#                    ["image_id" , "some caption...." ]
#                ]}
#   , "album_id": {
#               "story_id": [
#                   ["image_id" , "some caption...." ]
#                   ["image_id" , "some caption...." ]
#                   ["image_id" , "some caption...." ]
#                   ["image_id" , "some caption...." ]
#                   ["image_id" , "some caption...." ]
#               ]
#               ,"story_id": [
#                    ["image_id" , "some caption...." ]
#                    ["image_id" , "some caption...." ]
#                    ["image_id" , "some caption...." ]
#                    ["image_id" , "some caption...." ]
#                    ["image_id" , "some caption...." ]
#                ]}
# }
#
def make_json_caption_file(sis):
    json_dict={}
    max_cap_len=0
    #max=''
    for album_id in sis.Albums:
        if(album_id not in json_dict):
            json_dict[album_id]={}
        for story_id in sis.Albums[album_id]['story_ids']:
            #sis.show_story(story)
            if (story_id not in json_dict[album_id]):
                json_dict[album_id][story_id]={}
            story = sis.Stories[story_id]
            list_cap=[]
            sent_ids = story['sent_ids']
            for sent_id in sent_ids:
                sent = sis.Sents[sent_id]
                list_cap.append((sent['img_id'],sent['text']))
                cap_len=len(sent['text'].split())
                if max_cap_len < cap_len:
                    max_cap_len=cap_len
                    #max=sent['text']
            json_dict[album_id][story_id]=list_cap
    with open('all_captions.txt', 'w') as outfile:
        json.dump(json_dict,outfile)
    #print(max)
    return max_cap_len


#Get all images that have captions
def all_images_with_captions(sis):
    img_dict={}
    for album_id in sis.Albums:
        for story_id in sis.Albums[album_id]['story_ids']:
            story = sis.Stories[story_id]
            sent_ids = story['sent_ids']
            for sent_id in sent_ids:
                sent = sis.Sents[sent_id]
                img_dict[sent['img_id']]=1
    with open('all_images.txt', 'w') as outfile:
        json.dump(list(img_dict.keys()),outfile)



#check if images exist in the downloaded splits
def get_album_ids_for_existing_images(images_dir,sis):
    existing_imgs_albums=[]
    for img in sis.images:
        path=images_dir+'/train/'+img['id']+'.jpg'
        if os.path.isfile(path):
            if img['album_id'] not in existing_imgs_albums:
                existing_imgs_albums.append(img['album_id'])
    return existing_imgs_albums

#here you can iterate over all albums & all stories inside these albums
existing_imgs_albums=get_album_ids_for_existing_images(images_dir,sis)
album_id = existing_imgs_albums[1]
story_ids = sis.Albums[album_id]['story_ids']
story_id = story_ids[0]
sis.show_story(story_id)
