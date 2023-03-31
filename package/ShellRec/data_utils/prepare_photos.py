import os
import glob
import json
from os.path import join
import random

def get_img_graph(path, extensions = ["*.JPG", "*.jpeg","*.jpg"],
                  file_to_save=None, 
                  drop_p = [0,0]):
    '''
    reads in all the images in a path, get a json with each item being [img1, img2, label]
    where label is whether they are from the same individual. 
    I assume that the string before first "-" is the individual identifier

    drop_p: [p1, p2] where p1 is the probability of dropping an pair from the different individual, 
    p2 is the probability of dropping a pair from same individuals
    '''
    img_graph = []
    if(len(extensions) == 0):
        extensions = ["*.JPG", "*.jpeg","*.jpg"]
    img_paths = []
    for ext in extensions:
        img_paths += glob.glob(os.path.join(path, ext))
    for i in range(len(img_paths)):
        for j in range(i, len(img_paths)):
            img1 = img_paths[i]
            img2 = img_paths[j]
            label = 1 if img1.split("-")[0] == img2.split("-")[0] else 0
            if random.uniform(0,1) > drop_p[label]:
                img_graph.append([img1, img2, label])
    if file_to_save:
        with open(file_to_save, "w") as f:
            json.dump(img_graph, f)
    return img_graph

def split_graph(img_graph, train_ratio=0.8, 
                val_ratio=0.1, 
                test_ratio=0.1, save_path=None):
    '''
    split the img_graph into train, val, test sets
    '''
    assert train_ratio + val_ratio + test_ratio == 1
    random.shuffle(img_graph)
    
    train_size = int(len(img_graph) * train_ratio)
    val_size = int(len(img_graph) * val_ratio)
    #test_size = len(img_graph) - train_size - val_size
    train_set = img_graph[:train_size]
    val_set = img_graph[train_size:train_size+val_size]
    test_set = img_graph[train_size+val_size:]
    
    if save_path:
        with open(join(save_path, "train.json"), "w") as f:
            json.dump(train_set, f)
        with open(join(save_path, "val.json"), "w") as f:
            json.dump(val_set, f)
        with open(join(save_path, "test.json"), "w") as f:
            json.dump(test_set, f)
    return train_set, val_set, test_set