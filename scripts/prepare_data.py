import os
import glob
import json
from os.path import join
from ShellRec.data_utils.prepare_photos import get_img_graph, split_graph

def main():
    all_train = get_img_graph(path = "./dataset/BoxTurtles")
    split_graph(all_train, save_path = "./dataset")
    _ = get_img_graph(path = "./dataset/BoxTurtlesHoldout", 
                             file_to_save = "./dataset/shell_rec_holdout.json")

if __name__ == "__main__":
    main()
