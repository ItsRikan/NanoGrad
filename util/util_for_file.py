import os
import pickle


def save_obj(obj,file_path):
    dir_name = os.path.dirname(file_path)
    os.makedirs(dir_name,exist_ok=True)
    with open(file_path,'wb') as f:
        pickle.dump(obj,f)


def load_obj(file_path):
    with open(file_path,'rb') as f:
        obj=pickle.load(f)
    return obj
