from nn import *
from engine import Matrics
import os
import pickle
import json

def save_obj(obj,file_path:str='artifacts/model.pkl'):
    dir_name = os.path.dirname(file_path)
    os.makedirs(dir_name,exist_ok=True)
    with open(file_path,'wb') as f:
        pickle.dump(obj,f)

def load_obj(file_path:str='artifacts/model.pkl'):
    with open(file_path,'rb') as f:
        obj=pickle.load(f)
    return obj

def to_json(obj,file_path:str='artifacts/params.json'):
    dir_name = os.path.dirname(file_path)
    os.makedirs(dir_name,exist_ok=True)
    with open(file_path,'w') as f:
        json.dump(obj,f)
        
def from_json(file_path:str='artifacts/params.json'):
    with open(file_path,'r') as f:
        obj=json.load(f)
    return obj