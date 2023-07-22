import json
import random
import torch
from torch.utils.data import Dataset
import os
from PIL import Image
from MMNDB.Data.process_raw import PreprocessRawData
from torch.utils.data import DataLoader
from torchvision import transforms
import clip
import pytorch_lightning as pl
from transformers import OFATokenizer



class CustomProcessorCocoDataset(Dataset):

    def __init__(self, path, retrieved, split="val"):

        self.path = path
        self.split = split
        self.files = sorted(os.listdir(f"{self.path}/{self.split}2017/"))
        self.retrieved = retrieved
        
        # all this function below load two arguments.
        # There is an underlying assumption that if one aldready exists
        # this applies to the other one as well.
        # it should be ok, because if it bugged then it should break soon.
        self.processed_path = f"{self.path}/../../processed"
        self.dict_img_obj, self.dict_obj_img = self.load_dict_img_obj()
        self.dict_obj_name_id, self.dict_obj_id_name = self.load_dict_obj_name_id()
        self.dict_img_id_to_filename , self.dict_filename_to_img_id = self.load_dict_img_id_filename()
        self.preprocess_img = self.preprocess_img()

    
    def preprocess_img(self, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], resolution=480):

        patch_resize_transform = transforms.Compose([
                lambda image: image.convert("RGB"),
                transforms.Resize((resolution, resolution), interpolation=Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
        
        return patch_resize_transform

    def load_dict_img_id_filename(self):
        if not os.path.isfile(f"{self.processed_path}/{self.split}_img_id_to_filename.json"):
             p = PreprocessRawData(f"{self.path}/../annotations", self.processed_path, split=self.split)
             p.create_dict_img_id_filename()
        
        dict_img_id_to_filename = json.load(open(f"{self.processed_path}/{self.split}_img_id_to_filename.json"))
        dict_filename_to_img_id = json.load(open(f"{self.processed_path}/{self.split}_filename_to_img_id.json"))

        return dict_img_id_to_filename, dict_filename_to_img_id

    
    def load_dict_obj_name_id(self):
        if not os.path.isfile(f"{self.processed_path}/obj_name_to_id.json.json"):
            p = PreprocessRawData(f"{self.path}/../annotations", self.processed_path, split=self.split)
            p.create_dict_obj_name()
        
        dict_obj_name_id = json.load(open(f"{self.processed_path}/obj_name_to_id.json"))
        dict_obj_id_name = json.load(open(f"{self.processed_path}/obj_id_to_name.json"))

        return dict_obj_name_id, dict_obj_id_name
    
    def load_dict_img_obj(self):
        if not os.path.isfile(f"{self.processed_path}/{self.split}_img_id_to_obj_id.json"):
            p = PreprocessRawData(f"{self.path}/../annotations", self.processed_path, split=self.split)
            p.create_dict_img_obj()
        
        dict_img_obj = json.load(open(f"{self.processed_path}/{self.split}_img_id_to_obj_id.json"))
        dict_obj_img = json.load(open(f"{self.processed_path}/{self.split}_obj_id_to_img_id.json"))
        
        return dict_img_obj, dict_obj_img


    def __len__(self):
        return len(self.retrieved)


    def __getitem__(self, index):

        img_id = self.retrieved[index]
        file = self.dict_img_id_to_filename[str(img_id.item())]

        img = Image.open(f"{self.path}/{self.split}2017/{file}")
        img = self.preprocess_img(img)
        
        return img_id , img



if __name__ == "__main__":
    pl.seed_everything()
    data = CustomProcessorCocoDataset("../test/support_materials/raw/images", torch.tensor([397133]))
    print(data[0])

    dataloader = DataLoader(data, batch_size=2, num_workers=0)
    for i in dataloader:
        print(i)
        print("\n\n\n\n\n\n\n\n\n\n")
    # model, preprocess = clip.load("RN50")

    # for i, item in enumerate(dataloader):
    #     img_ids = item[0]
    #     imgs = item[1]
    #     preprocess(imgs) 
    #     print(i, item)
    # print(data[0])




