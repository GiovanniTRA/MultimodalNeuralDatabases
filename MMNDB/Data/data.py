import json
import torch
from torch.utils.data import Dataset
import os
from PIL import Image
from MMNDB.Data.process_raw import PreprocessRawData
from torch.utils.data import DataLoader
from torchvision import transforms
import clip



class CustomCocoDataset(Dataset):

    def __init__(self, path, split="val", tensorize=False):
        self.path = path
        self.split = split
        self.files = sorted(os.listdir(f"{self.path}/{self.split}2017/"))
        self.tensorize = tensorize

        self.convert_tensor = transforms.ToTensor()

        
        # all this function below load two arguments.
        # There is an underlying assumption that if one aldready exists
        # this applies to the other one as well.
        # it should be ok, because if it bugged then it should break soon.
        self.processed_path = f"{self.path}/../../processed"
        self.dict_img_obj, self.dict_obj_img = self.load_dict_img_obj()
        self.dict_obj_name_id, self.dict_obj_id_name = self.load_dict_obj_name_id()
        self.dict_img_id_to_filename , self.dict_filename_to_img_id = self.load_dict_img_id_filename()

    
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
        return len(os.listdir(f"{self.path}/{self.split}2017/"))


    def __getitem__(self, index):
        
        file = self.files[index]
        img = Image.open(f"{self.path}/{self.split}2017/{file}").convert("RGB")

        if self.tensorize:
            img = self.convert_tensor(img)

        img_id = self.dict_filename_to_img_id[file]

        return img_id, img

    
    @staticmethod
    def custom_collate(batch):
        img_id = [i[0] for i in batch]
        img = [i[1] for i in batch]
        return img_id, img


if __name__ == "__main__":
    data = CustomCocoDataset("../test/support_materials/raw/images")
    dataloader = DataLoader(data, batch_size=2, collate_fn=data.custom_collate)

    # model, preprocess = clip.load("RN50")

    # for i, item in enumerate(dataloader):
    #     img_ids = item[0]
    #     imgs = item[1]
    #     preprocess(imgs) 
    #     print(i, item)
    # print(data[0])




