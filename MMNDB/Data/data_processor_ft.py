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
import pickle



class CustomProcessorFTCocoDataset(Dataset):

    def __init__(self, path, split="val", checkpoint="OFA-large", prob=0.5):

        self.path = path
        self.split = split
        self.files = sorted(os.listdir(f"{self.path}/{self.split}2017/"))
        self.checkpoint = f"{self.path}/../../../../../{checkpoint}"
        self.prob = prob # probability of label=1
        
        # all this function below load two arguments.
        # There is an underlying assumption that if one aldready exists
        # this applies to the other one as well.
        # it should be ok, because if it bugged then it should break soon.
        self.processed_path = f"{self.path}/../../processed"
        self.dict_img_obj, self.dict_obj_img = self.load_dict_img_obj()
        self.dict_obj_name_id, self.dict_obj_id_name = self.load_dict_obj_name_id()
        self.dict_img_id_to_filename , self.dict_filename_to_img_id = self.load_dict_img_id_filename()
        self.preprocess_img = self.preprocess_img()
        with open("test/dmg_dict_obj_id_to_img_id.pkl", "rb") as f:
            self.dict_obj_id_to_img_id = pickle.load(f)

        


    
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
        return len(os.listdir(f"{self.path}/{self.split}2017/"))


    def __getitem__(self, index):

        file = self.files[index]
        img_id = self.dict_filename_to_img_id[file]

        img = Image.open(f"{self.path}/{self.split}2017/{file}")
        img = self.preprocess_img(img)

        win_ticket = torch.rand(1).item()

        if self.split == "train":
            if str(img_id) not in self.dict_img_obj.keys():
                label = 0
                obj_id = random.choice(list(self.dict_obj_id_name.keys()))
            elif win_ticket < 0.6:
                obj_id = random.choice(list(self.dict_img_obj[str(img_id)].keys()))
                label = self.dict_img_obj[str(img_id)][obj_id]
            elif win_ticket < 0.8:
                obj_id = random.choice(list(set(self.dict_obj_id_name.keys()) - set(self.dict_img_obj[str(img_id)].keys())))
                label = 0
            else:
                obj_id = random.choice(list(self.dict_obj_id_name.keys()))
                dmg_list = self.dict_obj_id_to_img_id[obj_id]
                if len(dmg_list) != 0:
                    img_id = random.choice(dmg_list)
                    label = 0
                    file_name = self.dict_img_id_to_filename[str(img_id)]
                    img = Image.open(f"{self.path}/{self.split}2017/{file_name}")
                    img = self.preprocess_img(img)
                else:
                    obj_id = random.choice(list(set(self.dict_obj_id_name.keys()) - set(self.dict_img_obj[str(img_id)].keys())))
                    label = 0
        else:
            if str(img_id) not in self.dict_img_obj.keys():
                label = 0
                obj_id = random.choice(list(self.dict_obj_id_name.keys()))
            elif win_ticket < 0.8:
                obj_id = random.choice(list(self.dict_img_obj[str(img_id)].keys()))
                label = self.dict_img_obj[str(img_id)][obj_id]
            else:
                obj_id = random.choice(list(set(self.dict_obj_id_name.keys()) - set(self.dict_img_obj[str(img_id)].keys())))
                label = 0

        # obj_name = self.dict_obj_id_name[obj_id]
        # text = f"How many {obj_name} are in the image? Answer with a number."
        # tokenizer = OFATokenizer.from_pretrained(self.checkpoint)

        # inputs = tokenizer([text], return_tensors="pt")

        return img , obj_id, label



if __name__ == "__main__":
    pl.seed_everything()
    data = CustomProcessorFTCocoDataset("../test/support_materials/raw/images")
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




