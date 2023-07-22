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



class CustomRetrieverFTCocoDataset(Dataset):

    def __init__(self, path, device, split="val", clip_model="RN50"):

        self.path = path
        self.split = split
        self.files = sorted(os.listdir(f"{self.path}/{self.split}2017/"))

        self.clip_base_model = clip_model
        self.device = device
        self.model, self.preprocess = clip.load(self.clip_base_model, device=self.device)
        
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
        img_id = self.dict_filename_to_img_id[file]

        img = Image.open(f"{self.path}/{self.split}2017/{file}")
        img = self.preprocess(img)

        if str(img_id) not in self.dict_img_obj.keys():
            text = "This image has no object annotated in it"
            obj_id = "0"
        else:
            text = f"A picture containing: "
            obj_in_img = list(self.dict_img_obj[str(img_id)].keys())
            # sample a random number of objects in the image
            obj_in_img = random.sample(obj_in_img, random.randint(1, len(obj_in_img)))
            # map each id in the list to its name
            strings_in_img = [self.dict_obj_id_name[obj_id] for obj_id in obj_in_img]
            # create a string with join on strings_in_img
            text += ", ".join(strings_in_img)

        text = clip.tokenize(text)

        return img , 1, text
        # return {
        #     "img": img,
        #     "id": obj_id,
        #     "text": text,
        #     "img_id": img_id
        # }

        
    
    def get_from_img_id(self, img_id):
        img_id = str(img_id)
        filename = self.dict_img_id_to_filename[img_id]
        img = Image.open(f"{self.path}/{self.split}2017/{filename}")
        img = self.preprocess(img)

        if str(img_id) not in self.dict_img_obj.keys():
            text = "This image has no object annotated in it"
            obj_id = "0"
        else:
            obj_id = random.choice(list(self.dict_img_obj[str(img_id)].keys()))
            obj_name = self.dict_obj_id_name[obj_id]
            #text = f"How many {obj_name} are in the image?"
            text = f"A picture containing a {obj_name}"

        text = clip.tokenize(text)

        return {
            "img": img,
            "id": obj_id,
            "text": text,
            "img_id": img_id,
        }




if __name__ == "__main__":
    pl.seed_everything()

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    data = CustomRetrieverFTCocoDataset("../../test/support_materials/raw/images", device=device)
    print(data[0])

    dataloader = DataLoader(data, batch_size=2, num_workers=0)
    for i in dataloader:
        # print(i)
        # print("\n\n\n\n\n\n\n\n\n\n")
        pass
    print("done")
    # model, preprocess = clip.load("RN50")

    # for i, item in enumerate(dataloader):
    #     img_ids = item[0]
    #     imgs = item[1]
    #     preprocess(imgs) 
    #     print(i, item)
    # print(data[0])




