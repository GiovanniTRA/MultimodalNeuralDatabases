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
from tqdm import tqdm


class CustomStopAlgoCocoDataset(Dataset):

    def __init__(self, path, split="val", clip_model="RN50", prob=0.5, device=torch.device("cuda:3" if torch.cuda.is_available() else "cpu")):
        self.path = path
        self.split = split
        self.files = sorted(os.listdir(f"{self.path}/{self.split}2017/"))
        self.clip_base_model = clip_model
        self.prob = prob # probability of label=1
        self.device = device
        

        
        # all this function below load two arguments.
        # There is an underlying assumption that if one aldready exists
        # this applies to the other one as well.
        # it should be ok, because if it bugged then it should break soon.
        self.processed_path = f"{self.path}/../../processed"
        self.dict_img_obj, self.dict_obj_img = self.load_dict_img_obj()
        self.dict_obj_name_id, self.dict_obj_id_name = self.load_dict_obj_name_id()
        self.dict_img_id_to_filename , self.dict_filename_to_img_id = self.load_dict_img_id_filename()
        self.create_clip_embeddings()

    
    def create_clip_embeddings(self):

        if not os.path.isdir(f"{self.processed_path}/{self.split}_embeddings_{self.clip_base_model.replace('/', '')}"):
            os.makedirs(f"{self.processed_path}/{self.split}_embeddings_{self.clip_base_model.replace('/', '')}", exist_ok=True)
            print(f"PROCESSING EMBEDDINGS FOR SPLIT {self.split}")
            self.preprocess_clip_embeddings()
    

    def preprocess_clip_embeddings(self):

        self.model, self.preprocess = clip.load(self.clip_base_model, device=self.device)
        self.model.eval()

        for file in tqdm(self.files):

            with torch.no_grad():

                img_id = self.dict_filename_to_img_id[file]
                img = Image.open(f"{self.path}/{self.split}2017/{file}")
                img = self.preprocess(img)
                self.model = self.model.to(self.device)
                img = self.model.encode_image(img.unsqueeze(0).to(self.device)).cpu().squeeze(0)

                torch.save(img, f"{self.processed_path}/{self.split}_embeddings_{self.clip_base_model.replace('/', '')}/{str(img_id)}.pt")            


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

        img_embedding = torch.load(f"{self.processed_path}/{self.split}_embeddings_{self.clip_base_model.replace('/', '')}/{str(img_id)}.pt").detach()

        if str(img_id) not in self.dict_img_obj.keys():
            label = 0
            obj_id = random.choice(list(self.dict_obj_id_name.keys()))
        elif torch.rand(1) < self.prob:
            obj_id = random.choice(list(self.dict_img_obj[str(img_id)].keys()))
            label = 1
        else:
            obj_id = random.choice(list(set(self.dict_obj_id_name.keys()) - set(self.dict_img_obj[str(img_id)].keys())))
            label = 0
        
        obj_name = self.dict_obj_id_name[obj_id]
        query = f"how many {obj_name} are in this picture"
        query = clip.tokenize(query).squeeze(0)

        return img_embedding, query, label



if __name__ == "__main__":
    pl.seed_everything()
    data = CustomStopAlgoCocoDataset("../test/support_materials/raw/images")
    print(data[0])

    dataloader = DataLoader(data, batch_size=2)
    for i in dataloader:
        print(i[2])
    # model, preprocess = clip.load("RN50")

    # for i, item in enumerate(dataloader):
    #     img_ids = item[0]
    #     imgs = item[1]
    #     preprocess(imgs) 
    #     print(i, item)
    # print(data[0])




