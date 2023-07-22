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
import logging
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
import open_clip

logger = logging.getLogger(__name__)


class CustomRetrieverCocoDataset(Dataset):

    def __init__(self, path, split="val", clip_model="RN50", device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), clip_library="openai"):
        self.path = path
        self.split = split
        self.files = sorted(os.listdir(f"{self.path}/{self.split}2017/"))
        self.clip_base_model = f"{clip_library}_{clip_model.replace('/', '_')}"
        self.clip_library = clip_library
        self.clip_model = clip_model
        self.device = device
        
        # all this function below load two arguments.
        # There is an underlying assumption that if one aldready exists
        # this applies to the other one as well.
        # it should be ok, because if it bugged then it should break soon.
        self.processed_path = f"{self.path}/processed"
        self.dict_img_obj, self.dict_obj_img = self.load_dict_img_obj()
        self.dict_obj_name_id, self.dict_obj_id_name = self.load_dict_obj_name_id()
        self.dict_img_id_to_filename , self.dict_filename_to_img_id = self.load_dict_img_id_filename()
        #self.path_save = "/home/giotra/Projects/MMNDB/test/support_materials/processed/train_embeddings_ViT-L14@336px"
        self.create_clip_embeddings()

    
    def create_clip_embeddings(self):
        if "/" in self.clip_base_model:
            new_path = self.clip_base_model.split("/")[-1].replace('.','')
            self.path_save = f"{self.processed_path}/{self.split}_embeddings_{new_path}"
        else:
            self.path_save = f"{self.processed_path}/{self.split}_embeddings_{self.clip_base_model.replace('/', '')}"
        if not os.path.isdir(self.path_save):
            os.makedirs(self.path_save, exist_ok=True)
            print(f"PROCESSINF EMBEDDINGS FOR SPLIT {self.split}")
            self.preprocess_clip_embeddings()
    

    def preprocess_clip_embeddings(self):
        if self.clip_library == "openai":
            self.model, self.preprocess = clip.load(self.clip_model, device=self.device)
        elif self.clip_library == "huggingface_CLIP":
            self.model = CLIPModel.from_pretrained(self.clip_model).to(self.device)
            self.preprocess = CLIPProcessor.from_pretrained(self.clip_model)
        elif self.clip_library == "open_clip":
            model, checkpoint = self.clip_model.split("/")
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(model,pretrained=checkpoint)

        self.model = self.model.to(self.device)
        self.model.eval()
        logger.info("Preprocessing Files not found... preprocessing Embeddings from start")


        for file in tqdm(self.files):

            with torch.no_grad():

                img_id = self.dict_filename_to_img_id[file]
                img = Image.open(f"{self.path}/{self.split}2017/{file}")
                img = self.preprocess(img)
                img = self.model.encode_image(img.unsqueeze(0).to(self.device)).cpu().squeeze(0)
                #torch.save(img, f"{self.processed_path}/{self.split}_embeddings_{self.clip_base_model.replace('/', '')}/{str(img_id)}.pt")
                torch.save(img, f"{self.path_save}/{str(img_id)}.pt")


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

        img_embedding = torch.load(f"{self.path_save}/{str(img_id)}.pt").detach()


        return img_id, img_embedding



if __name__ == "__main__":
    pl.seed_everything()
    data = CustomRetrieverCocoDataset("../../test/support_materials/raw/images")
    print(data[0])

    dataloader = DataLoader(data, batch_size=2)
    for i in dataloader:
        print(i[0])
    # model, preprocess = clip.load("RN50")

    # for i, item in enumerate(dataloader):
    #     img_ids = item[0]
    #     imgs = item[1]
    #     preprocess(imgs) 
    #     print(i, item)
    # print(data[0])




