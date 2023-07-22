from argparse import ArgumentParser
import clip
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from MMNDB.Data.data_retriever_ft import CustomRetrieverFTCocoDataset
import numpy as np
import wandb
import os
import pytorch_lightning as pl
import torch.nn as nn
import hydra


import torch
from torch.utils.data import Dataset, DataLoader, BatchSampler, Sampler


class UniqueIDSampler(Sampler):
    def __init__(self, data_source):
        self.data_source = data_source
        self.unique_ids = set(sample["id"] for sample in data_source)
    
    def __iter__(self):
        for unique_id in self.unique_ids:
            indices = [idx for idx, sample in enumerate(self.data_source) if sample["id"] == unique_id]
            yield from indices
    
    def __len__(self):
        return len(self.data_source)


class DifferentIDBatchSampler(BatchSampler):
    def __init__(self, data_source, batch_size=4, drop_last=False):
        self.data_source = data_source
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.indices = list(range(len(self.data_source)))
        self.index_groups = self._create_index_groups()
    
    def __iter__(self):
        #for group in self.index_groups:
        return iter(self.index_groups[1])
    
    def __len__(self):
        return len(self.index_groups)
    
    def _create_index_groups(self):
        id_to_indices = {}
        for idx in self.indices:
            sample_id = self.data_source[idx]["id"]
            if sample_id not in id_to_indices:
                id_to_indices[sample_id] = []
            id_to_indices[sample_id].append(idx)
        groups = []
        group = []
        for sample_indices in id_to_indices.values():
            if len(group) + len(sample_indices) > self.batch_size:
                groups.append(group)
                group = sample_indices
            else:
                group += sample_indices
        if group:
            groups.append(group)
        return groups

def RetrieverFT(opt):

    pl.seed_everything(42)
    wandb.init(project="MMNDB_RetrieverFT", config=opt, entity="diagml")
    
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    dataset_train = CustomRetrieverFTCocoDataset(opt.dataset_path, device=device, split="train", clip_model=opt.clip_model)
    dataloader_train = DataLoader(dataset_train, batch_size=22, drop_last=True, num_workers=8)

    dataset_val = CustomRetrieverFTCocoDataset(opt.dataset_path, device=device, split="val", clip_model=opt.clip_model)
    dataloader_val = DataLoader(dataset_val, batch_size=22, drop_last=True, num_workers=8)


    model, _ = clip.load(opt.clip_model, device=device)
    model = model.float()
    print("MODEL PARAMETERS ARE: ", sum(p.numel() for p in model.parameters()))

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    loss_img = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(opt.num_epochs):
        epoch_loss = []
        model.train()
        for item in tqdm(dataloader_train):

            optimizer.zero_grad()

            images = item[0].to(device)
            # obj_ids = item[1]
            texts = item[2].to(device)

            logits_per_image, logits_per_text = model(images, texts.squeeze(-2))
            ground_truth = torch.arange(len(images), dtype=torch.long, device=device)

            total_loss = loss_img(logits_per_image,ground_truth)



            total_loss.backward()
            optimizer.step()

            epoch_loss.append(total_loss.item())
            # print(loss.item())
        
        wandb.log({"train_epoch_loss": np.mean(epoch_loss)})

        model.eval()
        with torch.no_grad():
            accuracies_img = []
            accuracies_text = []
            # preds_ = []
            # labels_ = []
            for item in tqdm(dataloader_val):

                images = item[0].to(device)
                # obj_ids = item[1]
                texts = item[2].to(device)

                logits_per_image, logits_per_text = model(images, texts.squeeze(-2))
                ground_truth = torch.arange(len(images), dtype=torch.long, device=device)

                acc_img = torch.argmax(logits_per_image, dim=-1)
                acc_text = torch.argmax(logits_per_text, dim=-1)

                accuracies_img += list((acc_img == ground_truth).float().cpu().numpy())
                accuracies_text += list((acc_text == ground_truth).float().cpu().numpy())
                # preds_ += list(preds.float().cpu().numpy())
                # labels_ += list(labels.float().cpu().numpy())
            
            print(f"ACCURACY IMAGE: {np.mean(accuracies_img)}")
            print(f"ACCURACY TEXT: {np.mean(accuracies_text)}")
            wandb.log({"accuracy image": np.mean(accuracies_img)})
            wandb.log({"accuracy text": np.mean(accuracies_text)})
        
        save_path = opt.model_path
        os.makedirs(save_path, exist_ok=True)
        torch.save(model.state_dict(), f"{save_path}/{opt.clip_model.replace('/', '')}_RetrieverFT.pt")



if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument("--dataset_path", default="support_materials/raw/images", type=str)
    parser.add_argument("--model_path", default="models", type=str)
    parser.add_argument("--split", default="train", type=str)
    parser.add_argument("--dataset_tensorize", default=False, type=bool)
    # clip available models: ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']
    parser.add_argument("--clip_model", default="ViT-L/14@336px", type=str)
    parser.add_argument("--num_epochs", default=1000, type=int)
    parser.add_argument("--lr", default=0.00001, type=float)
    parser.add_argument("--weightCE", default=0.51, type=float)

    args = parser.parse_args()

    RetrieverFT(args)