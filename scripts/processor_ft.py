from argparse import ArgumentParser
from transformers import ViltProcessor, ViltForQuestionAnswering
import clip
import numpy as np
from torchvision import transforms
from PIL import Image
from transformers import OFATokenizer, OFAModel
import torch
from word2number import w2n
import pytorch_lightning as pl
import wandb
from torch.utils.data import DataLoader
from MMNDB.Data.data_processor_ft import CustomProcessorFTCocoDataset
from tqdm import tqdm
from transformers.models.ofa.modeling_ofa import shift_tokens_right
import torch.nn as nn
import os


def ProcessorFT(opt):

    pl.seed_everything(42)
    wandb.init(project="MMNDB_ProcessorFT", config=opt, entity="diagml")

    save_path = f"{opt.dataset_path}/../../models"
    os.makedirs(save_path, exist_ok=True)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset_train = CustomProcessorFTCocoDataset(opt.dataset_path, "train", opt.checkpoint)
    dataloader_train = DataLoader(dataset_train, batch_size=12, drop_last=True, num_workers=28)

    dataset_val = CustomProcessorFTCocoDataset(opt.dataset_path, "val", opt.checkpoint)
    dataloader_val = DataLoader(dataset_val, batch_size=12, num_workers=28, drop_last=True, shuffle=False)


    tokenizer = OFATokenizer.from_pretrained(opt.checkpoint)
    model = OFAModel.from_pretrained(opt.checkpoint, use_cache=True).to(device)

    print("MODEL'S PARAMETERS: ", sum(p.numel() for p in model.parameters()))

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

    criterion = nn.CrossEntropyLoss()
    model.train()

    for epoch in range(opt.num_epochs):
        epoch_loss = []
        model.train()
        optimizer.zero_grad()
        for i, item in enumerate(tqdm(dataloader_train)):


            images = item[0].to(device)
            obj_ids = item[1]
            labels = item[2]  # .to(device)

            obj_names = [dataset_train.dict_obj_id_name[obj_id] for obj_id in obj_ids]
            text = [f"How many {obj_name} are in the image? Answer with a number." for obj_name in obj_names]
            inputs = tokenizer(text, return_tensors="pt", padding=True)

            labels_token = tokenizer(list(map(str, labels.tolist())), return_tensors="pt", padding=True).to(device)

            preds = model(input_ids=inputs.input_ids.to(device), decoder_input_ids = labels_token.input_ids[:, :-1],
                          attention_mask=labels_token.attention_mask[:, :-1], patch_images=images)

            preds = preds.logits.reshape(-1, preds.logits.size(-1))
            loss = criterion(preds, labels_token.input_ids[:, 1:].reshape(-1))
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
            epoch_loss.append(loss.item())

            if i % 20 == 0:
                wandb.log({"train_step_loss": loss.item()})


        
        wandb.log({"train_epoch_loss": np.mean(epoch_loss)})

        model.eval()
        with torch.no_grad():
            accuracies = []
            losses = []
            for item in tqdm(dataloader_val):

                images = item[0].to(device)
                obj_ids = item[1]
                labels = item[2]  # .to(device)

                # this could be done in the dataset, but it needs a custom collate
                obj_names = [dataset_val.dict_obj_id_name[obj_id] for obj_id in obj_ids]
                text = [f"How many {obj_name} are in the image? Answer with a number." for obj_name in obj_names]
                inputs = tokenizer(text, return_tensors="pt", padding=True)

                labels_token = tokenizer(list(map(str, labels.tolist())), return_tensors="pt", padding=True).to(device)

                preds = model(input_ids=inputs.input_ids.to(device), decoder_input_ids = labels_token.input_ids[:, :-1].to(device),
                          attention_mask=labels_token.attention_mask[:, :-1].to(device), patch_images=images)

                losses.append(criterion(preds.logits.reshape(-1, preds.logits.size(-1)), labels_token.input_ids[:, 1:].reshape(-1)).item())


                accuracies += list((labels_token.input_ids[:, 1] == torch.argmax(preds.logits, dim=2)[:, 0]).float().cpu().numpy())

            
            print(f"ACCURACY: {np.mean(accuracies)}")
            print(f"VALIDATION LOSS: {np.mean(losses)}")
            wandb.log({"accuracy": np.mean(accuracies)})
            wandb.log({"val_loss": np.mean(losses)})
        
        save_path = f"{opt.dataset_path}/../../models"
        model.save_pretrained(f"{save_path}/OFA-large_epoch_{epoch}_lr_{opt.lr}_FT_dmg.bin")




if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument("--dataset_path", default="support_materials/raw/images", type=str)
    parser.add_argument("--split", default="train", type=str)
    parser.add_argument("--checkpoint", default="OFA-Sys/ofa-large", type=str)
    parser.add_argument("--num_epochs", default=5, type=int)
    parser.add_argument("--lr", default=0.00005, type=float)
    parser.add_argument("--weightCE", default=0.51, type=float)

    args = parser.parse_args()

    ProcessorFT(args)