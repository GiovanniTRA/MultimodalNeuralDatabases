from argparse import ArgumentParser
from MMNDB.Data.data import CustomCocoDataset
import clip
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from MMNDB.Data.data_stop_algo import CustomStopAlgoCocoDataset
from MMNDB.Model.stopping_retriever import StoppingModel
import numpy as np
import wandb
from sklearn.metrics import confusion_matrix
import os
import pytorch_lightning as pl



def stopAlgo(opt):

    pl.seed_everything(42)
    wandb.init(project="MMNDB_StopAlgo", config=opt, entity="diagml")
    

    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

    clip_model, _ = clip.load(opt.clip_model, device=device)
    clip_model.eval()

    dataset_train = CustomStopAlgoCocoDataset(opt.dataset_path, "train", opt.clip_model)
    dataloader_train = DataLoader(dataset_train, batch_size=32, num_workers=16)

    dataset_val = CustomStopAlgoCocoDataset(opt.dataset_path, "val", opt.clip_model)
    dataloader_val = DataLoader(dataset_val, batch_size=32)

    img_size = dataset_train[0][0].size(-1)
    text_len = dataset_train[0][1].size(-1)
    
    model = StoppingModel(img_size * 2).to(device)
    print(sum(p.numel() for p in model.parameters()))
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    criterion = torch.nn.NLLLoss(weight=torch.tensor([1 - opt.weightCE, opt.weightCE]).float().to(device))
    model.train()

    for epoch in range(opt.num_epochs):
        epoch_loss = []
        model.train()
        for item in tqdm(dataloader_train):

            optimizer.zero_grad()

            img_embeddings = item[0].to(device)
            queries = item[1].to(device)
            labels = item[2].to(device)

            with torch.no_grad():
                queries = clip_model.encode_text(queries)

            preds = model(img_embeddings, queries)

            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()

            epoch_loss.append(loss.item())
            # print(loss.item())
        
        wandb.log({"train_epoch_loss": np.mean(epoch_loss)})

        model.eval()
        with torch.no_grad():
            accuracies = []
            preds_ = []
            labels_ = []
            for item in tqdm(dataloader_val):

                img_embeddings = item[0].to(device)
                queries = item[1].to(device)
                labels = item[2].to(device)

                queries = clip_model.encode_text(queries)

                preds = model(img_embeddings, queries)
                preds = torch.argmax(preds, dim=-1)


                accuracies += list((preds == labels).float().cpu().numpy())
                preds_ += list(preds.float().cpu().numpy())
                labels_ += list(labels.float().cpu().numpy())
            
            print(f"ACCURACY: {np.mean(accuracies)}")
            print(f"PREDICTIONS MEAN: {np.mean(preds_)}")
            print(f"LABELS MEAN: {np.mean(labels_)}")
            wandb.log({"accuracy": np.mean(accuracies)})
            wandb.log({"predictions mean": np.mean(preds_)})
            wandb.log({"labels mean": np.mean(labels_)})

            wandb.log({"conf_mat" : wandb.plot.confusion_matrix(probs=None,
                                    y_true=np.array(labels_), preds=np.array(preds_),
                                    class_names=["no_relevant", "relevant"])})
            print(f"CONFUSION MATRIX:\n {confusion_matrix(np.array(labels_), np.array(preds_), labels=[0, 1])}")
        
        save_path = f"{opt.dataset_path}/../../models"
        os.makedirs(save_path, exist_ok=True)
        torch.save(model.state_dict(), f"{save_path}/{opt.clip_model.replace('/', '')}_StopAlgo.pt")



if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument("--dataset_path", default="support_materials/raw/images", type=str)
    parser.add_argument("--split", default="val", type=str)
    parser.add_argument("--dataset_tensorize", default=False, type=bool)
    # clip available models: ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']
    parser.add_argument("--clip_model", default="RN50", type=str)
    parser.add_argument("--num_epochs", default=1, type=int)
    parser.add_argument("--lr", default=0.0001, type=float)
    parser.add_argument("--weightCE", default=0.50, type=float)

    args = parser.parse_args()

    stopAlgo(args)