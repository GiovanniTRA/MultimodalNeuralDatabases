from MMNDB.Data.data_retriever import CustomRetrieverCocoDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import clip
import torch
from MMNDB.Model.stopping_retriever import StoppingModel
from transformers import CLIPProcessor, CLIPModel
import logging
import open_clip
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)



class Retriever:

    # given a question you have to provide a set of embedding for every image and query

    def __init__(self, config, query, plot=False, use_grid_search=True):

        self.config = config
        self.obj_id = config.data.obj_id
        self.path = config.data.data_path
        self.device = config.device
        self.stop_algo_path = config.retriever.stop_algo_path
        self.split = config.data.split
        self.t = config.retriever.t
        self.clip_model = config.retriever.clip_model
        self.clip_library = self.config.retriever.clip_library
        self.k = config.retriever.k
        self.retriever_config = config.retriever
        self.query = query
        self.broken_classes = []
        self.grid_search_value = []
        self.results_dict = {}
        self.plot = plot
        self.use_grid_search = use_grid_search

        self.dataset = CustomRetrieverCocoDataset(
            self.path, self.split, self.clip_model, self.device, clip_library = self.clip_library
        )

        if self.retriever_config.clip_library == "openai":
            self.model, self.preprocess = clip.load(self.clip_model, device=self.device)
        elif self.retriever_config.clip_library == "huggingface_CLIP":
            self.model = CLIPModel.from_pretrained(self.clip_model).to(self.device)
            self.preprocess = CLIPProcessor.from_pretrained(self.clip_model)
        elif self.retriever_config.clip_library == "open_clip":
            model, checkpoint = self.clip_model.split("/")
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(model, pretrained=checkpoint)
            self.model = self.model.to(self.device)
        elif self.retriever_config.clip_library == "huggingface_BLIP":
            pass
        self.model.eval()


    def retrieve_documents_stopAlgo(self, obj_id):

        dataloader = DataLoader(
            self.dataset,
            batch_size=self.retriever_config.batch_size,
            num_workers=self.retriever_config.num_workers,
            shuffle=False,
        )
        obj_name = self.dataset.dict_obj_id_name[str(obj_id)]
        print("RETRIEVING IMAGES FOR OBJECT: ", obj_name)

        query = self.query.replace("{}", obj_name)
        text = clip.tokenize(query).to(self.device)
        with torch.no_grad():
            text = self.model.encode_text(text).to(self.device)
        # scores = (embeddings @ text_features.float().T)
        text = text.expand(dataloader.batch_size, -1).to(self.device)

        img_size = self.dataset[0][1].size(-1)
        stopping_model = StoppingModel(img_size + text.size(-1))
        stopping_model.load_state_dict(torch.load(self.stop_algo_path, map_location=self.device))
        stopping_model.to(self.device)
        stopping_model.eval()

        img_ids = []
        preds = []

        for item in tqdm(dataloader):

            img_ids += list(item[0].cpu().numpy())
            img_embeddings = item[1].to(self.device)

            with torch.no_grad():
                if img_embeddings.shape[0] != text.shape[0]:
                    text = text[: img_embeddings.shape[0], ...]
                pred = stopping_model(img_embeddings, text)
                pred_max = torch.argmax(pred, dim=-1)

            preds += list(pred_max.cpu().numpy())

        if len(preds) == 0:
            preds[0] = 1
        return torch.tensor(img_ids)[torch.tensor(preds) > 0]


    def retrieve_documents(self, obj_id, mode="topk"):
        dataloader = DataLoader(
            self.dataset,
            batch_size=self.retriever_config.batch_size,
            num_workers=self.retriever_config.num_workers,
        )
        obj_name = self.dataset.dict_obj_id_name[str(obj_id)]
        logger.info(f"RETRIEVING IMAGES FOR OBJECT:{obj_name}")

        query = self.query.replace("{}", obj_name)
        #query = f"A photo of a {obj_name}"
        text = clip.tokenize(query).to(self.device)
        with torch.no_grad():
            text = self.model.encode_text(text).to(self.device)

        scores = []
        img_ids = []
        for item in tqdm(dataloader):
            img_ids += list(item[0].cpu().numpy())
            img_embeddings = item[1].to(self.device)

            score = torch.nn.functional.cosine_similarity(img_embeddings, text)
            scores += list(score.cpu().numpy())

        if mode == "topk":
            logger.info("Retrieving documents with topk")
            scores = sorted(list(zip(img_ids, scores)), key=lambda x: x[1])[::-1]
            scores = torch.tensor([scores[i][0] for i in range(self.k)])
        elif mode == "threshold":
            logger.info("Retrieving documents with threshold")
            # plot ordered scores as a histogram and save it on a file
            if self.plot:
                odered_scores = sorted(list(zip(img_ids, scores)), key=lambda x: x[1])[::-1]
                plt.hist([x[1] for x in odered_scores], bins=100)
                plt.savefig(f"test/imgs/class_{obj_id}.png")
            if self.use_grid_search:
                self.grid_search(scores, img_ids, obj_id)
            scores = torch.tensor(img_ids)[torch.tensor(scores) > self.t]
        elif mode == "stop_algo":
            logger.info("Retrieving documents with stop_algo")
            scores = self.retrieve_documents_stopAlgo(obj_id)
        elif mode == "mixed":
            logger.info("Retrieving documents with mixed")
            scores1 = sorted(list(zip(img_ids, scores)), key=lambda x: x[1])[::-1]
            scores1 = torch.tensor([scores1[i][0] for i in range(10)])
            scores2 = self.retrieve_documents_stopAlgo(obj_id)
            scores = torch.tensor(list(set(scores1.tolist()).union(set(scores2.tolist()))))
        else:
            logger.error(f"Mode: {mode}")
            raise ValueError("Mode not supported")
        return scores, img_ids

    
    def grid_search(self, orginal_scores, img_ids, obj_id, num_samples=25):
        """
        Grid search for the best threshold
        """
        #sample 5 numbers between 0 and 1
        thresholds = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]

        f1_results = []
        for threshold in thresholds:
            #Clone original scores to all_scores
            all_scores = orginal_scores.copy()
            scores = torch.tensor(img_ids)[torch.tensor(all_scores) > threshold]
            res = self.compute_metrics(scores, obj_id)
            precision = res["precision"]
            recall = res["recall"]
            f1 = res["f1"]
            value_to_max = (f1)
            f1_results.append((threshold, value_to_max))

        #sort the results according to the f1+recall
        f1_results = sorted(f1_results, key=lambda x: x[1])[::-1]
        best_threshold = f1_results[0][0]
        self.grid_search_value.append(best_threshold)
        if self.results_dict.get(best_threshold) is None:
            self.results_dict[best_threshold] = 1
        else:
            self.results_dict[best_threshold] += 1
    
    
    def compute_metrics(self, scores, object_id):


        scores = list(scores.numpy())
        scores = set(list(map(str, scores)))
        relevant = set(self.dataset.dict_obj_img[str(object_id)])
        retrieved_correctly = scores.intersection(relevant)
        if len(scores) == 0:
            self.broken_classes.append(object_id)
            precision = 0
        else:
            precision = len(retrieved_correctly) / len(scores)
        if len(relevant) == 0:
            recall = 0
        else:
            recall = len(retrieved_correctly) / len(relevant)
        if precision + recall == 0:
            f1 = 0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)
        fp = len(scores) - len(retrieved_correctly)
        fn = len(relevant) - len(retrieved_correctly)
        tp = len(retrieved_correctly)
        tn = len(self.dataset) - tp - fp - fn

        return_dict = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
        }
        return return_dict


if __name__ == "__main__":

    ret = Retriever(
        obj_id=10,
        path="../../test/support_materials/raw/images",
        device="cpu",
        stop_algo_path="../test/support_materials/models/RN50_StopAlgo.pt",
    )
    retrieved = ret.retrieve_documents_threshold()
    print(ret.evaluate_precision(retrieved))
    print(ret.evaluate_recall(retrieved))
    print(ret.evaluate_f1(retrieved))
