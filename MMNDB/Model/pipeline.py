import pytorch_lightning.utilities.seed
import torch
from MMNDB.Model.retrieve import Retriever
from MMNDB.Model.process import Process
import logging
import numpy as np

logger = logging.getLogger(__name__)

class Pipeline:
    def __init__(self, config):

        self.config = config
        self.retriver_config = config.retriever
        self.processor_config = config.processor

        if self.config.task.query_type == "count":
            #self.query = "A picture of {}"
            self.query = "How many {} are in the image?"
        elif self.config.task.query_type == "max":
            self.query = "How many {} are in the image?"
        elif self.config.task.query_type == "in":
            self.query = "How many {} are in the image?"
        
        if "/" in self.config.retriever.clip_model:
            self.model_name = self.config.retriever.clip_library + "_" + self.config.retriever.clip_model.split("/")[-1].replace('.','')
        else:
            self.model_name = self.config.retriever.clip_library + "_" + self.config.retriever.clip_model


        self.retriever = Retriever(config=self.config, query=self.query)

        self.processor = Process(
            obj_id=self.config.data.obj_id,
            data_path=self.config.data.data_path,
            split=self.config.data.split,
            checkpoint=self.config.processor.checkpoint_processor,
            device=self.config.device,
            query=self.query,
            config=self.config,
            batch_size=self.config.processor.batch_size,
        )

        self.dmg_dict = {}

    def retrieve(self, obj_id):

        retrieved, _ = self.retriever.retrieve_documents(obj_id, mode=self.retriver_config.stop_algo_type)
        scores = self.retriever.compute_metrics(retrieved, obj_id)
        scores['percentage_RT'] = len(retrieved)/ len(self.retriever.dataset)
        logger.info(f"RETRIEVER DOCUMENTS: {len(retrieved)/ len(self.retriever.dataset)}")
        logger.info(f"RETRIEVER PRECISION IS: {scores['precision']}")
        logger.info(f"RETRIEVER RECALL IS: {scores['recall']}")
        logger.info(f"RETRIEVER F1 SCORE IS: {scores['f1']}")

        return retrieved, scores

    def process(self, retrieved, obj_id):
        self.processor._init_dataset(retrieved)
        scores = {}
        img_ids, answers = self.processor.process(obj_id)
        accuracy_rel, accuracy_irr, accuracy_tot = self.processor.evaluate_accuracy(img_ids, answers, obj_id)
        error_rel, error_irr, error_tot = self.processor.compute_partial_error(img_ids, answers,obj_id)
        total_error_tp, total_error_fp, total_error_fn, total_error = self.processor.compute_total_error(img_ids, answers, obj_id)

        logger.info(f"PROCESSOR EXACT MATCH ACCURACY TP: {accuracy_rel}")
        logger.info(f"PROCESSOR DELTA ERROR TP: {error_rel}")
        logger.info(f"PIPELINE TOTAL ERROR: {total_error}")
        scores['exact_match_acc_tp'] = accuracy_rel
        scores['exact_match_acc_fp'] = accuracy_irr
        scores['exact_match_acc'] = accuracy_tot
        scores['delta_error_tp'] = error_rel
        scores['delta_error_fp'] = error_irr
        scores['total_delta_error'] = error_tot
        scores['total_error'] = total_error
        scores['total_error_tp'] = total_error_tp
        scores['total_error_fp'] = total_error_fp
        scores['total_error_fn'] = total_error_fn
        
        return img_ids, answers, scores

    def pipeline(self, obj_id, mode="full_pipeline"):
        
        
        if mode == "retriever":
            _ , retriever_scores = self.retrieve(obj_id)
            return None, retriever_scores
        elif mode == "perfect_ir":
            retrieved = torch.tensor(list(map(int , self.retriever.dataset.dict_obj_img[str(obj_id)].keys())))
            img_ids, answers, processor_scores = self.process(retrieved, obj_id)
            return processor_scores, None
        elif mode == "damaging_ir":
            perfect_retrieved = torch.tensor(list(map(int , self.retriever.dataset.dict_obj_img[str(obj_id)].keys())))

            top_k_retrieved, _ = self.retriever.retrieve_documents(obj_id, mode="topk")
            # take documents that are in topk but not in retrieved
            damaging_imgs = list(set(top_k_retrieved.tolist()) - set(perfect_retrieved.tolist()))
            self.dmg_dict[obj_id] = damaging_imgs
            retrieved = torch.tensor(list(set(top_k_retrieved.tolist()).union(set(perfect_retrieved.tolist()))))

            #retrieved = torch.tensor(perfect_retrieved.tolist() + damaging_imgs)
            #processor_scores = None
            img_ids, answers, processor_scores = self.process(retrieved, obj_id)
            return processor_scores, None

        elif mode == "noisy_ir":

            n_noise = self.config.retriever.noisy_ir_noise
            retrieved = set(list(self.retriever.dataset.dict_obj_img[str(obj_id)].keys()))
            not_relevant = list(set(list(self.retriever.dataset.dict_img_obj.keys())) - retrieved)
            noisy_imgs = list(np.random.choice(not_relevant, n_noise)) if len(not_relevant) > 0 else []

            retrieved = retrieved.union(set(noisy_imgs))
            retrieved = torch.tensor(list(map(int , retrieved)))

            img_ids, answers, processor_scores = self.process(retrieved, obj_id)
            return processor_scores, None

        else:
            retrieved, retriever_scores = self.retrieve(obj_id)
            img_ids, answers, processor_score = self.process(retrieved, obj_id)
            return processor_score, retriever_scores


if __name__ == "__main__":

    pipe = Pipeline(
        obj_id=10,
        data_path="support_materials/raw/images",
        split="val",
        t=-0.5,
        clip_model="RN50",
        stop_algo_path="support_materials/models/RN50_StopAlgo.pt",
        checkpoint_processor="../../OFA-large",
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    )

    pipe.pipeline()