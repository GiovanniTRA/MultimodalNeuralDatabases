from PIL import Image
from MMNDB.Data.data_processor import CustomProcessorCocoDataset
import torch
from torch.utils.data import DataLoader
from transformers import OFATokenizer, OFAModel
from tqdm import tqdm
from word2number import w2n



class Process():

    def __init__(self, obj_id, data_path, split, checkpoint, device, query, config, batch_size=8):

        self.obj_id = obj_id
        self.data_path = data_path
        self.split = split
        self.checkpoint = checkpoint
        self.device = device
        self.batch_size = batch_size
        self.query = query
        self.config = config

        self.query_type = self.config.task.query_type

        self.tokenizer = OFATokenizer.from_pretrained(self.checkpoint)
        self.model = OFAModel.from_pretrained(self.checkpoint, use_cache=True).to(self.device)
        self.model.eval()


    def _init_dataset(self, retrieved):
        self.dataset = CustomProcessorCocoDataset(path=self.data_path, retrieved=retrieved, split=self.split)

    def process(self, obj_id):

        dataloader = DataLoader(self.dataset, batch_size=self.batch_size, num_workers=28, shuffle=False)

        obj_name = self.dataset.dict_obj_id_name[str(obj_id)]
        print("PROCESSING IMAGES FOR OBJECT: ", obj_name)
        exact_query = self.query.replace("{}", obj_name) + " Answer with a number."

        text = [exact_query for _ in range(dataloader.batch_size)]
        inputs = self.tokenizer(text, return_tensors="pt")
        inp = inputs.input_ids.to(self.device)
        att_mask = inputs.attention_mask.to(self.device)

        answers = []
        img_ids = []
        for item in tqdm(dataloader):

            img_ids += list(item[0].cpu().numpy())
            images = item[1].to(self.device)

            generator = self.model.generate(inputs=inp[:images.size(0)],
             attention_mask=att_mask[:images.size(0)], 
             patch_images=images, num_beams=5, no_repeat_ngram_size=3)
            answers += self.tokenizer.batch_decode(generator, skip_special_tokens=True)

        return img_ids, answers
    

    def process_raw_answer(self, answers, resort_to=0):

        answers_processed = []
        for answer in answers:
            try:
                answer = w2n.word_to_num(answer.strip())
            except:
                answer = resort_to    
            answers_processed.append(answer)

        return answers_processed

    def evaluate_accuracy_count(self, img_ids, answers, obj_id):

        answers_processed = self.process_raw_answer(answers)

        # collect gt
        ground_truth = []
        # for every image in retrieved
        for img_id in img_ids:
            img_id = str(img_id.item())
            # check if image has known objects in it
            if img_id in self.dataset.dict_img_obj.keys():
                obj_id = str(obj_id)
                # check if object of interest is in image
                if obj_id in self.dataset.dict_img_obj[img_id].keys():
                    ground_truth.append(self.dataset.dict_img_obj[img_id][obj_id])
                else:
                    ground_truth.append(0)
            else:
                ground_truth.append(0)


        # this tells you how many images you got EXACTLY right
        total_match_accuracy = (torch.tensor(answers_processed) == torch.tensor(ground_truth))
        match_accuracy_relevant = total_match_accuracy[torch.tensor(ground_truth) > 0].float().mean().item()
        match_accuracy_irrelevant = total_match_accuracy[torch.tensor(ground_truth) == 0].float().mean().item()
        n1 = total_match_accuracy[torch.tensor(ground_truth) > 0].shape[0]
        n2 = total_match_accuracy[torch.tensor(ground_truth) == 0].shape[0]
        
        if n1 == 0 and n2 == 0:
            total_match_accuracy = 0
        elif n1 == 0:
            total_match_accuracy = match_accuracy_irrelevant
        elif n2 == 0:
            total_match_accuracy = match_accuracy_relevant
        else:
            total_match_accuracy = (n1 * match_accuracy_relevant + n2 * match_accuracy_irrelevant) / (n1 + n2)

        return match_accuracy_relevant, match_accuracy_irrelevant, total_match_accuracy

    def compute_partial_error_count(self, img_ids, answers, obj_id):
        

        answers_processed = self.process_raw_answer(answers)

        ground_truth = []
        for img_id in img_ids:
            img_id = str(img_id.item())
            if img_id in self.dataset.dict_img_obj.keys():
                obj_id = str(obj_id)
                if obj_id in self.dataset.dict_img_obj[img_id].keys():
                    ground_truth.append(self.dataset.dict_img_obj[img_id][obj_id])
                else:
                    ground_truth.append(0)
            else:
                ground_truth.append(0)

        total_delta_error = (torch.tensor(answers_processed) - torch.tensor(ground_truth)) ** 2
        delta_error_relevant = torch.sqrt(total_delta_error[torch.tensor(ground_truth) > 0].float().mean()).item()
        delta_error_irrelevant = torch.sqrt(total_delta_error[torch.tensor(ground_truth) == 0].float().mean()).item()
        total_delta_error = torch.sqrt(total_delta_error.float().mean()).item()
        return delta_error_relevant, delta_error_irrelevant, total_delta_error

    def compute_total_error(self, img_ids, answers, obj_id):
        if self.query_type == "count":
            return self.compute_total_error_count(img_ids, answers, obj_id)
        elif self.query_type == "in":
            return 1, 1, 1, self.compute_total_error_in(img_ids, answers, obj_id)
        elif self.query_type == "max":
            return 1, 1, 1, self.compute_total_error_max(img_ids, answers, obj_id)
    
    def compute_partial_error(self, img_ids, answers, obj_id):
        if self.query_type == "count":
            return self.compute_partial_error_count(img_ids, answers, obj_id)
        elif self.query_type == "in":
            return 1, 1, 1
        elif self.query_type == "max":
            return 1, 1, self.compute_partial_error_max(img_ids, answers, obj_id)

    def evaluate_accuracy(self, img_ids, answers, obj_id):
        if self.query_type == "count":
            return self.evaluate_accuracy_count(img_ids, answers, obj_id)
        elif self.query_type == "in":
            return  1, 1, self.evaluate_accuracy_in(img_ids, answers, obj_id)
        elif self.query_type == "max":
            return  1, 1, self.evaluate_accuracy_max(img_ids, answers, obj_id)
        

    def compute_total_error_count(self, img_ids, answers, obj_id):

        answers_processed = self.process_raw_answer(answers)

        ground_truth = []
        for img_id in img_ids:
            img_id = str(img_id.item())
            if img_id in self.dataset.dict_img_obj.keys():
                obj_id = str(obj_id)
                if obj_id in self.dataset.dict_img_obj[img_id].keys():
                    ground_truth.append(self.dataset.dict_img_obj[img_id][obj_id])
                else:
                    ground_truth.append(0)
            else:
                ground_truth.append(0)
        
        perfect_ir = set(list(self.dataset.dict_obj_img[str(obj_id)].keys()))
        not_retrieved = list(perfect_ir - set(list(map(str, img_ids))))
        gt_not_retrieved = [self.dataset.dict_img_obj[str(img_id)][str(obj_id)] for img_id in not_retrieved]

        gt_perfect_ir = [self.dataset.dict_img_obj[str(img_id)][str(obj_id)] for img_id in perfect_ir]

        sum_gt = sum(gt_perfect_ir)

        answers_processed = torch.tensor(answers_processed)
        ground_truth = torch.tensor(ground_truth)

        total_error_tp = torch.abs(torch.sum(answers_processed[ground_truth > 0].float()) - torch.sum(ground_truth[ground_truth > 0].float())).item()
        total_error_fp = torch.abs(torch.sum(answers_processed[ground_truth == 0].float()) - torch.sum(ground_truth[ground_truth == 0].float())).item()
        total_error_fn = sum(gt_not_retrieved)

        total_error = total_error_tp + total_error_fp + total_error_fn
        total_error_tp = total_error_tp / sum_gt
        total_error_fp = total_error_fp / sum_gt
        total_error_fn = total_error_fn / sum_gt
        total_error = total_error / sum_gt

        return total_error_tp, total_error_fp, total_error_fn, total_error


    ################### MAX ############################

    def evaluate_accuracy_max(self, img_ids, answers, obj_id):

        """
        This function computes the exact match accuracy for the max types of query.
        It specifies if the processor is able to find the maximum among the retrieved documents
        ps. among the retrieved, which might not include the true maximum.

        It return 1 for success, 0 for failure
        """

        answers_processed = self.process_raw_answer(answers)

        ground_truth = []
        for img_id in img_ids:
            img_id = str(img_id.item())
            if img_id in self.dataset.dict_img_obj.keys():
                obj_id = str(obj_id)
                if obj_id in self.dataset.dict_img_obj[img_id].keys():
                    ground_truth.append(self.dataset.dict_img_obj[img_id][obj_id])
                else:
                    ground_truth.append(0)
            else:
                ground_truth.append(0)
        
        id_max_answ = torch.argmax(torch.tensor(answers_processed))
        max_answ = torch.max(torch.tensor(answers_processed))

        id_max_gt = torch.argmax(torch.tensor(ground_truth))
        max_gt = torch.max(torch.tensor(ground_truth))

        match = 0
        if (id_max_answ == id_max_gt) or (max_answ == max_gt):
            match = 1

        return match

    
    def compute_partial_error_max(self, img_ids, answers, obj_id):

        """
        This function computes the partial error for the max types of query.
        It specifies how far numerically the processor is from the max among the retrieved documents
        ps. among the retrieved, which might not include the true maximum.

        It return the numerical difference (energy), always positive
        """

        answers_processed = self.process_raw_answer(answers)

        ground_truth = []
        for img_id in img_ids:
            img_id = str(img_id.item())
            if img_id in self.dataset.dict_img_obj.keys():
                obj_id = str(obj_id)
                if obj_id in self.dataset.dict_img_obj[img_id].keys():
                    ground_truth.append(self.dataset.dict_img_obj[img_id][obj_id])
                else:
                    ground_truth.append(0)
            else:
                ground_truth.append(0)
        
        max_answ = torch.max(torch.tensor(answers_processed))
        max_gt = torch.max(torch.tensor(ground_truth))

        return torch.abs(max_answ - max_gt)

    
    def compute_total_error_max(self, img_ids, answers, obj_id):

        """
        This function computes the total error for the max types of query.
        It specifies how far numerically the processor is from the true max.
        ps. among all documents, which include the true maximum.

        It return the numerical difference (energy), always positive
        """

        answers_processed = self.process_raw_answer(answers)

        ground_truth = []
        for img_id in img_ids:
            img_id = str(img_id.item())
            if img_id in self.dataset.dict_img_obj.keys():
                obj_id = str(obj_id)
                if obj_id in self.dataset.dict_img_obj[img_id].keys():
                    ground_truth.append(self.dataset.dict_img_obj[img_id][obj_id])
                else:
                    ground_truth.append(0)
            else:
                ground_truth.append(0)
        
        perfect_ir = set(list(self.dataset.dict_obj_img[str(obj_id)].keys()))
        not_retrieved = list(perfect_ir - set(list(map(str, img_ids))))
        gt_not_retrieved = [self.dataset.dict_img_obj[str(img_id)][str(obj_id)] for img_id in not_retrieved]

        answers_processed = torch.tensor(answers_processed + [0 for _ in range(len(not_retrieved))])
        ground_truth = torch.tensor(ground_truth + gt_not_retrieved)

        max_answ = torch.max(answers_processed)
        max_gt = torch.max(ground_truth)

        return torch.abs(max_answ - max_gt) / torch.abs(max_gt)

    ################### IN  ############################

    def evaluate_accuracy_in(self, img_ids, answers, obj_id):

        """
        This function evaluate the exact match accuracy among retrieved document for IN queries.
        It return a vector a binary vector.
        It considers only the retrieved documents
        """

        answers_processed = self.process_raw_answer(answers)

        ground_truth = []
        for img_id in img_ids:
            img_id = str(img_id.item())
            if img_id in self.dataset.dict_img_obj.keys():
                obj_id = str(obj_id)
                if obj_id in self.dataset.dict_img_obj[img_id].keys():
                    ground_truth.append(self.dataset.dict_img_obj[img_id][obj_id])
                else:
                    ground_truth.append(0)
            else:
                ground_truth.append(0)
        
        answers_processed = (torch.tensor(answers_processed) > 0)
        ground_truth = (torch.tensor(ground_truth) > 0)

        return (answers_processed == ground_truth).float().mean().item()
    
    def compute_total_error_in(self, img_ids, answers, obj_id):

        """
        This function computes total error for IN queries.
        It return a binary vector, the difference between found pictures and the true amount
        It considers all documents.
        """

        answers_processed = self.process_raw_answer(answers)

        ground_truth = []
        for img_id in img_ids:
            img_id = str(img_id.item())
            if img_id in self.dataset.dict_img_obj.keys():
                obj_id = str(obj_id)
                if obj_id in self.dataset.dict_img_obj[img_id].keys():
                    ground_truth.append(self.dataset.dict_img_obj[img_id][obj_id])
                else:
                    ground_truth.append(0)
            else:
                ground_truth.append(0)
        
        perfect_ir = set(list(self.dataset.dict_obj_img[str(obj_id)].keys()))
        not_retrieved = list(perfect_ir - set(list(map(str, img_ids))))
        gt_not_retrieved = [self.dataset.dict_img_obj[str(img_id)][str(obj_id)] for img_id in not_retrieved]

        answers_processed = torch.tensor(answers_processed + [0 for _ in range(len(not_retrieved))])
        ground_truth = torch.tensor(ground_truth + gt_not_retrieved)

        answers_processed = (answers_processed > 0)
        ground_truth = (ground_truth > 0)

        return torch.abs(answers_processed.sum() - ground_truth.sum()).item() / ground_truth.sum().item()

