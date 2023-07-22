import json
import os


class PreprocessRawData():

    def __init__(self, raw_path, processed_path, split="train"):

        self.raw_path = raw_path
        self.processed_path = processed_path
        self.split = split
        self.instances = self.load_file()
        os.makedirs(f"{self.processed_path}", exist_ok=True)


    def load_file(self):

        instances = json.load(open(f"{self.raw_path}/instances_{self.split}2017.json"))
        
        return instances
    

    def create_dict_img_obj(self):

        '''
        This function creates two dictionaries:
        a.  from image to obj id --> for each image containing at least one object, 
            it has a dictionary with the object id as key and the number of occurences
            as value.
        b.  from obj id to image --> for each object contained in at least an image,
            it has a dictionary with the image as key and the number of occurences
            as value.
        '''

        instances_ = self.instances["annotations"]
        img_to_obj = {}
        obj_to_img = {}

        for instance in instances_:
            img_id = instance["image_id"]
            obj_id = instance["category_id"]

            if img_id not in img_to_obj.keys():
                img_to_obj[img_id] = {}
            if obj_id not in img_to_obj[img_id].keys():
                img_to_obj[img_id][obj_id] = 1
            else:
                img_to_obj[img_id][obj_id] += 1
            
            if obj_id not in obj_to_img.keys():
                obj_to_img[obj_id] = {}
            if img_id not in obj_to_img[obj_id].keys():
                obj_to_img[obj_id][img_id] = 1
            else:
                obj_to_img[obj_id][img_id] += 1
        

        with open(f"{self.processed_path}/{self.split}_img_id_to_obj_id.json", "w") as f:
            json.dump(img_to_obj, f)
        with open(f"{self.processed_path}/{self.split}_obj_id_to_img_id.json", "w") as f:
            json.dump(obj_to_img, f)


    
    def create_dict_obj_name(self):

        instances_ = self.instances["categories"]

        obj_name_to_id = {}
        obj_id_to_name = {}

        for obj in instances_:
            name = obj["name"]
            idx = obj["id"]

            obj_name_to_id[name] = idx
            obj_id_to_name[idx] = name
        
        with open(f"{self.processed_path}/obj_name_to_id.json", "w") as f:
            json.dump(obj_name_to_id, f)
        with open(f"{self.processed_path}/obj_id_to_name.json", "w") as f:
            json.dump(obj_id_to_name, f)
        
        return obj_name_to_id, obj_id_to_name
        

    def create_dict_img_id_filename(self):

        # load json file with annotations
        instances_ = self.instances["images"]

        img_id_to_filename = {}
        filename_to_img_id = {}
        for image in instances_:
            img_filename = image["file_name"]
            img_id = image["id"]

            img_id_to_filename[img_id] = img_filename
            filename_to_img_id[img_filename] = img_id

        with open(f"{self.processed_path}/{self.split}_img_id_to_filename.json", "w") as f:
            json.dump(img_id_to_filename, f)
        with open(f"{self.processed_path}/{self.split}_filename_to_img_id.json", "w") as f:
            json.dump(filename_to_img_id, f) 

        return img_id_to_filename, filename_to_img_id



if __name__ == "__main__":
    p = PreprocessRawData("../../test/support_materials/raw/annotations", "../../test/support_materials/processed", split="val")
    # p.create_dict_img_id_filename()
    # p.create_dict_obj_name()
    p.create_dict_img_obj()

    # print(p.instances)







# print(instances)