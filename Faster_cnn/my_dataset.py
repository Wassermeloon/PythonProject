from torch.utils.data import Dataset
import os
import torch
from PIL import Image
import json
from lxml import etree
import numpy as np

class VOC2012DataSet(Dataset):

    def __init__(self,
                 voc_root,
                 tranforms,
                 train_set=True):

        super(VOC2012DataSet, self).__init__()

        self.transforms = tranforms
        self.root = os.path.join(voc_root, "VOCdevkit", "VOC2012")
        self.img_root = os.path.join(self.root, "JPEGImages")
        self.annotations_root = os.path.join(self.root, "Annotations")

        # read train.txt or val.txt file

        if train_set:
            txt_list = os.path.join(self.root, "ImageSets", "Main", "train.txt")

        else:
            txt_list = os.path.join(self.root, "ImageSets", "Main", "val.txt")
            # this is just a str, it has no content inside, it must through some other method to get the content

        with open(txt_list) as txt_read:
            self.xml_list = [os.path.join(self.annotations_root, line.strip() + ".xml")
                             for line in txt_read.readlines()]
            # we get a list of all training set xml document

        # read class dict
        try:
            json_file = open("./pascal_voc_classes.json", "r")
            self.class_dict = json.load(json_file)
        except Exception as e:
            print(e)
            exit(-1)

    def __len__(self):
        return len(self.xml_list)

    def __getitem__(self, item):
        # read xml
        xml_path = self.xml_list[item]
        with open(xml_path) as file_name:
            xml_file_name = file_name.read()
            # get the content of xml file, (complex content), but we should get the info in forms that we want
        xml = etree.fromstring(xml_file_name)
        # read xml data
        data = self.parse_xml_to_dict(xml)["annotation"]
        img_path = os.path.join(self.img_root, data["filename"])
        image = Image.open(img_path)
        print((np.asarray(image)).shape)

        if image.format != "JPEG":
            raise ValueError("Image format not JPEG")
        boxes = []
        labels = []
        iscrowed = []

        for obj in data["object"]:
            xmin = float(obj["bndbox"]["xmin"])
            ymin = float(obj["bndbox"]["ymin"])
            xmax = float(obj["bndbox"]["xmax"])
            ymax = float(obj["bndbox"]["ymax"])
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.class_dict[obj["name"]])
            iscrowed.append(int(obj["difficult"]))

        # convert everything into a tensor.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        iscrowed = torch.as_tensor(iscrowed, dtype=torch.int64)
        image_id = torch.as_tensor(item, dtype=torch.int64)
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        # write all information into a dict
        target = {"boxes": boxes,
                  "labels": labels,
                  "iscrowed": iscrowed,
                  "image_id": image_id,
                  "area": area}

        if self.transforms is not None:
            target, image = self.transforms(target, image)

        return target, image

    def get_height_und_width(self, item):
        xml_path = self.xml_list[item]
        with open(xml_path) as file:
            xml_str = file.read()

        xml = etree.fromstring(xml_str)
        data = self.parse_xml_to_dict(xml)["annotation"]
        height = int(data["size"]["height"])
        width = int(data["size"]["width"])
        return height, width

    def parse_xml_to_dict(self, xml):

        if len(xml) == 0:
            return {xml.tag: xml.text}

        result = {}
        for child in xml:
            child_result = self.parse_xml_to_dict(child)
            if child.tag != "object":
                result[child.tag] = child_result[child.tag]
            else:
                if child.tag not in result:
                    result[child.tag] = []
                result[child.tag].append(child_result[child.tag])

        return {xml.tag: result}