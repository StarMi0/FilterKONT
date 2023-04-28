import numpy as np
import pandas as pd
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer
from transformers import DataCollatorForSeq2Seq
import evaluate
import argparse
from transformers import EvalPrediction
import torch
from transformers import AutoModelForSemanticSegmentation, TrainingArguments, Trainer
from transformers import TrainingArguments, Trainer
from datasets import Dataset
import json
import sys
from tqdm.auto import tqdm
from transformers import AutoImageProcessor
from torch import nn
from torchvision.transforms import ColorJitter
from PIL import Image

# получение параметров для инференса
def get_args_parser():
    parser = argparse.ArgumentParser('Image segmentation', add_help=False)
    parser.add_argument('--model_name', default="segformer-b5/checkpoint-15000", type=str)
    parser.add_argument('--image_path', default="a1.jpg", type=str)
    parser.add_argument('--save_path', default="a1_res.png", type=str)
    return parser

# инициализация класса модели для инференса
class SegmentationModel:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.image_processor = AutoImageProcessor.from_pretrained("nvidia/mit-b5", size={'height': 960, 'width': 960}, reduce_labels=True)
        self.model = AutoModelForSemanticSegmentation.from_pretrained(args.model_name).to(self.device)
    
    def __call__(self, image):
        inputs = self.image_processor(image, return_tensors="pt")
        inputs = inputs.pixel_values.to(self.device)
        outputs = self.model(pixel_values=inputs)
        logits = outputs.logits.cpu()
        upsampled_logits = nn.functional.interpolate(
            logits,
            size=image.size[::-1],
            mode="bilinear",
            align_corners=False,
        )
        pred_seg = (1-upsampled_logits.argmax(dim=1)[0].numpy())*255
        pred_seg = Image.fromarray(pred_seg.astype('uint8'))
        return pred_seg
    
# функция для обработки одной картинки
def run_image():
    img = Image.open(args.image_path)
    res = model(img)
    res.save(args.save_path)
        
# запуск скрипта
if __name__ == '__main__':
    parser = argparse.ArgumentParser('Image segmentation', parents=[get_args_parser()])
    args = parser.parse_args()
    model = SegmentationModel()
    run_image()
