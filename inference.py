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
import cv2
from PIL import Image

def get_args_parser():
    parser = argparse.ArgumentParser('Image segmentation', add_help=False)
    parser.add_argument('--model_name', default="segformer-b5/checkpoint-800", type=str)
    parser.add_argument('--data_path', default='scene_parse_150', type=str)
    parser.add_argument('--json_output', default='result.json', type=str)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--metric', default="mean-iou", type=str) 
    parser.add_argument('--download_from_hf', default='True', type=str) #True, если надо скачать, иначе False
    return parser

def val_transforms(example_batch):
    images = [x for x in example_batch["image"]]
    labels = [x for x in example_batch["annotation"]]
    inputs = image_processor(images, labels, return_tensors="pt")
    return inputs


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Image segmentation', parents=[get_args_parser()])
    args = parser.parse_args()
    
    if args.download_from_hf != 'True':
        imgs = os.listdir(args.data_path)
        data = []
        for img_path in imgs:
            cur = {'image': Image.open(args.data_path+'/'+img_path)}
            data.append(cur)
    else:
        data = load_dataset(args.data_path)
        data = data['test']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image_processor = AutoImageProcessor.from_pretrained("nvidia/mit-b5", reduce_labels=True)
    model = AutoModelForSemanticSegmentation.from_pretrained(args.model_name).to(device)
    predictions = dict()
    for ids in tqdm(range(len(data))):
        try:
            inputs = image_processor(data[ids]['image'], return_tensors="pt")
        except:
            img = data[ids]['image']
            img = np.dstack((img, img, img))
            inputs = image_processor(img, return_tensors="pt")
        inputs = inputs.pixel_values.to(device)
        image = data[ids]['image']
        outputs = model(pixel_values=inputs)
        logits = outputs.logits.cpu()
        upsampled_logits = nn.functional.interpolate(
            logits,
            size=image.size[::-1],
            mode="bilinear",
            align_corners=False,
        )
        pred_seg = Image.fromarray(upsampled_logits.argmax(dim=1)[0].numpy()*255)
        predictions[ids] = {ids: pred_seg}
    with open(args.json_output, 'w') as f:
        json.dump(predictions, f)
