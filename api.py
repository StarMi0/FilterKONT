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
import time, os, werkzeug, zipfile, cv2, shutil
from flask import Flask, render_template, make_response, request, Blueprint, send_file
from flask_restx import Api, Resource, fields, reqparse
from werkzeug.utils import secure_filename
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# допустимые разсширения имен файлов с изображениями
ALLOWED_EXTENSIONS = {'zip', 'rar', '7z', 'tar', 'gz'}

# имя каталога для загрузки изображений
UPLOAD_FOLDER = 'inputs/'
RESULT_FOLDER = 'result/'


def allowed_file(file_name):
    """
    Функция проверки расширения файла
    """
    return '.' in file_name and file_name.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def add_affix(filename, affix='_CENTR'):
    name, extension = os.path.splitext(filename)
    return name + affix + '.png'


def zip_folder(name):
    print(name)
    zip_name = name + '.zip'
    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zip_ref:
        for folder_name, subfolders, filenames in os.walk(name):
            for filename in filenames:
                file_path = os.path.join(folder_name, filename)
                zip_ref.write(file_path, arcname=os.path.relpath(file_path, name))

    zip_ref.close()


# создаем WSGI приложение
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

# создаем API-сервер
api = Api(app)

# создаем парсер API-запросов
parser = reqparse.RequestParser()
parser.add_argument('image_file', type=werkzeug.datastructures.FileStorage, help='Binary Image in png format (zip, rar, 7z, tar, gz)', location='files', required=True)
parser.add_argument('scale', type=werkzeug.datastructures.Accept, required=True)

# инициализация класса модели для инференса
class SegmentationModel:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.image_processor = AutoImageProcessor.from_pretrained("nvidia/mit-b5", size={'height': 1008, 'width': 1344},
                                                                  reduce_labels=True)
        self.model = AutoModelForSemanticSegmentation.from_pretrained("segformer-b5/checkpoint-13000").to(self.device)
    
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


model = SegmentationModel()


@api.route('/images', methods=['GET', 'POST'])
@api.produces(['/application'])
class Images(Resource):
    # если POST-запрос
    @api.expect(parser)
    def post(self):

        try:
            # определяем текущее время
            start_time = time.time()

            # проверка наличия файла в запросе
            if 'image_file' not in request.files:
                raise ValueError('No input file')


            # получаем файл из запроса
            f = request.files['image_file']
            save_path = request.files['save_path']

            # проверка на наличие имени у файла
            if f.filename == '':
                raise ValueError('Empty file name')

            # проверка на допустимое расширение файла (png, jpg, jpeg, tga, dds)
            if not allowed_file(f.filename):
                raise ValueError('Upload an image in one of the formats (zip, rar, 7z, tar, gz)')

            # имя файла
            image_file_name = secure_filename(f.filename)
            # задаем полный путь к файлу
            image_file_path = os.path.join(app.config['UPLOAD_FOLDER'], image_file_name)
            
            # инференс 1 картинки
            img = Image.open(image_file_path)
            res_image = model(img)
            Image.save(save_path, res_image)

            response = send_file(res_image, download_name=os.path.basename(res_image), as_attachment=True, mimetype='application/png')
            return response
        except ValueError as err:
            dict_response = {
                'error': err.args[0],
                'filename': f.filename,
                'time': (time.time() - start_time)
            }
            return dict_response

        except:
            dict_response = {
                'error': 'Unknown error',
                'time': (time.time() - start_time)
            }
            return dict_response


# запускаем сервер на порту 8008 (или на любом другом свободном порту)
if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=8888)


