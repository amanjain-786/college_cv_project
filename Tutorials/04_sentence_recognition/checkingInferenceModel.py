import cv2
import typing
import numpy as np

from mltu.inferenceModel import OnnxInferenceModel
from mltu.utils.text_utils import ctc_decoder, get_cer, get_wer
from mltu.transformers import ImageResizer
import csv
import pandas as pd
import random
import os
from line_breaker import segment_lines

from textblob import TextBlob

def change_resolution(image, factor):
    width = image.shape[1]
    height = image.shape[0]
    new_width = int(width * factor)
    new_height = int(height * factor)
    resized_image = cv2.resize(image, (new_width, new_height))
    return resized_image

def convert_to_bw(image):
    bw_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bw_image = cv2.cvtColor(bw_image, cv2.COLOR_GRAY2BGR)
    return bw_image


def spell_check_textblob(text):
    blob = TextBlob(text)
    corrected_text = str(blob.correct())
    return corrected_text


class ImageToWordModel(OnnxInferenceModel):
    def __init__(self, char_list: typing.Union[str, list], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.char_list = char_list

    def predict(self, image: np.ndarray):
        image = ImageResizer.resize_maintaining_aspect_ratio(image, *self.input_shape[:2][::-1])

        image_pred = np.expand_dims(image, axis=0).astype(np.float32)

        preds = self.model.run(None, {self.input_name: image_pred})[0]

        text = ctc_decoder(preds, self.char_list)[0]

        return text

import pandas as pd
from tqdm import tqdm
from mltu.configs import BaseModelConfigs

configs = BaseModelConfigs.load(r'D:\amanIaf\mltu_sentences\mltu\Tutorials\04_sentence_recognition\Models\04_sentence_recognition\202301131202\configs.yaml')

model = ImageToWordModel(model_path=configs.model_path, char_list=configs.vocab)

def image_to_predicted_text(image_path):
    img_list = segment_lines(image_path)

    wo_correction = ""
    w_correction = ""

    for image in img_list:
        prediction_text = model.predict(image)
        modified_text = spell_check_textblob(prediction_text)

        wo_correction += prediction_text + "\n"
        w_correction += modified_text + "\n"

    w_correction = spell_check_textblob(w_correction)
    return wo_correction, w_correction