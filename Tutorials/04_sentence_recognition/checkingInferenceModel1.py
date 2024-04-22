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
from line_breaker1 import segment_lines

from textblob import TextBlob
from PIL import Image
from PIL import ImageTk

# Mukul's Addition

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

# Mukul's Addition ended


def spell_check_textblob(text):
    blob = TextBlob(text)
    corrected_text = str(blob.correct())  # Corrects spelling mistakes
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

# if __name__ == "__main__":
#     import pandas as pd
#     from tqdm import tqdm
#     from mltu.configs import BaseModelConfigs

#     configs = BaseModelConfigs.load(r'''C:\Users\Rudra\Desktop\IAF Intern\mltu_sentences\mltu\Tutorials\04_sentence_recognition\Models\04_sentence_recognition\202301131202\configs.yaml''')

#     model = ImageToWordModel(model_path=configs.model_path, char_list=configs.vocab)

import pandas as pd
from tqdm import tqdm
from mltu.configs import BaseModelConfigs

configs = BaseModelConfigs.load(r'''D:\amanIaf\mltu_sentences\mltu\Tutorials\04_sentence_recognition\Models\04_sentence_recognition\202301131202\configs.yaml''')

model = ImageToWordModel(model_path=configs.model_path, char_list=configs.vocab)
    # df = pd.read_csv(r'''C:\Users\Rudra\Desktop\IAF Intern\mltu_sentences\mltu\Tutorials\04_sentence_recognition\Models\04_sentence_recognition\202301131202\val.csv''').values.tolist()

    # # f = open('abc.csv', 'w', newline='')
    # # writer = csv.writer(f)

    # accum_cer, accum_wer = [], []
    # corrected_accum_cer ,corrected_accum_wer = [],[]

    # for image_path, label in tqdm(df):
    #     image_path=r"C:\Users\Rudra\Desktop\IAF Intern\mltu_sentences\mltu\Tutorials\04_sentence_recognition\Datasets\IAM_Sentences\sentences\\"+image_path.split("/")[-1]
    #     image = cv2.imread(image_path)
    #     # print(image_path)
    #     original = label
    #     prediction_text = model.predict(image)
    #     corrected_text=spell_check_textblob(prediction_text)

    #     cer = get_cer(prediction_text, label)
    #     wer = get_wer(prediction_text, label)
    #     c_cer=get_cer(corrected_text,label)
    #     c_wer=get_wer(corrected_text,label)

        

    #     # print("Image: ", image_path)
    #     print("Label:", label)
    #     print("Prediction: ", prediction_text)
    #     print("After Correction: ", corrected_text)
    #     print(f"CER: {cer}; WER: {wer}")
        
    #     temp_str = str(cer) + "," + str(wer)

    #     # writer.writerow([temp_str])

    #     accum_cer.append(cer)
    #     accum_wer.append(wer)
    #     corrected_accum_cer.append(c_cer)
    #     corrected_accum_wer.append(c_wer)

    #     # cv2.imshow(prediction_text, image)
    #     # cv2.waitKey(0)
    #     # cv2.destroyAllWindows()

    # # dict = {'accum_cer': accum_cer, 'accum_wer': accum_wer}

    # # df = pd.DataFrame(dict)
    
    # # df.to_csv('pqr.csv')

    # # f.close()
    # print(f"Average CER: {np.average(accum_cer)}, Average WER: {np.average(accum_wer)}")
    # print(f"Average C_CER: {np.average(corrected_accum_cer)}, Average W_WER: {np.average(corrected_accum_wer)}")

#dir = r'C:\Users\Rudra\Desktop\IAF Intern\mltu_sentences\sampleImages'  
    
# dir = r'C:\Users\Rudra\Desktop\IAF Intern\mltu_sentences\mltu\Tutorials\04_sentence_recognition\Datasets\IAM_Sentences\sentences'
 
#filename = random.choice(dircache.listdir(dir))
# filename = random.choice(os.listdir(dir))
# image_path = os.path.join(dir, filename)
# image_path = r'C:\Users\Rudra\Desktop\IAF Intern\mltu_sentences\mltu\Tutorials\04_sentence_recognition\Datasets\IAM_Sentences\aman.png'

def image_to_predicted_text(image_path):
    img_list,img_rect = segment_lines(image_path)

    wo_correction = ""
    w_correction = ""

    for image in img_list:
        prediction_text = model.predict(image)
        modified_text = spell_check_textblob(prediction_text)
        # print(f"Prediction: {prediction_text} \n")

        wo_correction += prediction_text + "\n"
        w_correction += modified_text

        # cv2.imshow(prediction_text, image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    w_correction = spell_check_textblob(w_correction)
    img_rect = cv2.cvtColor(img_rect, cv2.COLOR_BGR2RGB)
    img_rect = Image.fromarray(img_rect)
    return wo_correction, w_correction,img_rect


#image_path=r"C:\Users\Rudra\Downloads\000-20240104T063507Z-001\000\a01-011u.png"
#wo_correction, w_correction = image_to_predicted_text(image_path)

#print(f"without correction :\n{wo_correction}\n")
#print(f"with correction :\n{w_correction}\n")
# image = cv2.imread(image_path)
# print(image_path)

# # Mukul's Addition
# # image = change_resolution(image, 0.5)
# # image = convert_to_bw(image)
# # cv2.imshow("Sample" ,image)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()
# # Mukul's Addition Ended
# prediction_text = model.predict(image)
# modified_text = spell_check_textblob(prediction_text)

# print("Image: ", image_path)
# print("Prediction: ", prediction_text)
# print("After Correction: ", modified_text)
# cv2.imshow(modified_text, image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()