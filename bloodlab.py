# modules

import gradio as gr
import os
# Let's pick the desired backend
# os.environ['USE_TF'] = '1'
os.environ['USE_TORCH'] = '1'
import matplotlib.pyplot as plt
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from PIL import Image
import numpy as np
import argparse

# util functions

def prepreocess(text):
    """
    convert to lowercase
    removing . from words
    convert string numbers to number
    """
    text = text.lower()
    try:
        text = float(text)
        return text
    except:
        pass
    text = text.replace('.', '')
    return text

normal_values = {
    'wbc': {"min": 4000, "max": 11000, "minwarning": "هپاتیت و آرتریت", "maxwarning": "وجود عفونت"},
    'rbc': {"min": 4.5, "max": 6.1, "minwarning": "بی‌ نظمی در گلوبول های قرمز خون", "maxwarning": "بی‌ نظمی در گلوبول های قرمز خون"},
    'hb': {"min": 12.3, "max": 17.50, "minwarning": "هموگلوبین کم است", "maxwarning": "هموگلوبین زیاد است"},
    'hgb': {"min": 12.3, "max": 17.50, "minwarning": "هموگلوبین کم است", "maxwarning": "هموگلوبین زیاد است"},
    'hct': {"min": 36.00, "max": 45.00, "minwarning": "نسبت گلوبولهای قرمر به خون نامتناسب است", "maxwarning": "نسبت گلوبولهای قرمر به خون نامتناسب است "},
    'mch': {"min": 27, "max": 33, "minwarning": "احتمال سوء تغذیه", "maxwarning": "احتمال کم خونی"},
    'mchc': {"min": 27, "max": 33, "minwarning": "احتمال سوء تغذیه", "maxwarning": "احتمال کم خونی"},
    'mcv': {"min": 80, "max": 96, "minwarning": "نشانه ای از کم خونی یا سندرم خستگی مزمن", "maxwarning": "نشانه ای از کم خونی یا سندرم خستگی مزمن"},
    'mchc': {"min": 33.4, "max": 35.5, "minwarning": "احتمال سوء تغذیه", "maxwarning": "احتمال کم خونی"},
    'plt': {"min": 150, "max": 450, "minwarning": "سطح تخمیر نرمال نیست", "maxwarning": "سطح تخمیر نرمال نیست"},
    'fbs': {"min": 70, "max": 100, "minwarning": "قند خون پایین", "maxwarning": "قند خون بالا"},
    'hbalc': {"min": 4000, "max": 11000, "minwarning": "", "maxwarning": ""},
    'urea': {"min": 4000, "max": 11000, "minwarning": "", "maxwarning": ""},
    'creatinine': {"min": 4000, "max": 11000, "minwarning": "", "maxwarning": ""},
    'cholesterol': {"min": 4000, "max": 11000, "minwarning": "", "maxwarning": ""},
    'triglycerides': {"min": 4000, "max": 11000, "minwarning": "", "maxwarning": ""},
    'hdl': {"min": 40, "max": 11000, "minwarning": "خطر ابتلا به بیماری قلبی عروقی", "maxwarning": ""},
    'ldl': {"min": 70, "max": 100, "minwarning": "خطر ابتلا به بیماری قلبی عروقی", "maxwarning": "خطر ابتلا به بیماری قلبی عروقی"},
    "rdw": {"min": 4000, "max": 11000, "minwarning": "", "maxwarning": ""},
    'rcdw': {"min": 4000, "max": 11000, "minwarning": "", "maxwarning": ""},
    'hgb': {"min": 4000, "max": 11000, "minwarning": "", "maxwarning": ""},
    'bp': {"min": 80, "max": 120, "minwarning": "فشار خون پایین", "maxwarning": "فشار خون بالا"},
    'bg': {"min": 70, "max": 150, "minwarning": "قند خون پایین", "maxwarning": "قند خون بالا"},
    'esr': {"min": 0, "max": 20, "minwarning": "امکان عفونت و یا التهاب", "maxwarning": " امکان عفونت و یا التهاب "},
    'crp': {"min": 0, "max": 1000, "minwarning": "امکان عفونت و یا التهاب", "maxwarning": " امکان عفونت و یا التهاب "},

}

predictor = ocr_predictor(pretrained=True)

def proceed(image):
    doc = DocumentFile.from_images(image) #reading input image
    result = predictor(doc) # extract words
    json_export = result.export() # convert to json
    out_words = [[prepreocess(w['value']) for w in line['words'] ] for line in json_export['pages'][0]['blocks'][0]['lines']] # extracting words
    final_words = []
    for item in out_words:
        final_words += item
    wanted = {
        'wbc': 'nan',
        'rbc': 'nan',
        'hb': 'nan',
        'hgb': 'nan',
        'hct': 'nan',
        'mcv': 'nan',
        'mch': 'nan',
        'mchc': 'nan',
        'mchc': 'nan',
        'plt': 'nan',
        'fbs': 'nan',
        'hbalc': 'nan',
        'urea': 'nan',
        'creatinine': 'nan',
        'cholesterol': 'nan',
        'triglycerides': 'nan',
        'hdl': 'nan',
        'ldl': 'nan',
        "rdw": 'nan',
        'rcdw': 'nan',
        'hgb': 'nan',
        'bp': 'nan',
        'esr': 'nan',
        'crp': 'nan',
    }
    # extracting value per each wanted keyword
    counter = 0
    while counter < len(final_words):
        if wanted.get(final_words[counter], 0)== 'nan':
            if type(final_words[counter+1]) == float:
                wanted[final_words[counter]] = final_words[counter+1]
                counter += 1
            if type(final_words[counter+2]) == float:
                wanted[final_words[counter]] = final_words[counter+2]
                counter += 2
        counter += 1

    # print(f"out is {out_gen}")
    # checking for normal values 
    printed_result = []
    for k,v in wanted.items():
        if v == 'nan':
            continue
        normal_value = normal_values.get(k, {})
        if v < normal_value.get('min', -1000):
            printed_result.append(normal_value.get('minwarning', ''))
        if v > normal_value.get('max', 100000):
            printed_result.append(normal_value.get('maxwarning', ''))
    return '\n'.join(printed_result)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', "--image", type=str, required=True, help='image file path')    
    args = parser.parse_args()
    out = proceed(args.image)
    print(out)
  
