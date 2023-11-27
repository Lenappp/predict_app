import streamlit as st
import shap
import pandas as pd
import numpy as np
import io
import time

import torch
import torch.nn as nn
from typing import Tuple
import torchvision
from torchvision import transforms
from torchvision.transforms import v2


from pathlib import Path
import pickle as pkl
import os
from PIL import Image
from torchvision.datasets import ImageFolder



# Класс модели
conv_size = 16
class BootsModel(nn.Module):
    def __init__(self, input_size: Tuple[int], conv_size: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_size[0], conv_size, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(conv_size, conv_size*2, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(conv_size*2, conv_size*4, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(conv_size*4, conv_size*8, 3, stride=2, padding=1),
            nn.ReLU())
        
    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        x = self.conv(inp)
        out = self.linear(x)
        return out

model = BootsModel([128, 32, 32], 16)


def main():
    st.title ('ОНЛАЙН рентгенолог')
    st.subheader ('Система помощи принятия врачебных решений')

main()

# загрузка изображения
image = Image.open('scale.jpg')
st.image(image, width = 500)

demo = st.button ("ОНЛАЙН ДЕМО", help="Запросить онлайн демонстрацию", on_click=None, type="primary", use_container_width=False, key='demo')
help = st.button ("СВЯЗАТЬСЯ С НАМИ", help="Оставить заявку для обратной связи", type="primary", use_container_width=False)


# форма обратной связи  
if 'clicked' not in st.session_state:
            st.session_state.clicked = False
def click_button():
    st.session_state.clicked = True
if st.session_state.clicked:
    st.success('Заявка успешно отправлена', icon="✅")
    time.sleep(1)
    st.session_state.clicked = False

feedback = False
if help:
    with st.form(key='my_form'):
        text_input = st.text_input(label="Введите ваше имя")
        text_input = st.text_input(label="Введите ваш телефон")
        text_input = st.text_input(label="Введите ваш email")
        submit_button = st.form_submit_button(label="Отправить", on_click = click_button)


# загрузка модели
@st.cache_data
def load_model(model_path):
    with open(model_path, 'rb') as file:
        model = pkl.load(file)
    return model

model_path = Path.cwd() / 'model_v1.0.pkl'
model = load_model(model_path)
class_labels = {0: 'normal', 1: 'opacity'}
class_to_idx = {label: index for index, label in class_labels.items()}
index_to_class = {value: key for key, value in class_to_idx.items()}


uploaded_image = st.file_uploader("Загрузите снимок в формате jpg", type='jpg', accept_multiple_files=False, help="Нажмите для загрузки файла", label_visibility="visible", key='upl_img')
if demo:
    uploaded_image

def load_image():
     if uploaded_image is not None:
        image_data = uploaded_image.getvalue()
        st.image(image_data)
        return Image.open(io.BytesIO(image_data))

result = st.button('Обработать', key='convert')
preprocessed_img = load_image()


# аугментации изображения
transform = v2.Compose([
        v2.Resize(size=(512, 512)),
        v2.Grayscale(num_output_channels=3),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True)
        ])   

def preprocessing_img(img):
    with torch.no_grad():
        transformed_img = transform (img)
    return transformed_img


def prediction(img):
    output = model(img.unsqueeze(0))
    class_index = output.argmax(dim=-1).item()
    if class_index >= len(class_labels):
        st.warning("Ошибка: Загрузите снимок ОГК")
        return
    probability = torch.softmax(output, dim=-1).tolist()[0]
    predicted_probability = probability[class_index]
    predicted_class = index_to_class[class_index]
    st.write(f'**Предсказанный класс:** {predicted_class}', f', **вероятность:** {predicted_probability:.2f}')


if result:
    preprocessed_img = preprocessing_img(preprocessed_img)
    preds = prediction(preprocessed_img)
