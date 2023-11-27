# -*- coding: utf-8 -*-
"""Выпускная работа_v2.0.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1nrFEGRrqquCqF_3HQW8dLWKPRDR1HnA6
"""

import numpy as np
import pandas as pd
import cv2

import torch
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import torchvision
from torchvision import utils as vutils
from torchvision import transforms
from torch import nn
from torch.nn import functional as F
from torch.utils import data
from torch.optim import SGD, Adam
from typing import Tuple, List, Dict, Union

from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_notebook

import pickle as pkl
from IPython.display import clear_output
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import os
from statistics import mean

!gdown --id 1dQntmAL5OsxkH7dXeoRzvehfY74uF_Ti

# https://drive.google.com/file/d/1dQntmAL5OsxkH7dXeoRzvehfY74uF_Ti/view?usp=drive_link  ---- pneumonia

!ls

# Загрузим архив с файлами
import zipfile
zip_ref = zipfile.ZipFile('lung.zip', 'r')
zip_ref.extractall()
zip_ref.close()

"""# Подготовка данных"""

from google.colab.patches import cv2_imshow
img0 = cv2.imread("/content/lung/normal/IM-0001-0001.jpeg")
print (img0.shape)

# Создание транформера изображений
from torchvision.transforms import v2
import matplotlib.pyplot as plt

train_transform = v2.Compose([
    v2.Resize(size=(512, 512)),
    v2.Grayscale(),
    v2.ToTensor(),
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomVerticalFlip(p=0.5),
    v2.Normalize(mean=[0.475, 0.450, 0.403], std=[0.499, 0.450, 0.458])
])

test_transform = v2.Compose([
    v2.Resize(size=(512, 512)),
    v2.ToTensor()
    ])

# Подготовка набора данных и загрузчика данных

from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
BATCH_SIZE = 32

dataset = ImageFolder(root='/content/lung', transform = train_transform)

from torchvision.utils import make_grid
grid = make_grid([dataset[i][0] for i in range(8)], nrow=4)

plt.figure(figsize=(15, 4))
plt.imshow(grid.permute(1, 2, 0));

# словарь {индекс класса: название класса}
index_to_class = {value: key for key, value in dataset.class_to_idx.items()}
index_to_class

target_indexes = torch.tensor(dataset.targets)

plt.bar([0, 1], target_indexes.bincount())
plt.xticks([0, 1], list(index_to_class.values()))
plt.show()

"""# Разделение на test и train"""

train_set, test_set = torch.utils.data.random_split(dataset, [0.8, 0.2])

# Subset принимает сам датасет и индексы, по которым он будет выбирать данные
train_subset = Subset(dataset, train_set.indices)
test_subset = Subset(dataset, test_set.indices)
len(train_set), len(test_set)

# инициализаия загрузчиков данных
train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

# проверка даталоадера
images, indxs = next(iter(train_loader))
images.shape, indxs

# Подсчет числа экземпляров классов 0 и 1 в train_dataset
train_class_0_count = 0
train_class_1_count = 0
for data, label in train_loader:
    train_class_0_count += torch.sum(label == 0).item()
    train_class_1_count += torch.sum(label == 1).item()

# Подсчет числа экземпляров классов 0 и 1 в test_dataset
test_class_0_count = 0
test_class_1_count = 0
for data, label in test_loader:
    test_class_0_count += torch.sum(label == 0).item()
    test_class_1_count += torch.sum(label == 1).item()

# Расчет соотношения 1 к 0 на трейне и тесте
    train_class_ratio = train_class_1_count / train_class_0_count
    test_class_ratio = test_class_1_count / test_class_0_count

print("Соотношение 1 к 0 в train_dataset:", train_class_ratio)
print("Соотношение 1 к 0 в test_dataset:", test_class_ratio)

"""# Архитектура модели"""

# функция отрисовки метрик, принимает словарь метрик - ошибка и accuracy
def plot_metrics(metrics: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    train_loss, val_loss, train_acc, val_acc = metrics.values()

    axes[0].set_title(f'Train loss: {train_loss[-1]:.2f}, Val loss: {val_loss[-1]:.2f}')
    axes[0].plot(train_loss, label='Train_loss')
    axes[0].plot(val_loss, label='Val_loss')

    axes[1].set_title(f'Train acc: {train_acc[-1]:.2f}, Val acc: {val_acc[-1]:.2f}')
    axes[1].plot(train_acc, label='Train_acc')
    axes[1].plot(val_acc, label='Val_acc')

    legend = [ax.legend() for ax in axes]
    plt.show()

torch.manual_seed(111)
# число сверток на первом слое сверточного блока
conv_size = 16

# класс для сверточной сети классификации
class BootsModel(nn.Module):
    def __init__(self, input_size: Tuple[int], conv_size: int):
        super().__init__()
        # сверточная часть сети
        self.conv = nn.Sequential(
            # input_size[0] - кол-во каналов  входной картинки
            nn.Conv2d(input_size[0], conv_size, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(conv_size, conv_size*2, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(conv_size*2, conv_size*4, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(conv_size*4, conv_size*8, 3, stride=2, padding=1),
            nn.ReLU())

        # расчет размерности линейного слоя
        linear_size = self.calc_linear_size(image_size)

        # линейные слои
        self.linear = nn.Sequential(nn.Flatten(),
            nn.Linear(linear_size, 512), nn.ReLU(), # 512 - число нейронов в скрытом слое
            nn.Linear(512, len(index_to_class)))  # на выходе столько нейронов сколько классов

    # функция расчета размерности выхода последнй свертки для входа линейного слоя
    @torch.inference_mode()
    def calc_linear_size(self, input_size: Tuple[int]) -> int:
        # массив случайных данных размера как у входных картинок
        empty_image = torch.empty(input_size)
        # пропустить этот массив через сверточный блок
        conv_out_size = self.conv(empty_image)
        linear_size = conv_out_size.numel()
        print(f'Model create. Conv out size: {conv_out_size.shape}, Linear size: {linear_size}')
        return linear_size

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        x = self.conv(inp)
        out = self.linear(x)
        return out

# взять первую картинку из датасета и записать ее размер (128, 32, 32)
image_size = tuple(dataset[0][0].shape)

# создание модели и перемещение на девайс
model = BootsModel(image_size, conv_size)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# оптимизатор и функция ошибки
LR = 0.001    #----Learning Rate (скорость обучения)
opt = torch.optim.Adam(model.parameters(), lr=LR)
class_weights = torch.tensor([5.0, 1.0])
loss_fn = nn.CrossEntropyLoss(weight = class_weights)
model

"""# Обучение модели"""

torch.manual_seed(111)

EPOCHS = 20
metrics = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

for epoch in tqdm(range(EPOCHS), desc='Epoch'):
    loss_epoch, acc_epoch  = 0, 0
    len_dataset = len(train_loader.dataset)

    for images, labels in tqdm_notebook(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        class_weights = class_weights.to(device)

        # расчет прогнозов
        outputs = model(images)
        # пропустить картинки через модель и посчитать ошибку
        loss = loss_fn(outputs, labels)

        # Расчет потерь с учетом весов классов
        loss = loss_fn(outputs, labels)
        weighted_loss = torch.mean(loss * class_weights[labels])

        # обучение сети: обратное распространение и оптимизация
        opt.zero_grad()
        loss.backward()
        opt.step()

        # так как ошибка усредняется по батчу, делаем обратную денормировку
        # чтобы потом просто разделить на длину датасета
        loss_epoch += loss.item() * labels.size(0)
        acc_epoch += torch.sum(outputs.argmax(dim=1) == labels).item()

    metrics['train_loss'].append(loss_epoch / len_dataset)
    metrics['train_acc'].append(acc_epoch / len_dataset)

# ---------------------Тестирование------------------------------------
    loss_epoch, acc_epoch  = 0, 0
    len_dataset = len(test_loader.dataset)

    with torch.inference_mode():
      for images, labels in test_loader:
    # Перенос данных на устройство для обучения (GPU, если доступен)
            images = images.to(device)
            labels = labels.to(device)

            # Расчет прогнозов
            outputs = model(images)

            # Выбор наиболее вероятного класса
            _, predicted = torch.max(outputs.data, 1)

            # Вычисление ошибки и точности модели
            loss = loss_fn(outputs, labels)
            loss_epoch += loss.item() * labels.size(0)
            acc_epoch += torch.sum(outputs.argmax(dim=1) == labels).item()


    metrics['val_loss'].append(loss_epoch / len_dataset)
    metrics['val_acc'].append(acc_epoch / len_dataset)

    # отрисовка графиков
    clear_output(True)
    plot_metrics(metrics)

"""# Сериализация модели

"""

# сохранение модели
with open ('model.pkl', 'wb') as f:
  pkl.dump (model, f)

# загрузка сохраненной модели
with open ('model.pkl', 'rb') as f:
  model = pkl.load(f)

"""# Инференс модели"""

model.eval()

"""### Рассчитаем F1 score и ROC/AUC

Рассчитаем **F1 score**, так как в датасете дисбаланс классов, accuracy может быть неустойчива в этом случае и не отражать действительную точность модели.
"""

from sklearn.metrics import f1_score

all_targets = []
all_predictions = []

with torch.no_grad():
    for batch_data, batch_labels in test_loader:
        batch_outputs = model(batch_data)
        _, batch_predictions = torch.max(batch_outputs, 1)
        all_targets.extend(batch_labels.numpy().tolist())
        all_predictions.extend(batch_predictions.numpy().tolist())

f1 = f1_score(all_targets, all_predictions)
print("F1 Score:", f1)

"""Вывод: Судя по значению **F1 score = 0.938**, можно сделать вывод, что модель имеет высокую точность и полноту в классификации данных.

Это означает, что модель хорошо справляется с определением истинных положительных и отрицательных примеров и минимизирует число ложноположительных и ложноотрицательных предсказаний.
"""

roc_auc = roc_auc_score(all_targets, all_predictions)
print("ROC AUC:", roc_auc)

"""### Протестируем работу модели на снимке с затемнением"""

!gdown --id 1PxYDvnAOZhJiUJtzv3mdxaDFdgxZH_kw

# https://drive.google.com/file/d/1R962DGz5SrmLx8pfdZqiaQmJMOvuXSng/view?usp=drive_link --- opacity
# https://drive.google.com/file/d/1PxYDvnAOZhJiUJtzv3mdxaDFdgxZH_kw/view?usp=drive_link --- opacity

pil_image = Image.open("/content/person20_virus_51.jpg").convert('RGB')
plt.imshow(pil_image)

# применить аугментации
tensor_image = test_transform(pil_image)

# пропустить картинку ботинка через модель, получить предсказания и переместить на девайс
with torch.inference_mode():
    logits = model(tensor_image.unsqueeze(0).to(device))

# получить индекс предсказанного класса - максимальный предикт
class_index = logits.argmax(dim=-1).item()

# получить вероятности предсказанных классов
probs = torch.softmax(logits, dim=-1).tolist()

print(f'Предсказанный класс: {index_to_class[class_index]}')
print('======================')
for class_name, prob in zip(index_to_class.values(), probs[0]):
    print(f'Класс: {class_name}, вероятность: {prob:.2f}')

"""### Теперь протестируем работу модели на снимке с нормой"""

!gdown --id 1zCOqFlg5pWCaJOjkEcl-Ujwd27_4QOxP

# https://drive.google.com/file/d/1yJJK11xivo90-IIYQdfth_2Vb6-RcCGB/view?usp=drive_link --- norma
# https://drive.google.com/file/d/1zCOqFlg5pWCaJOjkEcl-Ujwd27_4QOxP/view?usp=drive_link --- norma

pil_image = Image.open("/content/norma.jpg").convert('RGB')
plt.imshow(pil_image)

# применить аугментации на которых модель обучалась (либо валидировалась если они отличались)
tensor_image = test_transform(pil_image)

# пропустить картинку ботинка через модель, получить предсказания
# не забыть добавить размерность и переместить на девайс
with torch.inference_mode():
    logits = model(tensor_image.unsqueeze(0).to(device))

# получить индекс предсказанного класса - максимальный предикт
class_index = logits.argmax(dim=-1).item()

# получить вероятности предсказанных классов
probs = torch.softmax(logits, dim=-1).tolist()

print(f'Предсказанный класс: {index_to_class[class_index]}')
print('======================')
for class_name, prob in zip(index_to_class.values(), probs[0]):
    print(f'Класс: {class_name}, вероятность: {prob:.2f}')