import os
import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, models, transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

class ImageClassifier:
    def __init__(self, model_path, data_dir):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.transforms_test = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.class_names = None
        self.model = None
        self.load_model(model_path)
        self.load_data(data_dir)

    def load_model(self, model_path):
        self.model = models.resnet34(pretrained=True)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, 3)
        self.model.load_state_dict(torch.load(model_path))
        self.model = self.model.to(self.device)
        self.model.eval()

    def load_data(self, data_dir):
        test_datasets = datasets.ImageFolder(os.path.join(data_dir, 'test'), self.transforms_test)
        self.class_names = test_datasets.classes

    def classify_image(self, image_path):
        image = Image.open(image_path)
        image = self.transforms_test(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(image)
            _, preds = torch.max(outputs, 1)

        return self.class_names[preds[0]]

    def imshow(self, input, title):
        input = input.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        input = std * input + mean
        input = np.clip(input, 0, 1)
        plt.imshow(input)
        plt.title(title)
        plt.show()