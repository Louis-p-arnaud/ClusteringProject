import torch
import timm
from PIL import Image
from torchvision import transforms
import numpy as np

class ViTDescriptor:
    def __init__(self):
        # On charge un modèle ViT pré-entraîné (le plus léger et efficace)
        self.model = timm.create_model('vit_tiny_patch16_224', pretrained=True, num_classes=0)
        self.model.eval()
        
        # Transformation standard pour ViT (224x224 pixels)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def get_features(self, image_path):
        try:
            img = Image.open(image_path).convert('RGB')
            img_t = self.transform(img).unsqueeze(0) # Ajout dimension batch
            
            with torch.no_grad():
                # Extrait un vecteur de 192 dimensions (pour vit_tiny)
                features = self.model(img_t)
            return features.numpy().flatten()
        except Exception as e:
            print(f"Erreur sur {image_path}: {e}")
            return None