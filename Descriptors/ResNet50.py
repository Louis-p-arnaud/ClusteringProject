import torch
import torchvision.models as models
model = models.resnet50(pretrained=True)
model.fc = torch.nn.Identity()  # Enlève classifieur, garde 2048D features
features = model(image_tensor)  # Vecteur 2048D par image