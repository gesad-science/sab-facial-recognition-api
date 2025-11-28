import base64
import io
from PIL import Image
import torch
from torchvision import transforms

def base64_to_tensor(image_base64: str) -> torch.Tensor:
    """
    Converte imagem base64 para tensor normalizado
    """
    # Decodificar base64
    image_data = base64.b64decode(image_base64)
    image = Image.open(io.BytesIO(image_data)).convert('RGB')
    
    # Transformações
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    return transform(image)