import torch
import torch.nn.functional as F
from torchvision import transforms
from .model_loader import load_model

class FacePredictor:
    def __init__(self):
        self.model = None
        self.class_names = []
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def load_model(self, model_path, class_names_path):
        """Carrega o modelo treinado"""
        self.model, self.class_names = load_model(
            model_path, 
            class_names_path, 
            self.device
        )
    
    def predict(self, image_tensor):
        """Faz a predição na imagem"""
        if self.model is None:
            raise ValueError("Modelo não carregado. Chame load_model() primeiro.")
        
        with torch.no_grad():
            image_tensor = image_tensor.to(self.device)
            outputs = self.model(image_tensor.unsqueeze(0))
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            import pickle
            with open("models/finetuned_class_names.pkl", 'rb') as f:
                class_names = pickle.load(f)

            return {
                "class_name": self.class_names[predicted.item()],
                "confidence": confidence.item(),
                "class_index": predicted.item()
            }