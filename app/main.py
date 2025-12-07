from fastapi import FastAPI, HTTPException
from datetime import datetime
from typing import List, Dict, Any
from fastapi.middleware.cors import CORSMiddleware
import nest_asyncio
from pyngrok import ngrok
import uvicorn

from app.models.predictor import FacePredictor
from app.schemas.schemas import ClassificationRequest, ClassificationResult
from app.services.api_client import APIClient
from app.utils.image_processing import base64_to_tensor

app = FastAPI(title="Face Recognition API", version="1.0.0")

# Inicializar componentes
predictor = FacePredictor()
api_client = APIClient("https://sua-api-de-destino.com/api/classifications")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """Carrega o modelo ao iniciar a aplicação"""
    try:
        predictor.load_model(
            model_path="models/finetuned_model.pth",
            class_names_path="models/finetuned_class_names.pkl"
        )
        print("Modelo carregado com sucesso!")
    except Exception as e:
        print(f"Erro ao carregar modelo: {str(e)}")
        raise

@app.get("/")
async def root():
    return {"message": "Face Recognition API está rodando!"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now()}

@app.post("/v1/ResNet", response_model=ClassificationResult)
async def classify_image(request: ClassificationRequest):
    """
    Classifica uma imagem em base64 e envia o resultado para API externa
    """
    try:
        image_tensor = base64_to_tensor(request.image_base64)
        prediction = predictor.predict(image_tensor)
        
        result = ClassificationResult(
            face_name=prediction["class_name"],
            confidence=prediction["confidence"],
            timestamp=datetime.now(),
            image_base64=request.image_base64 
        )
        
        # envia para API externa
        success = await api_client.send_classification(result)
        
        if not success:
            print("Atenção: Não foi possível enviar para API externa")
        
        attendance_record = {
            "face_name": prediction["class_name"],
            "confidence": prediction["confidence"],
            "timestamp": datetime.now().isoformat(),
            "image_base64": request.image_base64
        }
        attendance_records.clear()
        attendance_records.append(attendance_record)

        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro na classificação: {str(e)}")

@app.get("/model/classes")
async def get_model_classes():
    """Retorna as classes que o modelo reconhece"""
    return {"classes": predictor.class_names}


attendance_records: List[Dict[str, Any]] = []

@app.get("/attendance")
async def get_attendance_list():
    return {
        "count": len(attendance_records),
        "records": attendance_records
    }

@app.post("/attendance")
async def post_attendance_list(record: Dict[str, Any]):
    try:
        # valida campos obrigatórios
        required_fields = ["face_name", "confidence", "timestamp", "image_base64"]
        for field in required_fields:
            if field not in record:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Campo obrigatório faltando: {field}"
                )
        
        # limpa e adiciona à lista
        attendance_records.clear()
        attendance_records.append(record)
        
        print(f"✅ Registro adicionado: {record['face_name']} - {record['timestamp']}")
        
        return {
            "message": "Registro adicionado com sucesso",
            "record": record
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao adicionar registro: {str(e)}")


# cria o túnel HTTPS público
static_domain = "nonpossibly-aspish-fletcher.ngrok-free.dev"
ngrok_tunnel = ngrok.connect(addr="8000", proto="http", domain=static_domain)

print('Public URL:', ngrok_tunnel.public_url)

nest_asyncio.apply()

uvicorn.run(app, host="0.0.0.0", port=8000)