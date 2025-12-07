from pydantic import BaseModel, Field
from datetime import datetime
from typing import List

class ClassificationRequest(BaseModel):
    image_base64: str = Field(..., description="Imagem codificada em base64")

class ClassificationResult(BaseModel):
    face_name: str = Field(..., description="Nome do rosto classificado")
    confidence: float = Field(..., description="Confiança da predição (0-1)")
    timestamp: datetime = Field(..., description="Horário da classificação")
    image_base64: str = Field(..., description="Imagem original em base64")

class AttendanceRecord(BaseModel):
    face_name: str
    confidence: float
    timestamp: str
    image_base64: str 

class AttendanceResponse(BaseModel):
    count: int
    records: List[AttendanceRecord]