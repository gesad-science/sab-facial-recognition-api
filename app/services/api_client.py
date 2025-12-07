import httpx
from app.schemas.schemas import ClassificationResult

class APIClient:
    def __init__(self, external_api_url: str = None):
        self.external_api_url = external_api_url
        self.internal_api_url = "http://127.0.0.1:8000/attendance"
    
    async def send_classification(self, result: ClassificationResult) -> bool:
        """
        Envia resultado da classificação para:
        1. Nossa própria API (sempre)
        2. API externa (se configurada)
        """
        payload = {
            "face_name": result.face_name,
            "confidence": round(result.confidence, 4),
            "timestamp": result.timestamp.isoformat(),
            "image_base64": result.image_base64
        }
        
        success_internal = await self._send_to_internal_api(payload)
        success_external = True
        
        # se há uma API externa configurada, enviar também
        if self.external_api_url:
            success_external = await self._send_to_external_api(payload)
        
        return success_internal and success_external
    
    async def _send_to_internal_api(self, payload: dict) -> bool:
        """
        Envia para nossa própria API
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.internal_api_url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=10.0
                )
                
                if response.status_code == 200:
                    print("✅ Classificação salva localmente com sucesso!")
                    return True
                else:
                    print(f"⚠️ Erro ao salvar localmente: {response.status_code}")
                    return False
                        
        except Exception as e:
            print(f"❌ Erro ao salvar na API local: {str(e)}")
            return False
    
    async def _send_to_external_api(self, payload: dict) -> bool:
        """
        Envia para API externa (se configurada)
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.external_api_url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=10.0
                )
                
                if response.status_code == 200:
                    print("✅ Classificação enviada para API externa!")
                    return True
                else:
                    print(f"⚠️ Erro ao enviar para API externa: {response.status_code}")
                    return False
                        
        except Exception as e:
            print(f"❌ Erro na comunicação com API externa: {str(e)}")
            return False