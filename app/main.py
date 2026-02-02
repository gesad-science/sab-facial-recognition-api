from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field
import base64

ex_base64 = Path("app/base64example.txt").read_text().strip()

people_present: list[str, any] = []

app = FastAPI()

class Image(BaseModel):
    base64: str = Field(default=ex_base64, examples=["iVBORw0KGgoAAAANSUhEUgAAAoAAAA..."])
    model: str = Field(default="ResNet")

@app.post("/classify/")
async def classifyFace(image: Image):
    try:
        base64.b64decode(image.base64, validate=True)
    except:
        raise HTTPException(status_code=422, detail="Base64 field is not a valid Base64")
    
    print("calling the function...")
    result = {
        "classify result": "Camila",
        "confidence": 0.8792132502
        }
    people_present.append(result)

    return result

@app.get("/attendance")
async def attendance() -> dict:
    return {
        "count": len(people_present),
        "people": people_present
    }

@app.get("/attendence")
async def fake_attendance() -> RedirectResponse:
    return RedirectResponse(url="https://youtu.be/dQw4w9WgXcQ?si=KeYXsJOF8bPp7N_Q")