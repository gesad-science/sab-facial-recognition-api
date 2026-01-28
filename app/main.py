"""
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

people_present: list[str, any] = []

app = FastAPI()

class Image(BaseModel):
    base64: str = Field(examples=["Base64"])
    model: str = Field(default=None, examples=["ResNet"])

@app.post("/classify/")
async def classifyFace(image: Image):

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
"""
from typing import Any, Annotated

from fastapi import FastAPI, Response, File, UploadFile
from fastapi.responses import JSONResponse, RedirectResponse, HTMLResponse
from pydantic import BaseModel, Field, EmailStr

app = FastAPI()

## minha tentativa real
class Image(BaseModel):
    base64: str = Field(examples=["Base64"])
    model: str = Field(default=None, examples=["ResNet"])
## --------------------

class UserIn(BaseModel):
    username: str
    password: str
    email: EmailStr
    full_name: str | None = None

class UserOut(BaseModel):
    username: str
    email: EmailStr
    full_name: str | None = None

@app.get("/link")
async def get_link(tp: bool = False) -> Response:
    if tp:
        return RedirectResponse(url="https://youtu.be/hPr-Yc92qaY?si=I3BPoMwHFA-EMGXk")
    return JSONResponse(content={"message": "Here's your interdimensional portal."})

@app.post("/user/", response_model=UserOut)
async def create_user(user: UserIn) -> Any:
    return user

## minha tentativa real
@app.post("/classify/")
async def classifyFace(image: Image):
    return image
## --------------------


######
@app.post("/files/")
async def create_files(
    files: Annotated[list[bytes], File(description="Multiple files as bytes")],
):
    return {"file_sizes": [len(file) for file in files]}


@app.post("/uploadfiles/")
async def create_upload_files(
    files: Annotated[
        list[UploadFile], File(description="Multiple files as UploadFile")
    ],
):
    return {"filenames": [file.filename for file in files]}


@app.get("/")
async def main():
    content = """
<body>
<form action="/files/" enctype="multipart/form-data" method="post">
<input name="files" type="file" multiple>
<input type="submit">
</form>
<form action="/uploadfiles/" enctype="multipart/form-data" method="post">
<input name="files" type="file" multiple>
<input type="submit">
</form>
</body>
    """
    return HTMLResponse(content=content)