import model

from typing import Annotated
from pathlib import Path
import random

from fastapi import FastAPI, HTTPException, Depends, Query, status, Request
from fastapi.responses import RedirectResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field as pField
import base64
import numpy as np
import cv2
from datetime import datetime
from sqlmodel import Field as sqlField, Session, SQLModel, create_engine, select, delete ### maybe we can use postgreSQL later

app = FastAPI()

templates = Jinja2Templates(directory="templates")

#middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins="nonpossibly-aspish-fletcher.ngrok-free.dev",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)
#end middleware

#classes
class PersonBase(SQLModel):
    name: str = sqlField(index=True)                            #index is True to search be easier (filter by name)
    base64: str
    distance: float

class Person(PersonBase, table=True):
    id: int | None = sqlField(default=None, primary_key=True)   #default is None 'cause we want to be automatic without specify the id
    date: datetime = sqlField(index=True, default_factory=datetime.utcnow)

class PersonPublic(PersonBase):
    id: int
    date: datetime

ex_base64 = Path("src/base64example.txt").read_text().strip()
class FaceImage(BaseModel):
    base64: str = pField(default=ex_base64, examples=["iVBORw0KGgoAAAANSUhEUgAAAoAAAA..."])
    model: str = pField(default="ResNet")
#end classes

#start database
sqlite_file_name = "database.db" 
sqlite_url = f"sqlite:///{sqlite_file_name}"

connect_args = {"check_same_thread": False}                     #False allow FastAPI to use the same SQLite database in different threads
engine = create_engine(sqlite_url, connect_args=connect_args)

def create_db_and_tables():
    SQLModel.metadata.create_all(engine)

def get_session():
    with Session(engine) as session:
        yield session                                           #ensures that we use a single session per request

SessionDep = Annotated[Session, Depends(get_session)]           #simplifies the code
#"end" start database

#functions and endpoints
def create_person(person: PersonBase, session: SessionDep):
    db_person = Person.model_validate(person)
    session.add(db_person)
    session.commit()
    session.refresh(db_person)
    return db_person

@app.on_event("startup")
async def on_startup():
    create_db_and_tables()
    model.load_models()

@app.get("/")
async def opening_message():
    return {"message": "API of facial classification"}

@app.delete("/restart", status_code=status.HTTP_204_NO_CONTENT)
def delete_person(session: SessionDep):
    with Session(engine) as session:
        session.exec(delete(Person))
        session.commit()
        return

@app.post("/mock/classify", response_model=PersonPublic, status_code=status.HTTP_201_CREATED)            #we don't actually return a PersonPublic so db_person will be ajusted to a PersonPublic ('cause of "responde_model=")
async def classify_face_mock(image: FaceImage, session: SessionDep):
    """
        this is the mock classification model endpoint
    """
    try:
        base64.b64decode(image.base64, validate=True)
    except:
        raise HTTPException(status_code=422, detail="Base64 field is not a valid Base64")
        
    classes = ["Guilherme",
               "David",
               "Camila",
               "Luna",
               "Alan",
               "Paulo Henrique",
               "Layza"]
    datas = PersonBase(
        name=random.choice(classes),
        base64=image.base64,
        distance=random.uniform(0, 0.5)
    )
    try:
        new_person = create_person(datas, session)
    except:
        raise HTTPException(status_code=400, detail="impossible to create a new person")    #generic exception (for now)
    return new_person

@app.post("/classify", response_model=PersonPublic, status_code=status.HTTP_201_CREATED)
async def classify_face(face_image: FaceImage, session: SessionDep):
    """
        this is the real classification model endpoint
    """
    try:
        base64.b64decode(face_image.base64, validate=True)
    except:
        raise HTTPException(status_code=422, detail="Base64 field is not a valid Base64")
    
    #converting base64 to image
    if "," in face_image.base64:
        face_image.base64 = face_image.base64.split(",")[1]

    image_bytes = base64.b64decode(face_image.base64)
    np_arr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    #end converting

    try:
        annotated_img, prediction_name, dist = model.classify_face(image)
    except:
        raise HTTPException(status_code=422, detail="the photo hasn't a face. impossible to classify")

    #converting image to base64
    _, buffer = cv2.imencode(".jpg", annotated_img)
    annotated_img_b64 = base64.b64encode(buffer).decode('utf-8')
    #end converting

    datas = PersonBase(
        name=prediction_name,
        base64=annotated_img_b64,
        distance=dist
    )

    try:
        new_person = create_person(datas, session)
    except:
        raise HTTPException(status_code=400, detail="impossible to create a new person")
    return new_person
    
@app.get("/attendance", response_model=list[PersonPublic])
async def read_people(
    session: SessionDep,
    offset: int = 0,
    limit: Annotated[int, Query(le=100)] = 100,
) -> list[Person]:
    people = session.exec(select(Person).offset(offset).limit(limit)).all()
    return people

@app.get("/gallery", response_class=HTMLResponse)
async def view_gallery(request: Request, session: SessionDep):
    statement = select(Person).order_by(Person.date.desc())
    people = session.exec(statement).all()
    
    return templates.TemplateResponse(
        "gallery.html", 
        {"request": request, "people": people}
    )

@app.get("/galery")
async def fake_gallery() -> RedirectResponse: 
    """
    just a joke
    """
    return RedirectResponse(url="https://youtu.be/dQw4w9WgXcQ?si=KeYXsJOF8bPp7N_Q")