from typing import Annotated
from pathlib import Path
import random

from fastapi import FastAPI, HTTPException, Depends, Query, status
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field as pField
import base64
from datetime import datetime
from sqlmodel import Field as sqlField, Session, SQLModel, create_engine, select, delete ### maybe we can use postgreSQL later

app = FastAPI()

#classes
class PersonBase(SQLModel):
    name: str = sqlField(index=True)                            #index is True to search be easier (filter by name)
    base64: str
    confidence: float

class Person(PersonBase, table=True):
    id: int | None = sqlField(default=None, primary_key=True)   #default is None 'cause we want to be automatic without specify the id
    date: datetime = sqlField(index=True, default_factory=datetime.utcnow)

class PersonPublic(PersonBase):
    id: int
    date: datetime = sqlField(index=True, default_factory=datetime.utcnow)

ex_base64 = Path("app/base64example.txt").read_text().strip()
class Image(BaseModel):
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

@app.on_event("startup")
def on_startup():
    create_db_and_tables()
#"end" start database

#functions and endpoints
def create_person(person: PersonBase, session: SessionDep):
    db_person = Person.model_validate(person)
    session.add(db_person)
    session.commit()
    session.refresh(db_person)
    return db_person

@app.delete("/restart/", status_code=status.HTTP_204_NO_CONTENT)
def delete_person(session: SessionDep):
    with Session(engine) as session:
        session.exec(delete(Person))
        session.commit()
        return

@app.post("/mock/classify/", response_model=PersonPublic, status_code=status.HTTP_201_CREATED)            #we don't actually return a PersonPublic so db_person will be ajusted to a PersonPublic ('cause of "responde_model=")
async def classify_face(image: Image, session: SessionDep):
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
        base64=random.choice([ex_base64, image.base64]),
        confidence=random.uniform(0.5, 1)
    )
    try:
        new_person = create_person(datas, session)
    except:
        raise HTTPException(status_code=400, detail="impossible to create a new person")    #generic exception (for now)
    return new_person
    
@app.get("/attendance/", response_model=list[PersonPublic])
def read_people(
    session: SessionDep,
    offset: int = 0,
    limit: Annotated[int, Query(le=100)] = 100,
) -> list[Person]:
    people = session.exec(select(Person).offset(offset).limit(limit)).all()
    return people

@app.get("/attendence")
async def fake_attendance() -> RedirectResponse: 
    return RedirectResponse(url="https://youtu.be/dQw4w9WgXcQ?si=KeYXsJOF8bPp7N_Q")