from typing import Annotated
from pathlib import Path

from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field as pField
import base64
from datetime import datetime
from sqlmodel import Field as sqlField, Session, SQLModel, create_engine, select ### maybe we can use postgreSQL later

app = FastAPI()

people_present: list[str, any] = []
class PersonBase(SQLModel):
    name: str = sqlField(index=True)                            #index is True to search be easier (filter by name)
    base64: str
    confidence: float

class Person(PersonBase, table=True):
    id: int | None = sqlField(default=None, primary_key=True)   #default is None 'cause we want to be automatic without specify the id
    #date: datetime = sqlField(index=True)                      #need to be automatic
    secret_data: str                                            #just a test

class PersonPublic(PersonBase):
    id: int

class PersonCreate(PersonBase):
    secret_data: str

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


ex_base64 = Path("app/base64example.txt").read_text().strip()
class Image(BaseModel):
    base64: str = pField(default=ex_base64, examples=["iVBORw0KGgoAAAANSUhEUgAAAoAAAA..."])
    model: str = pField(default="ResNet")


@app.on_event("startup")
def on_startup():
    create_db_and_tables()

@app.post("/people/")
def create_person(person: Person, session: SessionDep) -> Person:
    session.add(person)
    session.commit()
    session.refresh(person)
    return person

@app.get("/people/")
def read_people(
    session: SessionDep,
    offset: int = 0,
    limit: Annotated[int, Query(le=100)] = 100,
) -> list[Person]:
    people = session.exec(select(Person).offset(offset).limit(limit)).all()
    return people

@app.get("/people/{person_id}")
def read_person(person_id: int, session: SessionDep) -> Person:
    person = session.get(Person, person_id)
    if not person:
        raise HTTPException(status_code=404, detail="Person not found")
    return person

@app.delete("/people/{person_id}")
def delete_person(person_id: int, session: SessionDep):
    person = session.get(Person, person_id)
    if not person:
        raise HTTPException(status_code=404, detail="Person not found")
    session.delete(person)
    session.commit()
    return {"ok": True}




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