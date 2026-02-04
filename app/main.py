from typing import Annotated
from pathlib import Path

from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field
import base64
from datetime import datetime
from sqlmodel import Field, Session, SQLModel, create_engine, select

ex_base64 = Path("app/base64example.txt").read_text().strip()

app = FastAPI()

people_present: list[str, any] = []
class Person(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(index=True)
    base64: str
    confidence: float
    date: datetime

sqlite_file_name = "database.db"
sqlite_url = f"sqlite:///{sqlite_file_name}"

connect_args = {"check_same_thread": False}
engine = create_engine(sqlite_url, connect_args=connect_args)

def create_db_and_tables():
    SQLModel.metadata.create_all(engine)

def get_session():
    with Session(engine) as session:
        yield session

SessionDep = Annotated[Session, Depends(get_session)]

class Image(BaseModel):
    base64: str = Field(default=ex_base64, examples=["iVBORw0KGgoAAAANSUhEUgAAAoAAAA..."])
    model: str = Field(default="ResNet")

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
    people = session.exec(select(Person).offset(offset).limit(limit).all())
    return people

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