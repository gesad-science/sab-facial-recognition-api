# SAB Camera Access - Backend

Backend responsible for facial recognition and automatic attendance recording in classrooms.

## Tech Stack

- Python
- FastApi
- SQLModel
- OpenCV
- ngrok
- uv

## Requirements

- Python 3.11+
- ngrok
- uv

## Setup

### 1. Configure ngrok

Create an account at https://ngrok.com and copy your authtoken.

Run: `ngrok config add-authtoken <YOUR_AUTHTOKEN>`

### 2. Install dependencies

`pip install uv`
`uv sync`

### 3. Run the API

terminal 1: `uv run fastapi dev src/main.py`

terminal 2: `ngrok http 8000`

## Endpoints

- `DELETE /restart` - restart database
- `POST /mock/classify` - classify a person with mock method
- `POST /classify` - classify a person with recognize model
- `GET /attendance` - verify all classifications
- `GET /gallery` - frontend to best view of attendance

### Request example of POST /classify

```json
{
    "base64": "<face-crop-base64>"
}
```

### Response exemple of POST /classify

```json
{
    "id": 1,
    "name": "Guilherme",
    "base64": "string",
    "distance": 0.19,
    "date": "2026-03-05T19:59:04.850Z"
}
```

## What is implemented

- FastAPI application setup
- Pydantic models for request validation
- SQLModel database
- `/mock/classify` endpoint
- `/classify` endpoint
- `/attendance` endpoint
- `/restart` endpoint
- `/gallery` endpoint
- Frontend to view classified images (`/gallery`)
- Base64 input validation
- Random classification response (`/mock/classify`)
- Middleware for ngrok HTTPS
- Integration with a ML model

## Work in progress

- Proper error handling and edge cases
- Project structure and file organization
- HTTPS setup (FastAPI Cloud)
- Unit tests