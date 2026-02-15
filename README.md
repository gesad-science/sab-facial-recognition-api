# FastAPI – API Rewrite

This branch contains an initial rewrite of the existing API using FastAPI.

The goal is to study the framework, its best practices, and gradually
rebuild the current endpoints.

## To run

FastAPI: `uv run fastapi dev src/main.py`

NGROK: `ngrok http 8000`

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
- Random classification response (mock)
- Middleware for ngrok HTTPS
- Integration with a ML model

## Work in progress

- Proper error handling and edge cases
- Project structure and file organization
- HTTPS setup (FastAPI Cloud)
- Unit tests

## Notes

- This is a study-driven rewrite.
- Current classification logic is mocked.
- Code structure and patterns may change as the study progresses.