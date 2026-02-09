# FastAPI – API Rewrite

This branch contains an initial rewrite of the existing API using FastAPI.
The goal is to study the framework, its best practices, and gradually
rebuild the current endpoints.

## To run

`uv run fastapi dev app/main.py`

## What is implemented

- FastAPI application setup
- Pydantic models for request validation
- SQLModel database
- `/mock/classify` endpoint
- `/attendance` endpoint
- `/restart` endpoint
- Base64 input validation
- Random classification response (mock)

## Work in progress

- Integration with a ML model
- Proper error handling and edge cases
- Project structure and file organization
- HTTPS setup (FastAPI Cloud)
- Unit tests

## Notes

- This is a study-driven rewrite.
- Current classification logic is mocked.
- Code structure and patterns may change as the study progresses.