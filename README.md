# FastAPI – API Rewrite

This branch contains an initial rewrite of the existing API using FastAPI.
The goal is to study the framework, its best practices, and gradually
rebuild the current endpoints.

## What is implemented

- FastAPI application setup
- Pydantic models for request validation
- `/classify` endpoint (stub implementation)
- Base64 input validation
- Basic response structure for classification
- `/attendance` endpoint
- Part of the database made

## Work in progress

- Real image decoding and preprocessing
- Integration with actual ML model
- Proper error handling and edge cases
- Finalize the database integration (SQL)
- Project structure and file organization
- HTTPS setup (as recommended by FastAPI documentation)
- Input/output schema refinement
- Unit tests
- Logging

## Notes

- This is a study-driven rewrite.
- Current classification logic is mocked.
- Code structure and patterns may change as the study progresses.