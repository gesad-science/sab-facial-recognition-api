# API of Facial Recognition Resnet Model
This is a simple API designed to recognize specific faces using a pre-trained **ResNet50** model.

## Pipeline
* A Base64 image is sent to the `/v1/ResNet` endpoint.
* The image is converted to the **RGB** color space.
* The processed image is passed to the **ResNet50** model.
* The model returns the recognition result.
* The API saves the result locally.
* The `/attendance` endpoint is cleared.
* The new result is sent to that same endpoint.

## Execution
1. Run the file `app/main.py`.
2. Within a few moments, the API will start.
3. The endpoint will then be available at:
   https://nonpossibly-aspish-fletcher.ngrok-free.dev

## Requirements
### Python
```
3.11.0
```
### Dependencies
```
fastapi[standard] >= 0.121.0
nest-asyncio >= 1.6.0
pillow >= 12.0.0
pydantic >= 2.12.4
pyngrok >= 7.4.1
torch >= 2.9.0
torchvision >= 0.24.0
```