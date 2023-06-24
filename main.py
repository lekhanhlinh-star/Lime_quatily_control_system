# Import libraries
from typing import Union
import numpy as np
import io
from ultralytics import YOLO
import supervision as sv 
from PIL import Image, ImageOps 
import uvicorn
from classify.model import predict_classification
import numpy as np
import cv2
from fastapi import FastAPI, WebSocket,WebSocketDisconnect
from fastapi.responses import HTMLResponse,RedirectResponse
import websocket
from tools.convert_PIL_base64 import base64_to_pil_image,pil_image_to_base64
from fastapi import FastAPI, Request, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates



model=YOLO(r"weights\best_detect.pt")
box_annotator = sv.BoxAnnotator(
    thickness=2,
    text_thickness=1,
    text_scale=0.7,
)





app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")


templates = Jinja2Templates(directory="templates")



# home page 


@app.get("/")
def read_root(request: Request):
   
    return templates.TemplateResponse("index.html", {"request": request}) # load template index html






# predict websocket api for streaming

@app.websocket("/predict")

async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        try:
            data = await websocket.receive_text()
            img=base64_to_pil_image(data.split(",")[1])
            result=model(img,conf=0.75,iou=0.6)
            detections=sv.Detections.from_yolov8(result[0])
            frame=result[0].orig_img
            box_list = []
            for box in detections.xyxy:
                box_list.append(tuple(np.array(box, np.int64)))

            crop_img_list = []

            for crop_region in box_list:
                crop_img = img.crop(crop_region)
                crop_img=crop_img.convert("RGB")
                crop_img_list.append(crop_img)

            predicted_labels = predict_classification(
                crop_img_list)
            labels = [
                f" Lemon {predicted_labels[idx]}"
                for idx, class_id in enumerate(detections.class_id)
            ]
            if predicted_labels is not None:
                img_pred = box_annotator.annotate(
                    scene=result[0].orig_img, detections=detections, labels=labels
                )
                img_pred=cv2.cvtColor(img_pred,cv2.cv2.COLOR_BGR2RGB)
                
                img_pred = Image.fromarray(img_pred)
               

            else:
                img_pred = box_annotator.annotate(
                    scene=result[0].orig_img, detections=detections
                )
                img_pred=cv2.cvtColor(img_pred,cv2.cv2.COLOR_BGR2RGB)
                
                img_pred = Image.fromarray(img_pred)
         
            result_img =pil_image_to_base64(img_pred).decode("utf-8")
            result_img="data:image/jpeg;base64,"+result_img
            
            
            # process the image and detect objects using OpenCV
            # ...
            # send the detected objects back to the client
            
            await websocket.send_text(result_img)
        except WebSocketDisconnect:
            break
    
    




# upload image and predict

@app.post("/upload-image")
async def predict_file(file: UploadFile = File(...)):
    contents = await file.read()
    
    # process the image data here
    img = Image.open(io.BytesIO(contents))
    # img= ImageOps.equalize(img, mask = None)
    result=model(img,conf=0.65)
    detections=sv.Detections.from_yolov8(result[0])
    
    box_list = []
    for box in detections.xyxy:
        box_list.append(tuple(np.array(box, np.int64)))

    crop_img_list = []

    for crop_region in box_list:
        crop_img = img.crop(crop_region)
        crop_img=crop_img.convert("RGB")
        crop_img_list.append(crop_img)

    predicted_labels = predict_classification(
        crop_img_list)
    labels = [
        f" {model.model.names[class_id]} {predicted_labels[idx]}"
        for idx, class_id in enumerate(detections.class_id)
    ]
    print(labels)
    if predicted_labels is not None:
        img_pred = box_annotator.annotate(
            scene=result[0].orig_img, detections=detections, labels=labels
        )
        img_pred=cv2.cvtColor(img_pred,cv2.cv2.COLOR_BGR2RGB)
        
        img_pred = Image.fromarray(img_pred)
        
    else:
        img_pred = box_annotator.annotate(
            scene=result[0].orig_img, detections=detections
        )
        img_pred=cv2.cvtColor(img_pred,cv2.cv2.COLOR_BGR2RGB)
        
       
        img_pred = Image.fromarray(img_pred)
        

        

    # img_pred.show()
    
    
    
    image_base64 = pil_image_to_base64(img_pred).decode("utf-8")

    # Render the HTML template with the image embedded
    context = {"request": "", "image_data": f"data:image/jpeg;base64,{image_base64}"}
    
    
    return context
    


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=5000)