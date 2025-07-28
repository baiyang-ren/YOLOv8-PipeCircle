from ultralytics import YOLO
import torch
model_name = 'yolov8n'
model = YOLO(f"{model_name}.pt")  # This loads a full pretrained model
#Check if the model_name is existing in the "models" directory, if not, save it there, do not use torch


# model.save('models/yolov8n.pt')  # saves to ./models/yolov8n.pt

with open(f"structure_{model_name}.txt", 'w') as f:
    f.write(str(model.model))  # This writes the model architecture to a text file
# print(model.model)  # This prints the entire model architecture
