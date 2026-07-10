from ultralytics import YOLO
model = YOLO('yolo26x.pt')  # load a pretrained YOLO26x model

results = model.predict(source ='input_videos/08fd33_4.mp4',show =False, conf=0.5, save=True, project='/home/kas/football/outputs', name='test', exist_ok=True)
print(results[0])
print('#########################################')
for box in results[0].boxes:
    print(box)
 