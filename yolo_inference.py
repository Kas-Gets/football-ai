from ultralytics import YOLO
model = YOLO('models/best.pt')  # load a  YOLO26x model that i trained on google colab

results = model.predict(source ='input_videos/08fd33_4.mp4',show =False, conf=0.5, save=True, project='/home/kas/football/outputs', name='test', exist_ok=True)
print(results[0])
print('#########################################')
for box in results[0].boxes:
    print(box)
 