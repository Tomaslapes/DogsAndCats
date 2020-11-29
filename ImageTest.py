import torch
import cv2
import os

model = torch.load("CatAndDog.pth")
model.eval()

for file in os.listdir("DemoImages"):
    image = cv2.imread(f"DemoImages/{file}")
    image = cv2.resize(image,(128*4,128*4))
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    predImg = torch.from_numpy(cv2.resize(image,(64,64))).unsqueeze(dim = 0).unsqueeze(dim = 0)

    pred = model(predImg.float())
    prediction = torch.argmax(pred)
    print(prediction)
    classes = ["Dog","Cat"]

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image,classes[prediction.item()],(250,100),font,2,(255,255,255),3,cv2.LINE_AA)

    cv2.imshow("Image",image)

    if cv2.waitKey(0) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()


