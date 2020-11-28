import cv2
import torch

cap = cv2.VideoCapture(0)

network = torch.load("CatAndDog.pth")
network.eval()

while True:
    ret,frame = cap.read()
    
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    img2 = cv2.resize(gray,(128,128))
    gray = cv2.resize(gray,(64,64))
    img = cv2.resize(img2,(720,480))
    gray = torch.from_numpy(gray).unsqueeze(dim = 0).unsqueeze(dim = 0)
    #print(gray.shape)
    pred = network(gray.float())
    print("Diff: ",pred[0][0]-pred[0][1])
    pred = torch.argmax(pred,dim = 1)
    #print(type(pred))
    printType = ""
    if pred == 1:
        print("CAT")
        printType = "Cat"
    else:
        print("DOG")
        printType = "Dog"

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, printType, (250, 100), font, 2, (255, 255, 255), 3, cv2.LINE_AA)
    
    cv2.imshow("frame",img)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    
cap.release()
cv2.destroyAllWindows()