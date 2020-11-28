import torch
import matplotlib.pyplot as plt
t = [0,1,2,3,4,5]
print(torch.version)

plt.plot(t)
plt.show()

shoudlSave = input("Save the model? Y/n")

if shoudlSave=="y":
    print("save Model")
    torch.save()
else:
    print("Cancel")