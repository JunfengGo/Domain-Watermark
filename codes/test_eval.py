import torch

tlabel=4

path_cifar = 'tensor_path/tensor_val.pt'
    
source_images_cifar = torch.load(path_cifar)

indices_cifar = torch.load("./indices.pth")
  
source_images_cifar = torch.stack([source_images_cifar[index] 
for index in indices_cifar])


model_w = torch.load("./saved_models/model_DW.pth")

model_b = torch.load("./saved_models/model_benign_1.pth")




print("Watermark Model's VSR:\n")
print(torch.sum(model_w(source_images_cifar).max(1)[1]==tlabel)/len(source_images_cifar))




print("Benign Model's VSR:\n")
print(torch.sum(model_b(source_images_cifar).max(1)[1]==tlabel)/len(source_images_cifar))