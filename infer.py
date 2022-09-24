from framework.model import Model
import torch


def infer():
    device="cpu"
    model = torch.load("model.pth").to(device)
    input_data = [1]
    with torch.no_grad():
        input_tensor = torch.tensor(input_data).unsqueeze(1).float()
        pred = model(input_tensor)
        print(pred)

if __name__=="__main__":
    infer()