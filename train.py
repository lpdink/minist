import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from framework.dataset import MyDataSet
from framework.loss import loss_fn
from framework.model import Model
from common.hparams import load_config

def train_loop(dataloader, model, loss_fn, optimizer, device):
    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device)
        # 前向，算结果，算损失
        pred = model(X)
        loss = loss_fn(pred, y)

        # 反向：（在计算本轮梯度前）梯度置0，计算本轮梯度，根据梯度优化。
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"loss {loss.item()} in training")

def test_loop(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred == y).type(torch.float).sum().item()
    print(f"test loss:{test_loss}")

def train(hps, train_data_loader, model, opt, test_data_loader):
    # epochs循环
    for i in range(hps.train.epoch):
        train_loop(train_data_loader,model, loss_fn, opt)
        test_loop(test_data_loader, model, loss_fn)
        print("-"*9+">", i)
    # 保存模型
    # torch.save(model.state_dict(), 'model_weights.pth')
    torch.save(model, hps.model.ckpt)



def main():
    hps = load_config("Your confg path")
    device = hps.train.device
    if device!="cpu":
        assert torch.cuda.is_available()

    # dataset
    train_set = MyDataSet(train=True)
    test_set = MyDataSet(train=False)
    # dataloader
    train_loader = DataLoader(train_set, batch_size=hps.train.batch_size,shuffle=True)
    test_loader = DataLoader(test_set, batch_size=hps.train.batch_size, shuffle=False)
    # model
    model = Model().to(device)
    # or
    # model = torch.load(hps.model.ckpt).to(device)
    # opt
    opt = torch.optim.Adam(model.parameters(), lr=hps.train.learning_rate)

    train(hps,train_loader, model, opt, test_loader)




if __name__=="__main__":
    main()