from torch.utils.data import Dataset
import numpy as np

def get_train_data():
    pass

def get_test_data():
    pass

class MyDataSet(Dataset):
    # 参数与官方数据集保持一致。
    # train参数表示生成训练或测试数据集。
    # transform传入一个function,对feature施加处理。
    # target_transform 传入一个function,对label施加处理
    def __init__(self, train=True) -> None:
        if train:
            self.data, self.label=get_train_data()
        else:
            self.data, self.label=get_test_data()
        assert len(self.data)==len(self.label)
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.label[index]