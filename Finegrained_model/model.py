import torch
from . import config
from . import LoadModel


def CUB():
    data = 'CUB'
    dataset = 'CUB'
    swap_num = [7, 7]
    backbone = 'resnet50'
    Config = config.LoadConfig(data, dataset, swap_num, backbone, 'test')
    Config.cls_2xmul = True
    model = LoadModel.MainModel(Config)
    model_dict = model.state_dict()
    pretrained_dict = torch.load('pretrained_models/CUB_Res_87.35.pth')
    pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items() if k[7:] in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    data = 'CUB'
    dataset = 'CUB'
    backbone = 'senet154'
    Config = config.LoadConfig(data, dataset, swap_num, backbone, 'test')
    Config.cls_2xmul = True
    model2 = LoadModel.MainModel(Config)
    model2_dict = model2.state_dict()
    pretrained_dict2 = torch.load('pretrained_models/CUB_SENet_86.81.pth')
    pretrained_dict2 = {k[7:]: v for k, v in pretrained_dict2.items() if k[7:] in model2_dict}
    model2_dict.update(pretrained_dict2)
    model2.load_state_dict(model2_dict)

    backbone = 'se_resnet101'
    Config = config.LoadConfig(data, dataset, swap_num, backbone, 'test')
    Config.cls_2xmul = True
    model3 = LoadModel.MainModel(Config)
    model3_dict = model3.state_dict()
    pretrained_dict3 = torch.load('pretrained_models/CUB_SE_86.56.pth')
    pretrained_dict3 = {k[7:]: v for k, v in pretrained_dict3.items() if k[7:] in model3_dict}
    model3_dict.update(pretrained_dict3)
    model3.load_state_dict(model3_dict)

    return model, model2, model3


def CAR():
    data = 'STCAR'
    dataset = 'STCAR'
    swap_num = [7, 7]
    backbone = 'resnet50'
    Config = config.LoadConfig(data, dataset, swap_num, backbone, 'test')

    Config.cls_2xmul = True
    model = LoadModel.MainModel(Config)
    model_dict = model.state_dict()
    pretrained_dict = torch.load('pretrained_models/STCAR_Res_94.35.pth')
    pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items() if k[7:] in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    backbone = 'senet154'
    Config = config.LoadConfig(data, dataset, swap_num, backbone, 'test')
    Config.cls_2xmul = True
    model2 = LoadModel.MainModel(Config)
    model2_dict = model2.state_dict()
    pretrained_dict2 = torch.load('pretrained_models/STCAR_SENet_93.36.pth')
    pretrained_dict2 = {k[7:]: v for k, v in pretrained_dict2.items() if k[7:] in model2_dict}
    model2_dict.update(pretrained_dict2)
    model2.load_state_dict(model2_dict)

    backbone = 'se_resnet101'
    Config = config.LoadConfig(data, dataset, swap_num, backbone, 'test')
    Config.cls_2xmul = True
    model3 = LoadModel.MainModel(Config)
    model3_dict = model3.state_dict()
    pretrained_dict3 = torch.load('pretrained_models/STCAR_SE_92.97.pth')
    pretrained_dict3 = {k[7:]: v for k, v in pretrained_dict3.items() if k[7:] in model3_dict}
    model3_dict.update(pretrained_dict3)
    model3.load_state_dict(model3_dict)

    return model, model2, model3
