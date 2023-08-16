import torchvision
import torch.nn as nn
from torch.nn.init import normal_, constant_
from copy import deepcopy


class _Init_Nc_Torchvision:

    def __init__(self):
        self.models = {
            'mobilenet',  # mobilenet_v2, mobilenet_v3_large, mobilenet_v3_small
            'shufflenet',  # shufflenet_v2_x0_5, shufflenet_v2_x1_0, shufflenet_v2_x1_5, shufflenet_v2_x2_0
            'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'resnext50', 'resnext101', # res-series
            'convnext',  # convnext_tiny, convnext_small, convnext_base, convnext_large
            'efficientnet'  # efficientnet_b0 -> efficientnet_b7, efficientnet_v2_s, efficientnet_v2_m, efficientnet_v2_l
        }

    def init_nc(self, model: nn.Module, choice: str, nc: int) -> None:
        model_name = choice.split('_')[0]
        assert model_name in self.models, 'Not supported model'
        if model_name in {'mobilenet', 'convnext', 'efficientnet'}: # self.classify
            m = getattr(model, 'classifier')
            if isinstance(m, nn.Sequential):
                m[-1] = nn.Linear(m[-1].in_features, nc)
        else: # self.fc
            model.fc = nn.Linear(model.fc.in_features, nc)


class MyModel:
    def __init__(self, model_cfgs: dict):
        self.model_cfgs = model_cfgs

        self.kwargs = model_cfgs['kwargs']
        self.num_classes = model_cfgs['num_classes']
        self.pretrained = model_cfgs['pretrained']
        self.backbone_freeze = model_cfgs['backbone_freeze']
        self.bn_freeze = model_cfgs['bn_freeze']
        self.bn_freeze_affine = model_cfgs['bn_freeze_affine']

        model_cfgs_copy = deepcopy(model_cfgs)
        model_cfgs_copy['kind'], model_cfgs_copy['choice'] = model_cfgs['choice'].split('-')

        # init num_classes if torchvision
        self.init_nc_torchvision = _Init_Nc_Torchvision() if model_cfgs_copy['kind'] == 'torchvision' else None
        # init model
        self.model = self.create_model(**model_cfgs_copy)
        del model_cfgs_copy

        if not self.pretrained: self.reset_parameters()

    def create_model(self, choice: str, num_classes: int = 1000, pretrained: bool = False, kind: str = 'torchvision',
                     backbone_freeze: bool = False, bn_freeze: bool = False, bn_freeze_affine: bool = False, **kwargs):
        assert kind in {'torchvision', 'custom'}, 'kind must be torchvision or custom'
        if kind == 'torchvision':
            model = torchvision.models.get_model(choice, weights = torchvision.models.get_model_weights(choice) if pretrained else None, **kwargs['kwargs'])
            # init num_classes from torchvision.models
            if self.init_nc_torchvision is not None:
                self.init_nc_torchvision.init_nc(model, choice, num_classes)

        else:
            pass
        if backbone_freeze: self.freeze_backbone()
        if bn_freeze: self.freeze_bn(bn_freeze_affine)
        return model

    def init_parameters(self, m: nn.Module):
        if isinstance(m, nn.Conv2d):
            normal_(m.weight, mean=0, std=0.02)
        elif isinstance(m, nn.BatchNorm2d):
            constant_(m.weight, 1)
            constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            normal_(m.weight, mean=0, std=0.02)
            constant_(m.bias, 0)

    def reset_parameters(self):
        self.model.apply(self.init_parameters)

    def freeze_backbone(self):
        kind, _ = self.model_cfgs['choice'].split('-')
        if kind == 'torchvision':
            for name, m in self.model.named_children():
                if name != 'fc':
                    for p in m.parameters():
                        p.requires_grad_(False)
            print('backbone freeze')
        else: pass

    def freeze_bn(self, bn_freeze_affine: bool = False):
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                if bn_freeze_affine:
                    m.weight.requires_grad_(False)
                    m.bias.requires_grad_(False)
