from model.backbone import hrnet, resnet


def build_backbone(backbone, out_stride):
    if backbone == 'resnet':
        return resnet.ResNet101(out_stride)
    elif backbone == 'hrnet':
        return hrnet.HRNet(out_stride)
    else:
        raise NotImplementedError
