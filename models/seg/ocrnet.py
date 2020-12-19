import torch
import torch.nn as nn
import torch.nn.functional as F
from models.backbone import build_backbone


def conv1d(inplane, plane):
    layer = [
        nn.Conv1d(inplane, plane, kernel_size=1, bias=False),
        nn.BatchNorm1d(plane),
        nn.ReLU(inplace=True)
    ]

    return nn.Sequential(*layer)


def conv2d(inplane, plane, kernel_size):
    layer = [
        nn.Conv2d(inplane, plane, kernel_size=kernel_size, bias=False),
        nn.BatchNorm2d(plane),
        nn.ReLU(inplace=True)
    ]

    return nn.Sequential(*layer)


class OCRNet(nn.Module):
    def __init__(self, num_cls, backbone='resnet', out_stride=8):
        super(OCRNet, self).__init__()

        self.backbone = build_backbone(backbone, out_stride)

        if backbone == 'resnet':
            low_feat_ch, out_ch = 1024, 2048
        elif backbone == 'hrnet':
            low_feat_ch, out_ch = 768, 1024
        else:
            raise NotImplementedError

        self.conv_3x3 = nn.Sequential(
            nn.Conv2d(out_ch, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512)
        )
        self.dsn_head = nn.Sequential(
            nn.Conv2d(low_feat_ch, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.Dropout2d(0.05),
            nn.Conv2d(512, num_cls, kernel_size=1, stride=1, padding=0, bias=True)
        )

        self.conv1 = conv1d(512, 256)
        self.conv2 = conv1d(512, 256)
        self.conv3 = conv1d(512, 256)
        self.conv4 = conv1d(256, 512)

        self.conv5 = conv2d(512 + 512, 512, 1)
        self.conv6 = conv2d(512, num_cls, 1)

    def forward(self, inpu):
        input_size = inpu.shape[2:]

        low_level_feat, x = self.backbone(inpu, 'ocrnet')

        soft_object_regions = self.dsn_head(low_level_feat)
        pixel_representation = self.conv_3x3(x)

        batch, n_class, height, width = soft_object_regions.shape

        soft_flat = soft_object_regions.view(batch, n_class, -1)
        soft_flat = torch.softmax(soft_flat, -1)

        pixel_flat = pixel_representation.view(batch, pixel_representation.shape[1], -1)

        Obj_Rejion_Rep = (soft_flat @ pixel_flat.transpose(1, 2)).transpose(1, 2)

        tmp1 = self.conv1(Obj_Rejion_Rep).transpose(1, 2)
        tmp2 = self.conv2(pixel_flat)

        pixel_region_relation = torch.softmax(tmp1 @ tmp2, 1)
        tmp3 = self.conv3(Obj_Rejion_Rep)

        Obj_Contextual_Rep = tmp3 @ pixel_region_relation
        Obj_Contextual_Rep = self.conv4(Obj_Contextual_Rep).view(batch, -1, height, width)

        Augmented_Rep = torch.cat([Obj_Contextual_Rep, pixel_representation], 1)
        Augmented_Rep = self.conv5(Augmented_Rep)
        Augmented_Rep = self.conv6(Augmented_Rep)

        out = F.interpolate(Augmented_Rep, size=input_size, mode='bilinear', align_corners=False)

        # if self.training:
        #     aux_out = F.interpolate(soft_object_regions, size=input_size, mode='bilinear', align_corners=False)
        #     return {'out': out, 'aux_out': aux_out}, None
        # else:
        #     return {}, out
        return out


if __name__ == '__main__':
    input = torch.rand(1, 3, 512, 512)
    model = OCRNet(num_cls=2)
    model.train()
    out, _ = model(input)
    print(out['out'].size())
    print(out['aux_out'].size())
    model.eval()
    _, out = model(input)
    print(out.size())
