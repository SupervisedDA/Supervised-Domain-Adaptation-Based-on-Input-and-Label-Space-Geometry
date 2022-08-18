import torch
import torch.nn as nn
import torchvision
import timm
from Utilities.TLIB_Utils import ImageClassifier
import TLIB.common.vision.models as models

##
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)

class DigitsBackbone(torch.nn.Module):
    def __init__(self, device,n_classes=10,hidden_dim=1152):
        super(DigitsBackbone, self).__init__()
        self.device = device
        self.phi= nn.Sequential(
            nn.Conv2d(1, 32, 3),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 3),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Dropout(0.5),
            Flatten(),
            nn.Linear(hidden_dim, 120),
            nn.ReLU(True),
            nn.Linear(120, 84),
            nn.ReLU(True)
        )

        self.hypothesis = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(84, n_classes),
        )
        # self.pool_layer=nn.Identity()
        # self.backbone = self.forward_source

    def forward(self, x):
        hiddens=self.phi(x)
        outputs=self.hypothesis(hiddens)
        if  not(self.training):
            return outputs
        else:
            return outputs, hiddens

##

def get_model(model_name, pretrain=True):
    if model_name in models.__dict__:
        # load models from common.vision.models
        backbone = models.__dict__[model_name](pretrained=pretrain)
    else:
        # load models from pytorch-image-models
        backbone = timm.create_model(model_name, pretrained=pretrain)
        try:
            backbone.out_features = backbone.get_classifier().in_features
            backbone.reset_classifier(0, '')
        except:
            backbone.out_features = backbone.head.in_features
            backbone.head = nn.Identity()
    return backbone

##
def GetModel(hp,device):
    if hp.Src.startswith('M') or hp.Src.startswith('U'):
        net = DigitsBackbone(device=device, n_classes=10)
    if hp.Src in ['A','W','D']:
            backbone = get_model('vgg16', pretrain=True)
            net = ImageClassifier(backbone, 31, bottleneck_dim=256,
                                  pool_layer=None, finetune=True)
            net.backbone = torchvision.models.vgg16(pretrained=True).features
            net.bottleneck[0] = nn.Linear(512, 256)

    # elif hp.Src.startswith('CCM') or hp.Src.startswith('CM'):
    #     net = CMNIST_HiddenDomainAlignment(device, n_classes)
    # elif hp.Src.startswith('ML'):
    #         n_ml = int(hp.Src[5:])
    #         if n_ml==5:
    #             net = HiddenDomainAlignment(device, n_classes=n_ml, hidden_dim=7296)
    #         elif n_ml==7:
    #             net = HiddenDomainAlignment(device, n_classes=n_ml, hidden_dim=10368)
    #         elif n_ml==10:
    #             net = HiddenDomainAlignment(device, n_classes=n_ml, hidden_dim=14976)
    #
    # elif hp.Src.startswith('CiCa'):
    #     backbone = get_model(args.arch, pretrain=not args.scratch)
    #     pool_layer = None
    #     net = ImageClassifier(backbone, 42, bottleneck_dim=args.bottleneck_dim,
    #                               pool_layer=pool_layer, finetune=not args.scratch).to(device)
    #     net.head=nn.Sequential(
    #         nn.Linear( args.bottleneck_dim, 1)
    #     )


    return net