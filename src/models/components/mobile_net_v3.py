from torch import nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights


class MobileNetV3(nn.Module):
    def __init__(self, transfer: bool = True) -> None:
        super(MobileNetV3, self).__init__()
        self.mv3s_model = mobilenet_v3_small(pretrained=transfer, weights=MobileNet_V3_Small_Weights.DEFAULT)
        self.feature_extractor = nn.Sequential(*list(self.mv3s_model.children())[:-1])
        if transfer:
            # freeze pre-trained mobilenet_v3_small parameters for training
            self.feature_extractor.eval()
            for param in self.feature_extractor.parameters():
                param.requires_grad = False
        
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Hardswish(),
            nn.Linear(576, 1),
        )
        self.model = nn.Sequential(
            self.feature_extractor,
            self.regressor,
        )

    def adjust_classifier_layer(self):
        out_backbone = self.model.classifier[-1].in_features
        self.model.classifier = self.model.classifier[:-1]
        self.model.classifier.add_module("3", nn.Linear(out_backbone, 1))
    
    def initialize_weights(self):
        nn.init.xavier_uniform_(self.model.classifier[-1].weight)

    def forward(self, x):
        return self.model(x)