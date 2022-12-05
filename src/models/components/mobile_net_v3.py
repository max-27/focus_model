from torch import nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights


class MobileNetV3(nn.Module):
    def __init__(self) -> None:
        super(MobileNetV3, self).__init__()
        self.model = mobilenet_v3_small(pretrained=True, weights=MobileNet_V3_Small_Weights.DEFAULT)
        self.adjust_classifier_layer()
        # self.initialize_weights()

    def adjust_classifier_layer(self):
        out_backbone = self.model.classifier[-1].in_features
        self.model.classifier = self.model.classifier[:-1]
        self.model.classifier.add_module("3", nn.Linear(out_backbone, 1))
    
    def initialize_weights(self):
        nn.init.xavier_uniform_(self.model.classifier[-1].weight)

    def forward(self, x):
        batch_size, channels, width, height = x.size()

        # (batch, 1, width, height) -> (batch, 1*width*height)
        #x = x.view(batch_size, -1)

        return self.model(x)