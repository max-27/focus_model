from torch import nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights


class MobileNetV3(nn.Module):
    def __init__(self, transfer: bool = True) -> None:
        super(MobileNetV3, self).__init__()
        self.mv3s_model = mobilenet_v3_small(pretrained=transfer, weights=MobileNet_V3_Small_Weights.DEFAULT)
        self.feature_extractor = nn.Sequential(*list(self.mv3s_model.children())[:-1])
        if transfer:
            # freeze pre-trained mobilenet_v3_small parameters for training
            for param in self.feature_extractor.parameters():
                param.requires_grad = False
        
        self.regressor = nn.Sequential(
            nn.Hardswish(),
            nn.Linear(576, 1),
        )
    
    def initialize_weights(self):
        nn.init.xavier_uniform_(self.model.classifier[-1].weight)

    def forward(self, x):
        batch_size, channels, width, height = x.size()
        x = self.feature_extractor(x)
        x = x.view(batch_size, -1)
        return self.regressor(x)
    

if __name__ == "__main__":
    import torch
    model = MobileNetV3()
    path_to_checkpoint = "/n/data2/hms/dbmi/kyu/lab/maf4031/trained_model/logs/wandb_sweep/runs/2022-12-31_18-16-15/checkpoints"
    model.load_state_dict(torch.load(path_to_checkpoint + "/epoch_086.ckpt", map_location="cpu"))
    model_script = torch.jit.script(model)
    model_script.save("/home/maf4031/focus_model/outputs"+ "/model.pt")