#!/usr/bin/python
#
# Copyright 2022 Azade Farshad
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License
import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)


from collections import OrderedDict
from typing import Tuple

import torch
import torch.nn as nn

from src.models.components.ynet import YNet_general
from src.models.components.ffc import FFC_BN_ACT, ConcatTupleLayer


class YNet_spatial(nn.Module):

    def __init__(self, in_channels=3, out_channels=1, init_features=32, ratio_in=0.5, ffc=True, skip_ffc=False,
                 cat_merge=True, activation_function="relu", **kwargs):
        super().__init__()
        self.ffc = ffc
        self.skip_ffc = skip_ffc
        self.ratio_in = ratio_in
        self.cat_merge = cat_merge

        features = init_features
        ############### Regular1 ##################################
        self.encoder1_1 = YNet_general._block(in_channels, features, name="enc1")
        self.pool1_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2_1 = YNet_general._block(features, features * 2, name="enc2")  # was 1,2
        self.pool2_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3_1 = YNet_general._block(features * 2, features * 4, name="enc3")
        self.pool3_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4_1 = YNet_general._block(features * 4, features * 4, name="enc4")  # was 8
        self.pool4_1 = nn.MaxPool2d(kernel_size=2, stride=2)

        ############### Regular2 ##################################
        self.encoder1_2 = YNet_general._block(in_channels, features, name="enc1")
        self.pool1_2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2_2 = YNet_general._block(features, features * 2, name="enc2")  # was 1,2
        self.pool2_2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3_2 = YNet_general._block(features * 2, features * 4, name="enc3")
        self.pool3_2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4_2 = YNet_general._block(features * 4, features * 4, name="enc4")  # was 8
        self.pool4_2 = nn.MaxPool2d(kernel_size=2, stride=2)      

        self.bottleneck = YNet_general._block(features * 8, features * 16, name="bottleneck")  # 8, 16

        if activation_function == "sigmoid":
            act_func = nn.Sigmoid()
        elif activation_function == "relu":
            act_func = nn.ReLU()
        elif activation_function == "tanh":
            act_func = nn.Tanh()

        self.regressor = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            act_func,
            nn.Linear(512, 1),
        )

    def apply_fft(self, inp, batch):
        ffted = torch.fft.fftn(inp)
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)

        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])
        return ffted

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]):
        batch = x.shape[0]
        # Spatial encoder 1
        enc1_1 = self.encoder1_1(x)
        enc2_1 = self.encoder2_1(self.pool1_1(enc1_1))

        enc3_1 = self.encoder3_1(self.pool2_1(enc2_1))

        enc4_1 = self.encoder4_1(self.pool3_1(enc3_1))
        enc4_1 = self.pool4_1(enc4_1)

        # Spatial encoder 2
        enc1_2 = self.encoder1_2(x)
        enc2_2 = self.encoder2_2(self.pool1_2(enc1_2))

        enc3_2 = self.encoder3_2(self.pool2_2(enc2_2))

        enc4_2 = self.encoder4_2(self.pool3_2(enc3_2))
        enc4_2 = self.pool4_2(enc4_2)

        # Bottleneck
        if self.cat_merge:
            a = torch.zeros_like(enc4_1)
            b = torch.zeros_like(enc4_2)

            enc4_1 = enc4_1.view(torch.numel(enc4_1), 1)
            enc4_2_2 = enc4_2_2.view(torch.numel(enc4_2), 1)

            bottleneck = torch.cat((enc4_1, enc4_2), 1)
            bottleneck = bottleneck.view_as(torch.cat((a, b), 1))

        else:
            bottleneck = torch.cat((enc4_1, enc4_2), 1)

        bottleneck = self.bottleneck(bottleneck)
        return self.regressor(bottleneck)

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )


if __name__ == "__main__":
    net = YNet_simplified()
    img = torch.rand((1, 3, 240, 240))
    out = net(img)
    print(out)