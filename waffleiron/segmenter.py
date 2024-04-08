# Copyright 2022 - Valeo Comfort and Driving Assistance - Gilles Puy @ valeo.ai
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
# limitations under the License.


import torch.nn as nn
from .backbone import WaffleIron
from .embedding import Embedding
import torch

class Segmenter(nn.Module):
    def __init__(
        self,
        input_channels,
        feat_channels,
        nb_class,
        depth,
        grid_shape,
        drop_path_prob=0,
        layer_norm=False,
    ):
        super().__init__()
        # Embedding layer
        self.embed = Embedding(input_channels, feat_channels)
        # WaffleIron backbone
        self.waffleiron = WaffleIron(feat_channels, depth, grid_shape, drop_path_prob, layer_norm)
        # Classification layer
        self.classif = nn.Conv1d(feat_channels, nb_class, 1)

    def compress(self):
        self.embed.compress()
        self.waffleiron.compress()

    def forward(self, feats, cell_ind, occupied_cell, neighbors):
        tokens, local_features = self.embed(feats, neighbors)
        tokens = self.waffleiron(tokens, cell_ind, occupied_cell)

        # B, C, N = tokens.shape

        #gather = []

        #for ind_nn in range(
        #    1, neighbors.shape[1]
        #):  # Remove first neighbors which is the center point
        #    temp = neighbors[:, ind_nn : ind_nn + 1, :].expand(-1, tokens.shape[1], -1)
        #    gather.append(torch.gather(tokens, 2, temp).unsqueeze(-1))
        #neighbor_gather = torch.cat(gather, -1).mean(-1).reshape(B, C, N)

        #return self.classif(torch.cat((tokens, neighbor_gather), dim=1))

        return self.classif(tokens + local_features)
