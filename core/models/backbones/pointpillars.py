import torch
import torch.nn as nn
import torch.nn.functional as F


class PointPillarsBackbone(nn.Module):
    """
    BaseBEVBackbone — implementation based on the PointPillars paper
    (Lang et al., 2019, arXiv:1812.05784), Section 3.2.

    Paper architecture (original):
      - 3-stage top-down backbone: Block1(S=2,L=3,C=128), Block2(S=2,L=5,C=128), Block3(S=2,L=5,C=256)
      - 3 ConvTranspose2d upsample branches with upsample_strides=[1,2,4] → all output at H/2
      - Concat → H/2, 768ch

    Adaptation for this repo (PIXOR-style dense head, out_size_factor=4):
      - Target resolution is H/4 (not H/2 as in paper)
      - neck1: Block1 (H/2) → H/4 via strided Conv2d (stride=2) [projection, not upsample]
      - neck2: Block2 (H/4) → H/4 via ConvTranspose2d (stride=1) [matches paper spirit]
      - neck3: Block3 (H/8) → H/4 via ConvTranspose2d (stride=2) [matches paper spirit]
      - Concat → H/4, 384ch  (backbone_out_dim=384 in config)

    References:
      - Paper:  https://arxiv.org/abs/1812.05784
      - Official impl: github.com/nutonomy/second.pytorch (class RPN)
    """

    def __init__(
        self,
        in_channels: int = 30,
        out_channels: list = [64, 128, 256],   # Paper: [128, 128, 256]
        layer_nums:   list = [3, 5, 5],          # Paper: [3, 5, 5]
        layer_strides: list = [2, 2, 2],          # Paper: [2, 2, 2]
        num_upsample_filters: list = [128, 128, 128],  # Paper uses 256; kept 128 for 384ch total
        upsample_strides:     list = [2, 1, 2],         # Effective strides for H/4 target
    ):
        super(PointPillarsBackbone, self).__init__()

        # ── Top-down backbone ────────────────────────────────────────────
        # Block 1: in → C[0], stride S[0]=2  →  H/2
        self.block1 = nn.Sequential(
            nn.ZeroPad2d(1),
            nn.Conv2d(in_channels, out_channels[0], 3, stride=layer_strides[0], bias=False),
            nn.BatchNorm2d(out_channels[0], eps=1e-3, momentum=0.01),
            nn.ReLU(inplace=True),
        )
        for i in range(layer_nums[0]):
            self.block1.add_module(f"conv_{i}", nn.Sequential(
                nn.ZeroPad2d(1),
                nn.Conv2d(out_channels[0], out_channels[0], 3, bias=False),
                nn.BatchNorm2d(out_channels[0], eps=1e-3, momentum=0.01),
                nn.ReLU(inplace=True),
            ))

        # Block 2: C[0] → C[1], stride S[1]=2  →  H/4
        self.block2 = nn.Sequential(
            nn.ZeroPad2d(1),
            nn.Conv2d(out_channels[0], out_channels[1], 3, stride=layer_strides[1], bias=False),
            nn.BatchNorm2d(out_channels[1], eps=1e-3, momentum=0.01),
            nn.ReLU(inplace=True),
        )
        for i in range(layer_nums[1]):
            self.block2.add_module(f"conv_{i}", nn.Sequential(
                nn.ZeroPad2d(1),
                nn.Conv2d(out_channels[1], out_channels[1], 3, bias=False),
                nn.BatchNorm2d(out_channels[1], eps=1e-3, momentum=0.01),
                nn.ReLU(inplace=True),
            ))

        # Block 3: C[1] → C[2], stride S[2]=2  →  H/8
        self.block3 = nn.Sequential(
            nn.ZeroPad2d(1),
            nn.Conv2d(out_channels[1], out_channels[2], 3, stride=layer_strides[2], bias=False),
            nn.BatchNorm2d(out_channels[2], eps=1e-3, momentum=0.01),
            nn.ReLU(inplace=True),
        )
        for i in range(layer_nums[2]):
            self.block3.add_module(f"conv_{i}", nn.Sequential(
                nn.ZeroPad2d(1),
                nn.Conv2d(out_channels[2], out_channels[2], 3, bias=False),
                nn.BatchNorm2d(out_channels[2], eps=1e-3, momentum=0.01),
                nn.ReLU(inplace=True),
            ))

        # ── Neck (upsample / project to common H/4 scale) ────────────────
        # neck1: Block1 is at H/2  → project DOWN to H/4 via strided Conv2d (stride=2).
        #   NOTE: paper targets H/2 with ConvTranspose2d(stride=1); here we adapt
        #   to H/4 because the PIXOR dense head requires out_size_factor=4.
        self.neck1 = nn.Sequential(
            nn.Conv2d(out_channels[0], num_upsample_filters[0], 3,
                      stride=upsample_strides[0], padding=1, bias=False),
            nn.BatchNorm2d(num_upsample_filters[0], eps=1e-3, momentum=0.01),
            nn.ReLU(inplace=True),
        )

        # neck2: Block2 is already at H/4  → ConvTranspose2d(stride=1) = lateral projection.
        #   Matches paper's ConvTranspose2d(stride=up_strides[1]) spirit.
        self.neck2 = nn.Sequential(
            nn.ConvTranspose2d(out_channels[1], num_upsample_filters[1],
                               upsample_strides[1], stride=upsample_strides[1], bias=False),
            nn.BatchNorm2d(num_upsample_filters[1], eps=1e-3, momentum=0.01),
            nn.ReLU(inplace=True),
        )

        # neck3: Block3 is at H/8  → ConvTranspose2d(stride=2) → H/4.
        #   Matches paper's ConvTranspose2d(stride=up_strides[2]) spirit.
        self.neck3 = nn.Sequential(
            nn.ConvTranspose2d(out_channels[2], num_upsample_filters[2],
                               upsample_strides[2], stride=upsample_strides[2], bias=False),
            nn.BatchNorm2d(num_upsample_filters[2], eps=1e-3, momentum=0.01),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # Top-down feature extraction
        x1 = self.block1(x)   # [B, C0, H/2,  W/2]
        x2 = self.block2(x1)  # [B, C1, H/4,  W/4]
        x3 = self.block3(x2)  # [B, C2, H/8,  W/8]

        # Neck: bring all to H/4
        up1 = self.neck1(x1)  # [B, F0, H/4, W/4]  (strided projection)
        up2 = self.neck2(x2)  # [B, F1, H/4, W/4]  (ConvTranspose2d stride=1)
        up3 = self.neck3(x3)  # [B, F2, H/4, W/4]  (ConvTranspose2d stride=2)

        # Concatenate multi-scale features  (paper: concat then feed to head)
        out = torch.cat([up1, up2, up3], dim=1)  # [B, F0+F1+F2, H/4, W/4] = 384ch
        return out


if __name__ == "__main__":
    model = PointPillarsBackbone()
    x = torch.randn(2, 30, 480, 480)
    out = model(x)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {out.shape}")
    # Expected: [2, 384, 120, 120]
