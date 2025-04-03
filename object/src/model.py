import torch
import torch.nn as nn
import timm


class Model(nn.Module):
    def __init__(self, cfg_model: dict, *, pretrained=True, verbose=True):
        super().__init__()

        # Timm encoder
        name = cfg_model['encoder']
        in_channels = 1
        out_channels = 1

        self.encoder = timm.create_model(name,
                                         in_chans=in_channels,
                                         features_only=True,
                                         pretrained=pretrained)
        encoder_channels = self.encoder.feature_info.channels()

        self.segmentation_head = nn.Conv2d(encoder_channels[-1], out_channels,
                                           kernel_size=3, padding=1)

        self.regression_head = nn.Conv2d(encoder_channels[-1], out_channels=2,
                                         kernel_size=3, padding=1)

        self.criterion_seg = nn.BCEWithLogitsLoss()
        self.criterion_reg = nn.MSELoss()

        if verbose:
            print(name)

    def forward(self, img: torch.Tensor):
        """
        img (Tensor): (batch_size, 1, H, W)

        h, w = H // 32, W // 32
        """
        features = self.encoder(img)
        out = features[-1]

        y_pred = self.segmentation_head(out)  # (batch_size, 1, h, w)
        t_pred = self.regression_head(out)    # (batch_size, 1, h, w)

        return y_pred, t_pred

    def loss(self,
             y_pred: torch.Tensor,
             t_pred: torch.Tensor,
             mask: torch.Tensor,
             offset: torch.Tensor) -> tuple[torch.Tensor]:
        """
        Compute segmentation BCE loss + regression MSE loss

        Args:
          y_pred (Tensor[float32]): (batch_size, 1, h, w)
          t_pred (Tensor[float32]): (batch_size, 2, h, w)
          mask   (Tensor[int]):     (batch_size, 1, h, w)
          offset (Tensor[float32]): (batch_size, 2, h, w)
        """
        assert y_pred.shape == mask.shape
        assert t_pred.shape == offset.shape
        lambda_coord = 1

        # Segmentation loss (BCE)
        y = mask.to(torch.float32)
        loss_seg = self.criterion_seg(y_pred, y)

        # Offset loss (MSE)  - THINK sigmoid
        idx = mask.to(bool).expand(-1, 2, -1, -1)
        loss_reg = self.criterion_reg(t_pred[idx], offset[idx])

        loss_total = loss_seg + lambda_coord * loss_reg
        return loss_seg, loss_reg, loss_total
