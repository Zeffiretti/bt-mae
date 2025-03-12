from functools import partial
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

from models_mae import MaskedAutoencoderDeiT
from util import misc
from timm.models.vision_transformer import PatchEmbed, Block, Attention, Mlp, DropPath

from util.pos_embed import get_2d_sincos_pos_embed


def compute_hog_pytorch(img, num_bins=9, cell_size=8):
    """
    Computes a differentiable HOG feature in PyTorch.

    Args:
        img (torch.Tensor): Input tensor of shape (N, C, H, W) with requires_grad=True
        num_bins (int): Number of orientation bins
        cell_size (int): Size of each cell (HOG parameter)

    Returns:
        hog_features (torch.Tensor): Extracted HOG features
    """

    # Convert to grayscale if RGB
    if img.shape[1] == 3:  # Assume (N, C, H, W)
        img = 0.299 * img[:, 0, :, :] + 0.587 * img[:, 1, :, :] + 0.114 * img[:, 2, :, :]  # Convert to grayscale
        img = img.unsqueeze(1)  # Restore channel dimension (N, 1, H, W)

    # Compute gradient using Sobel filter
    sobel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).view(1, 1, 3, 3).to(img.device)
    sobel_y = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]).view(1, 1, 3, 3).to(img.device)

    gx = F.conv2d(img, sobel_x, padding=1)
    gy = F.conv2d(img, sobel_y, padding=1)

    # Compute gradient magnitude and orientation
    magnitude = torch.sqrt(gx**2 + gy**2 + 1e-6)  # Avoid division by zero
    orientation = torch.atan2(gy, gx)  # Angle in radians

    # Convert orientation to histogram bins (0 to π)
    orientation_bins = torch.floor((orientation + np.pi) * (num_bins / (2 * np.pi))).long()
    orientation_bins = torch.clamp(orientation_bins, 0, num_bins - 1)

    # Compute HOG features per cell
    h, w = img.shape[2], img.shape[3]
    cells_x, cells_y = h // cell_size, w // cell_size
    hog_features = torch.zeros((img.shape[0], num_bins, cells_x, cells_y), device=img.device)

    for i in range(cells_x):
        for j in range(cells_y):
            cell_mag = magnitude[:, :, i * cell_size : (i + 1) * cell_size, j * cell_size : (j + 1) * cell_size]
            cell_ori = orientation_bins[:, :, i * cell_size : (i + 1) * cell_size, j * cell_size : (j + 1) * cell_size]

            for b in range(num_bins):
                # Ensure correct dimension after summation
                hog_features[:, b, i, j] = torch.sum(cell_mag * (cell_ori == b).float(), dim=(2, 3)).squeeze(-1)

    return hog_features.view(img.shape[0], -1)  # Flatten to vector


class BootMAEDeiT(MaskedAutoencoderDeiT):
    """Bootstrapped Masked Autoencoder with DeiT backbone

    This implementation extends the original MAE to use features from a pretrained
    MAE encoder as the reconstruction target instead of pixel values.
    """

    def __init__(
        self,
        target_encoder=None,  # Pretrained MAE encoder to extract target features
        target_layer_index=-1,  # Which encoder layer to extract features from (-1 means the last layer)
        feature_dim=None,  # Dimension of target encoder features (if None, use patch_size^2 * in_chans)
        img_size=32,
        patch_size=4,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
        norm_pix_loss=False,
        # Deit specific params
        feature_class="latent",  # latent or hog
        layer_scale_init_value=1e-5,
        drop_path_rate=0.0,
        enable_ema=False,
        use_new_feature_predictor=False,
    ):
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            decoder_embed_dim=decoder_embed_dim,
            decoder_depth=decoder_depth,
            decoder_num_heads=decoder_num_heads,
            mlp_ratio=mlp_ratio,
            norm_layer=norm_layer,
            norm_pix_loss=norm_pix_loss,
            # Deit specific params
            layer_scale_init_value=layer_scale_init_value,
            drop_path_rate=drop_path_rate,
        )

        self.target_encoder = target_encoder
        self.target_layer_index = target_layer_index
        self.feature_class = feature_class

        self.enable_ema = enable_ema
        self.use_new_feature_decoder = use_new_feature_predictor

        if self.feature_class == "hog":
            print("Using HOG features")
        #     feature_dim = get_hog_feature_size(img_size)
        #     print(f"Feature dimension for hog: {feature_dim}")
        #     # self.decoder_pred = nn.Linear(self.decoder_blocks[-1].norm2.normalized_shape[0], feature_dim, bias=True)

        #     # initialize nn.Linear and nn.LayerNorm
        #     self.apply(self._init_weights)

        if target_encoder is not None:
            if feature_dim is None:
                if self.feature_class == "latent":
                    # Default to the same dimension as the encoder output
                    feature_dim = target_encoder.blocks[target_layer_index].norm2.normalized_shape[0]
                elif self.feature_class == "hog":
                    feature_dim = patch_size**2 * in_chans
            if not self.use_new_feature_decoder:
                # Get feature dimension from the target encoder if not provided

                # Replace the decoder prediction head to match target feature dimensions
                self.decoder_pred = nn.Linear(self.decoder_blocks[-1].norm2.normalized_shape[0], feature_dim, bias=True)

                # initialize nn.Linear and nn.LayerNorm
                self.apply(self._init_weights)

                # self.init_feature_extraction_weights()

                # Freeze the target encoder
                for param in self.target_encoder.parameters():
                    param.requires_grad = False
            else:
                # build new feature predictor
                self.feature_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
                self.feature_mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

                # self.apply(self._init_weights)
                self.feature_pos_embed = nn.Parameter(
                    torch.zeros(1, self.patch_embed.num_patches + 1, decoder_embed_dim), requires_grad=False
                )
                feature_depth = decoder_depth // 2
                feature_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, feature_depth)]
                self.feature_blocks = nn.ModuleList(
                    [
                        Block(
                            dim=decoder_embed_dim,
                            num_heads=decoder_num_heads,
                            mlp_ratio=mlp_ratio,
                            qkv_bias=True,
                            drop=drop_path_rate,
                            attn_drop=drop_path_rate,
                            drop_path=dpr,
                            norm_layer=norm_layer,
                        )
                        for dpr in feature_dpr
                    ]
                )
                self.feature_norm = norm_layer(decoder_embed_dim)
                self.feature_pred = nn.Linear(decoder_embed_dim, feature_dim, bias=True)

                torch.nn.init.normal_(self.feature_mask_token, std=0.02)

                self.init_feature_extraction_weights()

        if self.enable_ema:
            # To keep MAE reconstruction ability, we need to keep the target decoder to learn pixel reconstruction
            # Meanwhile, we create new feature decoder to learn feature reconstruction (a shallow decoder)
            self.use_new_feature_decoder = True
            self.target_encoder = None

            if feature_dim is None:
                feature_dim = self.blocks[target_layer_index].norm2.normalized_shape[0]

            self.feature_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
            self.feature_mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

            # self.apply(self._init_weights)
            self.feature_pos_embed = nn.Parameter(
                torch.zeros(1, self.patch_embed.num_patches + 1, decoder_embed_dim), requires_grad=False
            )
            feature_depth = decoder_depth // 2
            feature_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, feature_depth)]
            self.feature_blocks = nn.ModuleList(
                [
                    Block(
                        dim=decoder_embed_dim,
                        num_heads=decoder_num_heads,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=True,
                        drop=drop_path_rate,
                        attn_drop=drop_path_rate,
                        drop_path=dpr,
                        norm_layer=norm_layer,
                    )
                    for dpr in feature_dpr
                ]
            )
            self.feature_norm = norm_layer(decoder_embed_dim)
            self.feature_pred = nn.Linear(decoder_embed_dim, feature_dim, bias=True)

            torch.nn.init.normal_(self.feature_mask_token, std=0.02)

            self.init_feature_extraction_weights()

    def init_feature_extraction_weights(self):
        feature_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1],
            int((self.patch_embed.num_patches + 2) ** 0.5),
            cls_token=True,
            distill_token=False,
        )
        self.feature_pos_embed.copy_(torch.from_numpy(feature_pos_embed).float().unsqueeze(0))

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def extract_target_features(self, imgs, mask, ids_restore=None):
        """Extract features from the target encoder to use as reconstruction targets"""
        # if self.target_encoder is None:
        #     # If no target encoder is provided, use pixel values as in original MAE
        #     return self.patchify(imgs)

        # target_encoder = self.target_encoder
        with torch.no_grad():
            self.eval()
            x = self.patch_embed(imgs)
            cls_token = self.cls_token + self.pos_embed[:, :1, :]
            x = torch.cat((cls_token.expand(x.shape[0], -1, -1), x), dim=1)

            for i, blk in enumerate(self.blocks):
                x = blk(x)

            return x[:, 1:]

    def extract_midlayer_features(self, imgs):
        if self.target_encoder is None:
            # If no target encoder is provided, use pixel values as in original MAE
            return self.patchify(imgs)

        with torch.no_grad():
            if not self.enable_ema:  # standard bootstrapped MAE
                # Forward pass through target encoder without masking (mask_ratio=0)
                self.target_encoder.eval()
                x = self.target_encoder.patch_embed(imgs) + self.target_encoder.pos_embed[:, 2:, :]

                # No masking for target feature extraction
                # Append cls token
                cls_token = self.target_encoder.cls_token + self.target_encoder.pos_embed[:, :1, :]
                x = torch.cat((cls_token.expand(x.shape[0], -1, -1), x), dim=1)

                # Pass through encoder blocks up to the target layer
                for i, blk in enumerate(self.target_encoder.blocks):
                    x = blk(x)
                    if i == self.target_layer_index:
                        break

                # Apply norm layer
                features = self.target_encoder.norm(x)

                # Return only patch features (exclude cls token)
                return features[:, 1:, :]
            else:  # bootstrapped MAE with EMA target encoder
                ema = self.target_encoder.ema
                ema.eval()
                x = ema.patch_embed(imgs) + ema.pos_embed[:, 2:, :]
                cls_token = ema.cls_token + ema.pos_embed[:, :1, :]
                x = torch.cat((cls_token.expand(x.shape[0], -1, -1), x), dim=1)

                for i, blk in enumerate(ema.blocks):
                    x = blk(x)
                    if i == self.target_layer_index:
                        break

                features = ema.norm(x)

                return features[:, 1:, :]

    def extract_hog_features(self, imgs):
        return compute_hog_pytorch(imgs)

    def extract_features(self, imgs):
        if self.feature_class == "latent":
            return self.extract_midlayer_features(imgs)
        if self.feature_class == "hog":
            return self.extract_hog_features(imgs)
        raise ValueError(f"Unknown feature class: {self.feature_class}")

    def forward(self, imgs, mask_ratio=0.15):
        # latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        # pred = self.forward_decoder(latent, ids_restore)
        # loss = self.forward_loss(imgs, pred, mask)  # reconstruct pixel loss

        if self.use_new_feature_decoder:
            latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
            pred = self.forward_decoder(latent, ids_restore)
            target = self.patchify(imgs)

            if self.target_encoder is None and self.norm_pix_loss:
                # Only normalize pixel values if using pixel reconstruction (no target encoder)
                mean = target.mean(dim=-1, keepdim=True)
                var = target.var(dim=-1, keepdim=True)
                target = (target - mean) / (var + 1.0e-6) ** 0.5

            # pixel loss
            loss = (pred - target) ** 2
            loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

            if self.target_encoder is not None or self.enable_ema:
                # feature_target = self.extract_midlayer_features(imgs)
                feature_target = self.extract_features(imgs)
                feature_pred = self.forward_features(latent, ids_restore)
                feature_loss = (feature_pred - feature_target) ** 2
                feature_loss = feature_loss.mean(dim=-1)  # [N, L], mean loss per patch
                loss = loss + feature_loss

            return loss.mean(), pred, mask
        else:
            latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
            pred = self.forward_decoder(latent, ids_restore)
            if self.feature_class == "hog":
                pred_hog = self.unpatchify(pred)
                pred_hog = compute_hog_pytorch(pred_hog)
                loss = self.forward_loss(imgs, pred_hog, mask)
            else:
                loss = self.forward_loss(imgs, pred, mask)
            return loss, pred, mask

    def forward_loss(self, imgs, pred, mask, feature_target=None, feature_pred=None):
        # target = self.extract_midlayer_features(imgs)
        target = self.extract_features(imgs)

        if self.target_encoder is None and self.norm_pix_loss:
            # Only normalize pixel values if using pixel reconstruction (no target encoder)
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5

        # MSE loss between predicted features and target features
        # print(f"pred shape: {pred.shape}, target shape: {target.shape}")
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        return (loss * mask).sum() / mask.sum()

    def forward_features(self, x, ids_restore):
        # embed tokens
        x = self.feature_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.feature_mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        # assert x.shape == (-1, 1, 256, 256), f"Shape is {ids_restore.shape}"
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        # assert x.shape == (-1, 1, 256, 256), f"Shape is {x.shape, x_.shape}"
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.feature_pos_embed

        # apply Transformer blocks
        for blk in self.feature_blocks:
            x = blk(x)
        x = self.feature_norm(x)

        # predictor projection
        x = self.feature_pred(x)

        return x[:, 1:]

    def state_dict(self, *args, **kwargs):
        state_dict = super().state_dict(*args, **kwargs)
        # remove target encoder from state dict, to avoid size inflation when saving
        if self.target_encoder is not None:
            for key in list(state_dict.keys()):
                if key.startswith("target_encoder."):
                    state_dict.pop(key)

        return state_dict

    def update_ema_model(self, ema_model):
        print("EMA model updated")
        self.target_encoder = ema_model


def bmae_deit_tiny_patch4_dec512d8b(**kwargs):
    model = BootMAEDeiT(
        patch_size=4,
        embed_dim=192,
        depth=12,
        num_heads=3,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def bmae_ema_deit_tiny_patch4_dec512d8b(**kwargs):
    model = BootMAEDeiT(
        patch_size=4,
        embed_dim=192,
        depth=12,
        num_heads=3,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        enable_ema=True,
        **kwargs,
    )
    return model


bmae_deit_tiny_patch4 = bmae_deit_tiny_patch4_dec512d8b
# bmae_ema_deit_tiny_patch4 = bmae_ema_deit_tiny_patch4_dec512d8b
