from functools import partial
import torch
from torch import nn

from models_mae import MaskedAutoencoderDeiT
from util import misc
from timm.models.vision_transformer import PatchEmbed, Block, Attention, Mlp, DropPath


class BootstrappedMaskedAutoencoderDeiT(MaskedAutoencoderDeiT):
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
        layer_scale_init_value=1e-5,
        drop_path_rate=0.0,
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

        # If we're using a target encoder, we need to modify the decoder prediction head
        # to output features with the same dimension as the target encoder
        if target_encoder is not None:
            self.target_encoder.eval()
            # Get feature dimension from the target encoder if not provided
            if feature_dim is None:
                # Default to the same dimension as the encoder output
                feature_dim = embed_dim
                print(f"Using target encoder feature dimension: {feature_dim}")
            feature_embed_dim = decoder_embed_dim
            feature_depth = decoder_depth

            # self.feature_pred = nn.Linear(feature_dim, feature_dim, bias=True)
            # --------------------------------------------------------------------------
            # Feature predict specifics
            num_patches = self.patch_embed.num_patches
            self.feature_embed = nn.Linear(embed_dim, feature_embed_dim, bias=True)

            self.feature_mask_token = nn.Parameter(torch.zeros(1, 1, feature_embed_dim))

            self.feature_pos_embed = nn.Parameter(
                torch.zeros(1, num_patches + 1, feature_embed_dim), requires_grad=False
            )  # fixed sin-cos embedding

            feature_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, feature_depth)]
            self.feature_blocks = nn.ModuleList(
                [
                    Block(
                        decoder_embed_dim,
                        decoder_num_heads,
                        mlp_ratio,
                        qkv_bias=True,
                        norm_layer=norm_layer,
                        drop_path=feature_dpr[i],
                    )
                    for i in range(feature_depth)
                ]
            )

            self.feature_norm = norm_layer(decoder_embed_dim)
            self.feature_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans * 4, bias=True)
            # --------------------------------------------------------------------------

            # Freeze the target encoder
            for param in self.target_encoder.parameters():
                param.requires_grad = False

    def extract_target_features(self, imgs, mask, ids_restore=None):
        """Extract features from the target encoder to use as reconstruction targets"""
        if self.target_encoder is None:
            # If no target encoder is provided, use pixel values as in original MAE
            return self.patchify(imgs)

        target_encoder = self.target_encoder
        with torch.no_grad():
            target_encoder.eval()
            x = target_encoder.patch_embed(imgs)
            cls_token = target_encoder.cls_token + self.pos_embed[:, :1, :]
            x = torch.cat((cls_token.expand(x.shape[0], -1, -1), x), dim=1)

            for i, blk in enumerate(target_encoder.blocks):
                x = blk(x)

            return x[:, 1:]

    def forward(self, imgs, mask_ratio=0.75):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        if self.target_encoder is not None:
            with torch.no_grad():
                feature_target = self.extract_target_features(imgs, mask, ids_restore)
            feature_pred = self.forward_features(latent, ids_restore)
            # print(f"feature_target shape: {feature_target.shape}")
            # print(f"feature_pred shape: {feature_pred.shape}")
            pred = self.forward_decoder(latent, ids_restore)
            loss = self.forward_loss(imgs, pred, mask, feature_target, feature_pred)
        else:
            pred = self.forward_decoder(latent, ids_restore)
            loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask

    def forward_loss(self, imgs, pred, mask, feature_target=None, feature_pred=None):
        reconstruct_loss = super().forward_loss(imgs, pred, mask)

        feature_loss = 0
        if feature_target is not None:
            # Compute feature loss using L2 distance
            feature_loss = torch.mean((feature_pred - feature_target) ** 2)

        return reconstruct_loss + feature_loss

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
                if key.startswith("target_encoder.") or key.startswith("feature"):
                    state_dict.pop(key)

        return state_dict


def bmae_deit_tiny_patch4_dec512d8b(**kwargs):
    model = BootstrappedMaskedAutoencoderDeiT(
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


bmae_deit_tiny_patch4 = bmae_deit_tiny_patch4_dec512d8b
