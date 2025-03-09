from functools import partial
import torch
from torch import nn

from models_mae import MaskedAutoencoderDeiT
from util import misc


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
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.target_encoder = target_encoder
        self.target_layer_index = target_layer_index

        # If we're using a target encoder, we need to modify the decoder prediction head
        # to output features with the same dimension as the target encoder
        if target_encoder is not None:
            # Get feature dimension from the target encoder if not provided
            if feature_dim is None:
                # Default to the same dimension as the encoder output
                feature_dim = target_encoder.blocks[target_layer_index].norm2.normalized_shape[0]

            # Replace the decoder prediction head to match target feature dimensions
            self.decoder_pred = nn.Linear(self.decoder_blocks[-1].norm2.normalized_shape[0], feature_dim, bias=True)

            # Freeze the target encoder
            for param in self.target_encoder.parameters():
                param.requires_grad = False

    def extract_target_features(self, imgs, mask_ratio=0):
        """Extract features from the target encoder to use as reconstruction targets"""
        if self.target_encoder is None:
            # If no target encoder is provided, use pixel values as in original MAE
            return self.patchify(imgs)

        with torch.no_grad():
            # Forward pass through target encoder without masking (mask_ratio=0)
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

    def forward_loss(self, imgs, pred, mask):
        # Get target features from the target encoder
        target = self.extract_target_features(imgs)

        if self.target_encoder is None and self.norm_pix_loss:
            # Only normalize pixel values if using pixel reconstruction (no target encoder)
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5

        # MSE loss between predicted features and target features
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        return (loss * mask).sum() / mask.sum()


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


def train_bootstrapped_mae(
    data_loader,
    num_bootstraps=3,
    mask_ratio=0.75,
    img_size=32,
    patch_size=4,
    in_chans=3,
    embed_dim=768,
    depth=12,
    num_heads=12,
    decoder_embed_dim=512,
    decoder_depth=8,
    decoder_num_heads=16,
    target_layer_index=-1,
    epochs_per_bootstrap=100,
    learning_rate=1.5e-4,
    weight_decay=0.05,
    device="cuda",
    save_path="./bootstrapped_mae_checkpoints",
):
    """Train a series of bootstrapped MAEs where each one uses the previous MAE's encoder as target"""
    import os
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR

    os.makedirs(save_path, exist_ok=True)

    models = []

    # First MAE is a regular MAE (MAE-1)
    print(f"Bootstrap 1/{num_bootstraps}: Training regular MAE")
    model = MaskedAutoencoderDeiT(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        decoder_embed_dim=decoder_embed_dim,
        decoder_depth=decoder_depth,
        decoder_num_heads=decoder_num_heads,
    ).to(device)
    model_without_ddp = model

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs_per_bootstrap)

    # Train MAE-1
    for epoch in range(epochs_per_bootstrap):
        model.train()
        total_loss = 0
        num_batches = 0

        for iter, batch in enumerate(data_loader):
            imgs = batch[0].to(device)
            optimizer.zero_grad()

            loss, _, _ = model(imgs, mask_ratio=mask_ratio)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            if iter % 10 == 0:
                print(f"Batch {iter}, Loss: {loss.item():.4f}")

        scheduler.step()
        avg_loss = total_loss / num_batches
        print(f"Bootstrap 1, Epoch {epoch+1}/{epochs_per_bootstrap}, Loss: {avg_loss:.4f}")
        misc.save_model(
            args=args,
            model=model,
            model_without_ddp=model_without_ddp,
            optimizer=optimizer,
            loss_scaler=loss_scaler,
            epoch=epoch,
        )

    # Save MAE-1
    torch.save(model.state_dict(), os.path.join(save_path, "mae_1.pth"))
    models.append(model)

    # Train subsequent bootstrapped MAEs (MAE-2 to MAE-k)
    for k in range(2, num_bootstraps + 1):
        print(f"Bootstrap {k}/{num_bootstraps}: Training bootstrapped MAE with target encoder from bootstrap {k-1}")

        # Use previous MAE's encoder as the target
        target_encoder = models[-1]

        # Create new bootstrapped MAE
        bootstrap_model = BootstrappedMaskedAutoencoderDeiT(
            target_encoder=target_encoder,
            target_layer_index=target_layer_index,
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            decoder_embed_dim=decoder_embed_dim,
            decoder_depth=decoder_depth,
            decoder_num_heads=decoder_num_heads,
        ).to(device)

        optimizer = AdamW(bootstrap_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs_per_bootstrap)

        # Train bootstrapped MAE
        for epoch in range(epochs_per_bootstrap):
            bootstrap_model.train()
            total_loss = 0
            num_batches = 0

            for batch in data_loader:
                imgs = batch[0].to(device)
                optimizer.zero_grad()

                loss, _, _ = bootstrap_model(imgs, mask_ratio=mask_ratio)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            scheduler.step()
            avg_loss = total_loss / num_batches
            print(f"Bootstrap {k}, Epoch {epoch+1}/{epochs_per_bootstrap}, Loss: {avg_loss:.4f}")

        # Save bootstrapped MAE
        torch.save(bootstrap_model.state_dict(), os.path.join(save_path, f"mae_{k}.pth"))
        models.append(bootstrap_model)

    print(f"Finished training {num_bootstraps} bootstrapped MAEs.")
    return models[-1]  # Return the final bootstrapped MAE (MAE-k)


# Example usage:
def main():
    import torch
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms

    # Set up data loading
    transform = transforms.Compose(
        [transforms.Resize(32), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    # Example with CIFAR-10 dataset
    dataset = datasets.CIFAR10(root="./datasets01/cifar10", train=True, download=True, transform=transform)
    data_loader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4)

    # Train bootstrapped MAE
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    final_model = train_bootstrapped_mae(
        data_loader=data_loader,
        num_bootstraps=3,  # Train MAE-1, MAE-2, and MAE-3
        mask_ratio=0.75,
        img_size=32,
        patch_size=4,
        epochs_per_bootstrap=10,  # Reduce for demonstration
        device=device,
    )

    # Save the final model
    torch.save(final_model.state_dict(), "bootstrapped_mae_final.pth")
    print("Final bootstrapped MAE model saved.")


if __name__ == "__main__":
    main()
