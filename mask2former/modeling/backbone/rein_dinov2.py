from detectron2.modeling import BACKBONE_REGISTRY, Backbone, ShapeSpec
from .reins import LoRAReins
from .dinov2 import D2DinoVisionTransformer


@BACKBONE_REGISTRY.register()
class D2ReinsDinoVisionTransformer(D2DinoVisionTransformer):
    def __init__(self, cfg, input_shape, **kwargs):
        super().__init__(cfg, input_shape, **kwargs)
        for param in self.parameters():
            param.requires_grad = False

        self.reins = LoRAReins(
            token_length=100,
            embed_dims=1024,
            num_layers=24,
            patch_size=16,
            link_token_to_query=True,
            lora_dim=16,
        )

    def forward_features(self, x, masks=None):
        B, _, h, w = x.shape
        H, W = h // self.patch_size, w // self.patch_size
        x = self.prepare_tokens_with_masks(x, masks)
        outs = []
        for idx, blk in enumerate(self.blocks):
            x = blk(x)
            x = self.reins.forward(
                x,
                idx,
                batch_first=True,
                has_cls_token=True,
            )
            if idx in self.out_indices:
                outs.append(
                    x[:, 1:, :].permute(0, 2, 1).reshape(B, -1, H, W).contiguous()
                )
        return self.reins.return_auto(outs)
