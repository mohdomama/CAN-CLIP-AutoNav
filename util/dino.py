import torch
from torch import nn
import torch.nn.functional as F 
from typing import Literal
import cv2
import einops
import torchvision.transforms as T


# Extract features from a Dino-v2 model
_DINO_V2_MODELS = Literal["dinov2_vits14", "dinov2_vitb14", \
                        "dinov2_vitl14", "dinov2_vitg14"]
_DINO_FACETS = Literal["query", "key", "value", "token"]
class DinoV2ExtractFeatures:
    """
        Extract features from an intermediate layer in Dino-v2
    """
    def __init__(self, dino_model: _DINO_V2_MODELS, layer: int, 
                facet: _DINO_FACETS="token", use_cls=False, 
                norm_descs=True, device: str = "cpu") -> None:
        """
            Parameters:
            - dino_model:   The DINO-v2 model to use
            - layer:        The layer to extract features from
            - facet:    "query", "key", or "value" for the attention
                        facets. "token" for the output of the layer.
            - use_cls:  If True, the CLS token (first item) is also
                        included in the returned list of descriptors.
                        Otherwise, only patch descriptors are used.
            - norm_descs:   If True, the descriptors are normalized
            - device:   PyTorch device to use
        """
        self.vit_type: str = dino_model
        self.dino_model: nn.Module = torch.hub.load(
                'facebookresearch/dinov2', dino_model)
        self.device = torch.device(device)
        self.dino_model = self.dino_model.eval().to(self.device)
        self.layer: int = layer
        self.facet = facet
        if self.facet == "token":
            self.fh_handle = self.dino_model.blocks[self.layer].\
                    register_forward_hook(
                            self._generate_forward_hook())
        else:
            self.fh_handle = self.dino_model.blocks[self.layer].\
                    attn.qkv.register_forward_hook(
                            self._generate_forward_hook())
        self.use_cls = use_cls
        self.norm_descs = norm_descs
        # Hook data
        self._hook_out = None
    
    def _generate_forward_hook(self):
        def _forward_hook(module, inputs, output):
            self._hook_out = output
        return _forward_hook
    
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """
            Parameters:
            - img:   The input image
        """
        with torch.no_grad():
            res = self.dino_model(img)
            if self.use_cls:
                res = self._hook_out
            else:
                res = self._hook_out[:, 1:, ...]
            if self.facet in ["query", "key", "value"]:
                d_len = res.shape[2] // 3
                if self.facet == "query":
                    res = res[:, :, :d_len]
                elif self.facet == "key":
                    res = res[:, :, d_len:2*d_len]
                else:
                    res = res[:, :, 2*d_len:]
        if self.norm_descs:
            res = F.normalize(res, dim=-1)
        self._hook_out = None   # Reset the hook
        return res
    
    def __del__(self):
        self.fh_handle.remove()



def main():
    dino = DinoV2ExtractFeatures("dinov2_vitb14", 11, "key", device="cuda:0")
    imgfile = 'data/inference/000620.png'
    img = cv2.imread(imgfile, cv2.IMREAD_ANYCOLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, c = img.shape
    preprocess = T.Compose(
        [
            T.ToTensor(), # Transform numpy image to torch tensor
            T.CenterCrop([(h//14)*14, (w//14)*14]), 
            T.Normalize(  # Normalize image (mean sub, std div)
                mean=[0.485, 0.456, 0.406],  # ImageNet mean
                std=[0.229, 0.224, 0.225],  # Imagenet std
            ),
        ],
    )
    h_new, w_new = (h // 14) * 14, (w // 14) * 14  # Resize (so fits with DINO patchsize)

    _img = preprocess(img)
    _img = _img[None, ...].cuda()  # C,H,W --> 1,C,H,W

    dinofeat = dino(_img)
    breakpoint()

    # Extracted DINOv2 features are at a resolution of 1, (h // 14) * (w // 14), feat_dim
    # Reshape them to (h // 14), (w // 14), feat_dim
    dinofeat = einops.rearrange(
        dinofeat[0],
        "(p_h p_w) d -> p_h p_w d",
        p_h=int(h_new/14),
        p_w=int(w_new/14),
    )
    breakpoint()


if __name__=='__main__':
    main()