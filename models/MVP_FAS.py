import torch
from torch import nn
import torch.nn.functional as F
from models.CLIP import clip
from models.modules.slot_attention_PQTK import SlotAttention_PQTK
from models.modules.head import Classifier, Slot_Projection, Projection, SupConProjector


spoof_templates = [
    'spoof face',
    'attack face',
    'fake face',
    'replay attack face',
    # 'video replay face',
    'screen replay face',
    'printed photo face spoof',
    # 'print attack face',
    # '2D attack face',
    'silicone face spoof',
    'latex face spoof',
    '3D mask face spoof',
    # 'full mask face spoof',
    # 'plastic mask face',
    # 'paper mask face spoof',
    # 'transparent mask face',
    # 'half mask face',
    # 'mannequin face spoof',
    # 'dummy face',
    # 'wax figure face',
    # 'doll face',
    # 'makeup attack face spoof',
    # 'heavy makeup face',
    # 'makeup obfuscation face',
    # 'makeup impersonation face',
    # 'partial occlusion face',
    # 'covered mouth face spoof',
    # 'covered eye face spoof',
    # 'fake glasses face spoof',
]

real_templates = [
    'real face',
    'bonafide face',
    'genuine face',
    'true face',
    'live face',
    # 'left half of real face',
    # 'right half of real face',
    # 'left half of live face',
    # 'right half of live face',
    # 'live human face',
    # 'natural face',
    # 'authentic face',
    # 'actual person face',
    # 'valid face',
    # 'legitimate face',
    # 'normal face',
    # 'clear human face',
    # 'non-attack face',
    # 'bona fide human face',
]

class mspt(nn.Module):

    def __init__(self, cfg=None, args=None, device='cpu', backbone="ViT-B/16"):
        super(mspt, self).__init__()
        self.cfg = cfg
        self.args = args
        self.head_type = 'cls'
        self.device = device
        self.model, _ = clip.load(backbone, device=self.device, strict=False)
        self._freeze_stages(self.model, exclude_key=['visual','learnable'])

        self.cls_projection = Projection()
        self.classifier = Classifier(512)

        # MVSlot
        self.MVSlot = SlotAttention_PQTK(
                dim=512,
                iters=3
            )

        self.slot_projection = Slot_Projection(head_type=self.head_type)
        self.patch_projection = Projection()
        self.MTPA_classifier = Classifier(512)
        self.supcon_projector = SupConProjector(in_dim=512, hidden_dim=256, out_dim=128)

        # Cache text embeddings — templates are static, no need to re-encode every forward pass
        with torch.no_grad():
            spoof_tokens = clip.tokenize(spoof_templates).to(self.device)
            real_tokens = clip.tokenize(real_templates).to(self.device)
            spoof_embs = self.model.encode_text(spoof_tokens).float()
            real_embs = self.model.encode_text(real_tokens).float()
        self.register_buffer('cached_spoof_text_emb', spoof_embs.mean(dim=0))
        self.register_buffer('cached_real_text_emb', real_embs.mean(dim=0))
        self.register_buffer('cached_text_features', torch.cat([spoof_embs, real_embs], dim=0))


    def _freeze_stages(self, model, exclude_key=None):
        """Freeze stages param and norm stats."""
        for n, m in model.named_parameters():
            if exclude_key:
                if isinstance(exclude_key, str):
                    if not exclude_key in n:
                        m.requires_grad = False
                elif isinstance(exclude_key, list):
                    count = 0
                    for i in range(len(exclude_key)):
                        i_layer = str(exclude_key[i])
                        if i_layer in n:
                            count += 1
                    if count == 0:
                        m.requires_grad = False
                    elif count>0:
                        print('Finetune layer in backbone:', n)
                else:
                    raise AttributeError("Dont support the type of exclude_key!")
            else:
                m.requires_grad = False

    def align_patches(self, spoof, real, patch, target):
        real_spoof_txts = [spoof.unsqueeze(0) if rf == 0 else real.unsqueeze(0) for rf in target['Is_real']]
        real_spoof_txts = torch.concat(real_spoof_txts, dim=0)
        patch_activations = self.patch_alignment(patch, real_spoof_txts)
        broad_patch = patch * patch_activations.unsqueeze(-1)
        patch_alignment_results = torch.sum(broad_patch, dim=1)
        return self.MTPA_classifier(patch_alignment_results)

    # def patch_alignment(self, visual_patch_proj, text_cls_proj, logit_scale):  # shapes =  [B, 196, 768], [B, 768]
    def patch_alignment(self, visual_patch_proj, text_cls_proj):  # shapes =  [B, 196, 768], [B, 768]
        # normalize visual patch tokens and then permute
        normalized_visual_patch_proj = F.normalize(visual_patch_proj, dim=-1)
        normalized_visual_patch_proj = normalized_visual_patch_proj.transpose(-2, -1)  # shapes =  [B, 768, 196]

        # normalize text cls token and unsqueeze (required for matmul)
        normalized_text_cls_proj = F.normalize(text_cls_proj, dim=-1)
        normalized_text_cls_proj = normalized_text_cls_proj.unsqueeze(1)  # shapes =  [B, 1, 768]

        # compute dot product
        # patch_activations = logit_scale * normalized_text_cls_proj @ normalized_visual_patch_proj  # shapes =  [B, 1, 196]
        patch_activations = normalized_text_cls_proj @ normalized_visual_patch_proj  # shapes =  [B, 1, 196]
        patch_activations = patch_activations.squeeze(1)  # shapes =  [B, 196]
        # because of dot product, the range is between -1 (least similar) to +1 (most similar)
        # multiply by 10 and apply sigmoid function. this squashes the range from 0 to 1 for every element (not necessarily sums to 1 like that of a softmax function)
        return torch.sigmoid(patch_activations*10)

    def forward(self, input, target=None):

        results = {'similarity': None, 'patch_alignment': None, 'embedding': None}

        spoof_class_embeddings = self.cached_spoof_text_emb
        real_class_embeddings = self.cached_real_text_emb
        text_features = self.cached_text_features

        cls_embedding, _, patch = self.model.encode_image(input)

        # cls projection
        cls_embedding = self.cls_projection(cls_embedding)

        # patch projection
        patch = self.patch_projection(patch)

        # MVS
        real_spoof_slot = self.MVSlot(patch + cls_embedding.unsqueeze(1), text_features)

        real_spoof_slot = real_spoof_slot.mean(dim=1)
        real_spoof_slot = self.slot_projection(real_spoof_slot)

        supcon_emb = self.supcon_projector(real_spoof_slot)
        results['embedding'] = F.normalize(supcon_emb, dim=-1)

        # CLIP patch align
        if target is not None:
            patch_alignment_results = self.align_patches(spoof=spoof_class_embeddings, real=real_class_embeddings, patch=patch, target=target)
            results['patch_alignment'] = patch_alignment_results

        results['similarity'] = self.classifier(real_spoof_slot)

        return results
