
#### 2. `src/model/metric_ground.py` (Core Model)

```python
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoProcessor
from .components import MetricInjector, SemanticMetricFusion, DistillationHead
from .adapters import LoRAAdapter

class MetricGround(nn.Module):
    def __init__(self, vlm_name="llava-hf/llava-1.5-7b-hf", dim=1024, adapter_rank=16):
        super().__init__()
        # Frozen 2D VLM backbone
        self.vlm = AutoModelForCausalLM.from_pretrained(vlm_name, torch_dtype=torch.float16)
        self.processor = AutoProcessor.from_pretrained(vlm_name)
        for param in self.vlm.parameters():
            param.requires_grad = False

        # Metric Injector (3D → metric features)
        self.metric_injector = MetricInjector(in_channels=3, out_dim=dim)  # e.g. PointNet++ or Voxel CNN

        # Lightweight adapters on VLM layers
        self.adapters = nn.ModuleList([
            LoRAAdapter(dim, rank=adapter_rank) for _ in range(8)  # apply to some layers
        ])

        # Fusion module
        self.fusion = SemanticMetricFusion(dim=dim, num_heads=8)

        # Distillation head (language → metric teacher)
        self.distill_head = DistillationHead(dim=dim)

        self.dim = dim

    def forward(self, images, point_clouds, texts, return_distill=False):
        # 2D semantic features (frozen)
        inputs = self.processor(images=images, text=texts, return_tensors="pt", padding=True)
        semantic_feats = self.vlm.vision_tower(inputs.pixel_values, output_hidden_states=True).hidden_states[-1]

        # Apply lightweight adapters
        for i, adapter in enumerate(self.adapters):
            semantic_feats = semantic_feats + adapter(semantic_feats)

        # 3D metric features
        metric_feats = self.metric_injector(point_clouds)  # [B, N, dim]

        # Semantic-Metric Fusion
        fused = self.fusion(semantic_feats, metric_feats)

        # Language output
        outputs = self.vlm(inputs_embeds=fused, **inputs)

        if return_distill:
            distill_logits = self.distill_head(fused)
            return outputs, distill_logits

        return outputs

    @torch.no_grad()
    def generate_sqa(self, images, point_clouds, question):
        """Generate SQA answer with metric grounding"""
        prompt = f"[INST] {question} [/INST]"
        inputs = self.processor(images=images, text=prompt, return_tensors="pt")
        # ... similar forward pass ...
        generated = self.vlm.generate(**inputs, max_new_tokens=128)
        return self.processor.decode(generated[0], skip_special_tokens=True)
