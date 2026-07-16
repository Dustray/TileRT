"""Core operations for Qwen3.6-35B-A3B."""

from tilert.models.qwen3_6.ops.broadcast_selected_token_ids import (
    broadcast_selected_token_ids,
)
from tilert.models.qwen3_6.ops.delta_net import (
    DeltaNetAlgorithm,
    DeltaNetOp,
    DeltaNetRefWeightsAlias,
    DeltaNetTilertWeightsAlias,
    DeltaNetWeightsConverter,
    delta_net,
)
from tilert.models.qwen3_6.ops.down_allreduce import (
    DownAllReduce,
    DownAllReduceAlgorithm,
    down_allreduce,
)
from tilert.models.qwen3_6.ops.eh_proj_allreduce import (
    EHProjAllReduce,
    EHProjAllReduceAlgorithm,
    eh_proj_allreduce,
)
from tilert.models.qwen3_6.ops.gqa_attention import (
    GQAAttention,
    GQAAttentionAlgorithm,
    GQAAttentionRefWeightsAlias,
    GQAAttentionTilertWeightsAlias,
    GQAAttentionWeightsConverter,
    gqa_attention,
)
from tilert.models.qwen3_6.ops.expert_down_allreduce import (
    ExpertDownAllReduce,
    ExpertDownAllReduceAlgorithm,
    expert_down_allreduce,
)
from tilert.models.qwen3_6.ops.expert_sel_up_gate_silu import (
    ExpertSelectUpGateSiLU,
    ExpertSelectUpGateSiLUAlgorithm,
)
from tilert.models.qwen3_6.ops.padded_allreduce_add import (
    PaddedAllReduceAdd,
    PaddedAllReduceAddAlgorithm,
    padded_allreduce_add,
)
from tilert.models.qwen3_6.ops.qkv_rope import (
    QKVRoPE,
    QKVRoPEAlgorithm,
    QKVRoPERefWeightsAlias,
    QKVRoPETilertWeightsAlias,
    qkv_rope,
)
from tilert.models.qwen3_6.ops.receive_selected_token_ids import (
    receive_selected_token_ids,
)
from tilert.models.qwen3_6.ops.rmsnorm_expert_proj import (
    RMSNormExpertProj,
    RMSNormExpertProjAlgorithm,
)
from tilert.models.qwen3_6.ops.rmsnorm_head_proj import (
    RMSNormHeadProj,
    RMSNormHeadProjAlgorithm,
)
from tilert.models.qwen3_6.ops.rmsnorm_quant import rmsnorm_quant
from tilert.models.qwen3_6.ops.rmsnorm_up_gate_silu import (
    RMSNormUpGateSiLU,
    RMSNormUpGateSiLUAlgorithm,
)
from tilert.models.qwen3_6.ops.rotate import (
    Rotate,
    RotateAlgorithm,
    RotateRefWeightsAlias,
    RotateTilertWeightsAlias,
    rotate,
    rotate_activation,
)
from tilert.models.qwen3_6.ops.topk import TopK, topk_accurate, topk_approximate
from tilert.models.qwen3_6.ops.unproj_o_allreduce import (
    UnProjOAllReduce,
    UnProjOAllReduceAlgorithm,
    unproj_o_allreduce,
)

__all__ = [
    "delta_net",
    "DeltaNetOp",
    "DeltaNetAlgorithm",
    "DeltaNetRefWeightsAlias",
    "DeltaNetTilertWeightsAlias",
    "DeltaNetWeightsConverter",
    "down_allreduce",
    "DownAllReduce",
    "DownAllReduceAlgorithm",
    "gqa_attention",
    "GQAAttention",
    "GQAAttentionAlgorithm",
    "GQAAttentionRefWeightsAlias",
    "GQAAttentionTilertWeightsAlias",
    "GQAAttentionWeightsConverter",
    "expert_down_allreduce",
    "ExpertDownAllReduce",
    "ExpertDownAllReduceAlgorithm",
    "unproj_o_allreduce",
    "rotate",
    "rotate_activation",
    "Rotate",
    "RotateAlgorithm",
    "RotateRefWeightsAlias",
    "RotateTilertWeightsAlias",
    "TopK",
    "topk_approximate",
    "topk_accurate",
    "qkv_rope",
    "QKVRoPE",
    "QKVRoPEAlgorithm",
    "QKVRoPERefWeightsAlias",
    "QKVRoPETilertWeightsAlias",
    "eh_proj_allreduce",
    "EHProjAllReduceAlgorithm",
    "rmsnorm_quant",
    "RMSNormExpertProj",
    "RMSNormExpertProjAlgorithm",
    "RMSNormUpGateSiLU",
    "RMSNormUpGateSiLUAlgorithm",
    "UnProjOAllReduce",
    "UnProjOAllReduceAlgorithm",
    "RMSNormHeadProj",
    "RMSNormHeadProjAlgorithm",
    "ExpertSelectUpGateSiLU",
    "ExpertSelectUpGateSiLUAlgorithm",
    "PaddedAllReduceAdd",
    "PaddedAllReduceAddAlgorithm",
    "padded_allreduce_add",
    "broadcast_selected_token_ids",
    "receive_selected_token_ids",
]
