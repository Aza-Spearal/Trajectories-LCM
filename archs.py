# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
#

from lcm.models.base_lcm.builder import (
    BaseLCModelConfig,
    LCMFrontendConfig,
    ProjectionConfig,
    TransformerConfig,
    lcm_arch,
)


# Every model must register a toy_{model_family}
@lcm_arch("toy_base_lcm")
def toy_base_lcm() -> BaseLCModelConfig:
    return BaseLCModelConfig(
        lcm=TransformerConfig(num_layers=2),
    )


@lcm_arch("base_lcm_1_6B")
def base_lcm_1_6B() -> BaseLCModelConfig:
    """Base 1.6B model
    Parameter Size: 1,647,635,456
    """
    model_dim: int = 1024 #2048
    num_attn_heads: int = 16
    return BaseLCModelConfig(
        max_seq_len=20, #4096,
        model_dim=model_dim,
        sonar_embed_dim=1024,
        sonar_normalizer_name="dummy_sonar_normalizer",
        frontend=LCMFrontendConfig(),
        lcm=TransformerConfig(
            final_dropout_p=0.0,
            attention_dropout_p=0.0,
            dropout_p=0.1,
            mha_output_proj_bias=True,
            ffn_inner_dim=model_dim * 4,
            num_attn_heads=num_attn_heads,
            num_layers= 16, #32,
            pos_embedding_style="rope",
            use_swiglu=True,
            layer_normalization_style="rms",
        ),
        postnet=ProjectionConfig(),
    )

@lcm_arch("base_lcm_max")
def base_lcm_max(config) -> BaseLCModelConfig:
    model_dim: int = config.model_dim
    num_attn_heads: int = 16
    return BaseLCModelConfig(
        max_seq_len=20,
        sonar_embed_dim=1024,
        model_dim=model_dim,
        sonar_normalizer_name="dummy_sonar_normalizer",
        frontend=LCMFrontendConfig(),
        lcm=TransformerConfig(
            final_dropout_p=0.0,
            attention_dropout_p=0.0,
            dropout_p=0.1,
            mha_output_proj_bias=True,
            #ffn_inner_dim=model_dim * config.lcm_ffn_inner_dim,
            ffn_inner_dim=model_dim * config.model_arch[0],
            num_attn_heads=num_attn_heads,
            #num_layers=config.lcm_num_layers,
            num_layers=config.model_arch[1],
            pos_embedding_style="rope",
            use_swiglu=True,
            layer_normalization_style = "rms",
            ),
        postnet=ProjectionConfig(),
    )
    
@lcm_arch("base_lcm_max_ray")
def base_lcm_max_ray(config) -> BaseLCModelConfig:
    model_dim: int = config['model_dim']
    num_attn_heads: int = 16
    return BaseLCModelConfig(
        max_seq_len=20,
        sonar_embed_dim=1024,
        model_dim=model_dim,
        sonar_normalizer_name="dummy_sonar_normalizer",
        frontend=LCMFrontendConfig(),
        lcm=TransformerConfig(
            final_dropout_p=0.0,
            attention_dropout_p=0.0,
            dropout_p=0.1,
            mha_output_proj_bias=True,
            #ffn_inner_dim=model_dim * config.lcm_ffn_inner_dim,
            ffn_inner_dim=model_dim * config['model_arch'][1],
            num_attn_heads=num_attn_heads,
            #num_layers=config.lcm_num_layers,
            num_layers=config['model_arch'][0],
            pos_embedding_style="rope",
            use_swiglu=True,
            layer_normalization_style = "rms",
            ),
        postnet=ProjectionConfig(),
    )
    
@lcm_arch("base_lcm_tuner")
def base_lcm_tuner(config) -> BaseLCModelConfig:
    model_dim: int = config.model_dim
    num_attn_heads: int = config.num_attn_heads
    return BaseLCModelConfig(
        max_seq_len=20,
        sonar_embed_dim=1024,
        model_dim=model_dim,
        #sonar_normalizer_name=config.sonar_normalizer_name,
        sonar_normalizer_name=None if config.sonar_normalizer_name != "dummy_sonar_normalizer" else "dummy_sonar_normalizer",
        frontend=LCMFrontendConfig(
            dropout_p=config.frontend_dropout_p,
            pre_linear_init_fn = config.frontend_pre_linear_init_fn,
            scale_embeddings = config.frontend_scale_embeddings,
            weight_normalization = config.frontend_weight_normalization,
            ),
        lcm=TransformerConfig(
            final_dropout_p=config.lcm_final_dropout_p,
            attention_dropout_p=config.lcm_attention_dropout_p,
            dropout_p=config.lcm_dropout_p,
            ffn_inner_dim=config.lcm_ffn_inner_dim * model_dim,
            num_attn_heads=num_attn_heads,
            num_layers=config.lcm_num_layers,
            pos_embedding_style=config.lcm_pos_embedding_style,
            use_swiglu=config.lcm_use_swiglu,
            ffn_inner_activation_name = 'relu' if config.lcm_use_swiglu else config.lcm_ffn_inner_activation_name,
            layer_normalization_style = config.lcm_layer_normalization_style,
            norm_order_style = config.lcm_norm_order_style,
            final_norm_order_style = config.lcm_final_norm_order_style,
            enable_qk_layernorm = config.lcm_enable_qk_layernorm,
            mha_qkv_weight_normalization = config.lcm_mha_qkv_weight_normalization,
            mha_output_weight_normalization = config.lcm_mha_output_weight_normalization,
            mha_output_proj_bias = config.lcm_mha_output_proj_bias,
            attention_output_init_fn = config.lcm_attention_output_init_fn,
            ),
        postnet=ProjectionConfig(
            dropout_p=config.postnet_dropout_p,
            linear_init_fn = config.postnet_linear_init_fn,
            weight_normalization = config.postnet_weight_normalization,
            layer_normalization_style = config.postnet_layer_normalization_style,
            activation_name = config.postnet_activation_name,
        ),
    )