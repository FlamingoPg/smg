//! DeepSeek V4 Flash Vision registry spec: model detection, placeholder
//! handling, and the extra-vocab sentinel expansion consumed by the engine.
//!
//! The checkpoint keeps `architectures: ["DeepseekV4ForCausalLM"]` and is a
//! vision model iff `vision_n_layers > 0` in its config. Image placeholders
//! (`<｜deepseek_image｜>`) expand into the official N-layout sentinel block:
//! every token is `vocab_size + sentinel_type` (the extra-vocab id range the
//! engine's MoE `bias_vl` and prefill SWA recognize).
//!
//! Each block starts with the maximal `COMPRESS_PAD_TO - 1` leading pads so
//! the replacement layer can trim `block_start % 4` of them at splice time
//! (see `replacement_alignment`); the engine trims `types` by the same rule,
//! keeping the IMAGE data region aligned to a multiple of 4 tokens exactly
//! like the official `build_image_block(start_pos=...)`.

use std::collections::HashMap;

use serde_json::{json, Value};

use crate::{
    encoder_inputs::{ModelSpecificValue, PreprocessedEncoderInputs},
    registry::{
        MediaPartOrder, ModelMetadata, ModelProcessorSpec, ModelRegistryError, RegistryResult,
    },
    types::{FieldLayout, Modality, PromptReplacement, TokenId},
    vision::processors::deepseek_v4::COMPRESS_PAD_TO,
};

pub const IMAGE_PLACEHOLDER: &str = "<｜deepseek_image｜>";

/// Hard cap on images per request; the official processor has no explicit
/// limit, so this only guards runaway requests.
const MAX_IMAGES: usize = 16;

pub(super) struct DeepseekV4Spec;

impl DeepseekV4Spec {
    /// `vocab_size` from the model config — the base of the extra-vocab
    /// sentinel id range. Must match the serving engine's tokenizer.
    fn vocab_size(metadata: &ModelMetadata) -> RegistryResult<TokenId> {
        metadata
            .config_u32(&["vocab_size"])
            .map(|v| v as TokenId)
            .ok_or_else(|| ModelRegistryError::MissingConfigField {
                field: "vocab_size".to_string(),
            })
    }

    /// Per-item sentinel `types` slices from the batched preprocessed output.
    fn item_types(preprocessed: &PreprocessedEncoderInputs) -> RegistryResult<Vec<Vec<i64>>> {
        let lengths = match preprocessed.model_specific.get("types_lengths") {
            Some(ModelSpecificValue::IntVec(v)) => v.clone(),
            _ => {
                return Err(ModelRegistryError::UnsupportedModel(
                    "deepseek_v4 preprocessed output missing types_lengths".to_string(),
                ))
            }
        };
        let flat = match preprocessed.model_specific.get("types") {
            Some(ModelSpecificValue::IntTensor { data, .. }) => data.clone(),
            _ => {
                return Err(ModelRegistryError::UnsupportedModel(
                    "deepseek_v4 preprocessed output missing types".to_string(),
                ))
            }
        };
        let mut out = Vec::with_capacity(lengths.len());
        let mut offset = 0usize;
        for len in lengths {
            let len = usize::try_from(len).map_err(|_| {
                ModelRegistryError::UnsupportedModel(
                    "deepseek_v4 preprocessed output missing types_lengths".to_string(),
                )
            })?;
            if offset + len > flat.len() {
                return Err(ModelRegistryError::UnsupportedModel(
                    "deepseek_v4 preprocessed output missing types".to_string(),
                ));
            }
            out.push(flat[offset..offset + len].to_vec());
            offset += len;
        }
        Ok(out)
    }
}

impl ModelProcessorSpec for DeepseekV4Spec {
    fn name(&self) -> &'static str {
        "deepseek_v4"
    }

    /// Vision variant of DeepSeek V4 only: same `model_type` as the text
    /// model, distinguished by a non-zero vision tower.
    fn matches(&self, metadata: &ModelMetadata) -> bool {
        metadata.config_model_type() == Some("deepseek_v4")
            && metadata
                .config_u32(&["vision_n_layers"])
                .is_some_and(|n| n > 0)
    }

    /// DeepSeek's template renders parts in authored order (see
    /// `encoding_dsv4.py`: text and `<｜deepseek_image｜>` interleave).
    fn media_part_order(&self) -> MediaPartOrder {
        MediaPartOrder::Authored
    }

    fn placeholder_token(&self, _metadata: &ModelMetadata) -> RegistryResult<String> {
        Ok(IMAGE_PLACEHOLDER.to_string())
    }

    /// The placeholder is a regular vocab token; resolve it through the
    /// tokenizer (the model config declares no `image_token_id` field).
    fn placeholder_token_id(&self, metadata: &ModelMetadata) -> RegistryResult<TokenId> {
        metadata.token_id(IMAGE_PLACEHOLDER)
    }

    fn modality_limits(
        &self,
        _metadata: &ModelMetadata,
    ) -> RegistryResult<HashMap<Modality, usize>> {
        Ok(HashMap::from([(Modality::Image, MAX_IMAGES)]))
    }

    /// Lift the vision geometry from the model config into the preprocessor
    /// config so `DeepseekV4Processor` sees one uniform source.
    fn processor_kwargs(&self, metadata: &ModelMetadata) -> RegistryResult<Value> {
        let keys = [
            "vision_patch_size",
            "vision_downsample_ratio",
            "vision_max_n_token",
            "vision_min_pixels",
            "vision_max_wh_ratio",
        ];
        let mut extra = serde_json::Map::new();
        for key in keys {
            if let Some(v) = metadata.config.get(key) {
                extra.insert(key.to_string(), v.clone());
            }
        }
        Ok(json!({ "extra": extra }))
    }

    /// One replacement per image: the full sentinel block as extra-vocab ids.
    /// The head carries `COMPRESS_PAD_TO - 1` pads for position trimming.
    fn prompt_replacements(
        &self,
        metadata: &ModelMetadata,
        preprocessed: &PreprocessedEncoderInputs,
    ) -> RegistryResult<Vec<PromptReplacement>> {
        let vocab = Self::vocab_size(metadata)?;
        let item_types = Self::item_types(preprocessed)?;
        Ok(item_types
            .into_iter()
            .map(|types| {
                let tokens: Vec<TokenId> = types.iter().map(|&t| vocab + t as TokenId).collect();
                PromptReplacement::sequence(Modality::Image, IMAGE_PLACEHOLDER, tokens)
            })
            .collect())
    }

    /// Sentinel blocks must start at a token index divisible by 4 (the
    /// aligner's compression alignment). The replacement layer trims
    /// `block_start % 4` leading pads from each expansion.
    fn replacement_alignment(&self) -> Option<u32> {
        Some(COMPRESS_PAD_TO as u32)
    }

    /// Patches are already per-item slices of the primary encoder input;
    /// side tensors are batched (per-item value lists).
    fn field_layouts(&self) -> HashMap<String, FieldLayout> {
        HashMap::from([
            ("pixel_values".to_string(), FieldLayout::Batched),
            ("types".to_string(), FieldLayout::Batched),
            ("perm".to_string(), FieldLayout::Batched),
            ("n_vit_h".to_string(), FieldLayout::Batched),
            ("n_vit_w".to_string(), FieldLayout::Batched),
            ("types_lengths".to_string(), FieldLayout::Batched),
        ])
    }

    /// The sentinel metadata is consumed on the engine host when building the
    /// image block; no need to stage it on GPU.
    fn keep_on_cpu_keys(&self) -> Vec<String> {
        vec![
            "types".to_string(),
            "perm".to_string(),
            "n_vit_h".to_string(),
            "n_vit_w".to_string(),
            "types_lengths".to_string(),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vision::processors::deepseek_v4::{
        IMAGE, IMAGE_END, IMAGE_NEW_LINE, IMAGE_PAD, IMAGE_START,
    };
    #[test]
    fn sentinel_constants_match_official() {
        assert_eq!(IMAGE_START, 0);
        assert_eq!(IMAGE_PAD, 1);
        assert_eq!(IMAGE, 2);
        assert_eq!(IMAGE_NEW_LINE, 3);
        assert_eq!(IMAGE_END, 4);
    }

    #[test]
    fn placeholder_uses_fullwidth_bars() {
        assert_eq!(IMAGE_PLACEHOLDER, "<｜deepseek_image｜>");
    }
}
