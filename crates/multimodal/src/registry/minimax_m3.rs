use std::collections::HashMap;

use serde_json::{json, Value};

use crate::{
    encoder_inputs::PreprocessedEncoderInputs,
    registry::{ModelMetadata, ModelProcessorSpec, ModelRegistryError, RegistryResult},
    types::{FieldLayout, Modality, PromptReplacement, TokenId},
};

const IMAGE_TOKEN: &str = "]<]image[>[";
const VISION_START_TOKEN: &str = "]<]start of image[>[";
const VISION_END_TOKEN: &str = "]<]end of image[>[";

pub(super) struct MiniMaxM3VisionSpec;

impl MiniMaxM3VisionSpec {
    fn image_token_id(metadata: &ModelMetadata) -> RegistryResult<TokenId> {
        metadata
            .config_u32(&["image_token_index"])
            .map(|value| value as TokenId)
            .ok_or_else(|| ModelRegistryError::MissingConfigField {
                field: "image_token_index".to_string(),
            })
    }
}

impl ModelProcessorSpec for MiniMaxM3VisionSpec {
    fn name(&self) -> &'static str {
        "minimax_m3_vl"
    }

    fn matches(&self, metadata: &ModelMetadata) -> bool {
        let id = metadata.model_id.to_ascii_lowercase();
        (id.contains("minimax") && id.contains("m3"))
            || metadata
                .config_model_type()
                .is_some_and(|model_type| model_type == "minimax_m3_vl")
    }

    fn placeholder_token(&self, _metadata: &ModelMetadata) -> RegistryResult<String> {
        Ok(IMAGE_TOKEN.to_string())
    }

    fn placeholder_token_id(&self, metadata: &ModelMetadata) -> RegistryResult<TokenId> {
        Self::image_token_id(metadata)
    }

    fn modality_limits(
        &self,
        _metadata: &ModelMetadata,
    ) -> RegistryResult<HashMap<Modality, usize>> {
        Ok(HashMap::from([(Modality::Image, 10)]))
    }

    fn processor_kwargs(&self, _metadata: &ModelMetadata) -> RegistryResult<Value> {
        Ok(json!({}))
    }

    fn prompt_replacements(
        &self,
        metadata: &ModelMetadata,
        preprocessed: &PreprocessedEncoderInputs,
    ) -> RegistryResult<Vec<PromptReplacement>> {
        let image_token_id = Self::image_token_id(metadata)?;
        let start_token_id = metadata.token_id(VISION_START_TOKEN)?;
        let end_token_id = metadata.token_id(VISION_END_TOKEN)?;

        Ok(preprocessed
            .feature_token_counts
            .iter()
            .map(|&token_count| {
                // The processor reports one text token for each 2x2 patch
                // group, i.e. grid_t * grid_h * grid_w / 4.
                let mut tokens = Vec::with_capacity(token_count + 2);
                tokens.push(start_token_id);
                tokens.extend(std::iter::repeat_n(image_token_id, token_count));
                tokens.push(end_token_id);
                PromptReplacement::sequence(Modality::Image, IMAGE_TOKEN, tokens)
            })
            .collect())
    }

    fn field_layouts(&self) -> HashMap<String, FieldLayout> {
        HashMap::from([
            (
                "pixel_values".to_string(),
                FieldLayout::flat("patches_per_image"),
            ),
            ("image_grid_thw".to_string(), FieldLayout::Batched),
            ("patches_per_image".to_string(), FieldLayout::Batched),
        ])
    }

    fn keep_on_cpu_keys(&self) -> Vec<String> {
        vec!["image_grid_thw".to_string()]
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::{IMAGE_TOKEN, VISION_END_TOKEN, VISION_START_TOKEN};
    use crate::{
        registry::{test_helpers::*, ModelMetadata, ModelRegistry},
        types::{ImageSize, TokenId},
    };

    const IMAGE_ID: u32 = 200025;
    const START_ID: u32 = 200029;
    const END_ID: u32 = 200030;

    fn tokenizer() -> TestTokenizer {
        TestTokenizer::new(&[
            (IMAGE_TOKEN, IMAGE_ID),
            (VISION_START_TOKEN, START_ID),
            (VISION_END_TOKEN, END_ID),
        ])
    }

    #[test]
    fn matches_checkpoint_id_and_model_type() {
        let tokenizer = tokenizer();
        let config = json!({
            "model_type": "minimax_m3_vl",
            "image_token_index": IMAGE_ID
        });
        let registry = ModelRegistry::new();

        for model_id in ["MiniMaxAI/MiniMax-M3-MXFP8", "custom-model"] {
            let metadata = ModelMetadata {
                model_id,
                tokenizer: &tokenizer,
                config: &config,
            };
            let spec = registry.lookup(&metadata).expect("minimax m3 spec");
            assert_eq!(spec.name(), "minimax_m3_vl");
            assert_eq!(spec.placeholder_token(&metadata).unwrap(), IMAGE_TOKEN);
            assert_eq!(
                spec.placeholder_token_id(&metadata).unwrap(),
                IMAGE_ID as TokenId
            );
        }
    }

    #[test]
    fn expands_image_with_start_merged_tokens_and_end() {
        let tokenizer = tokenizer();
        let config = json!({
            "model_type": "minimax_m3_vl",
            "image_token_index": IMAGE_ID
        });
        let metadata = ModelMetadata {
            model_id: "MiniMaxAI/MiniMax-M3-MXFP8",
            tokenizer: &tokenizer,
            config: &config,
        };
        let registry = ModelRegistry::new();
        let spec = registry.lookup(&metadata).expect("minimax m3 spec");

        // A 4x4 patch grid is merged 2x2, yielding four repeated image tokens.
        let replacements = spec
            .prompt_replacements(
                &metadata,
                &test_preprocessed_with_tokens(&[ImageSize::new(56, 56)], &[4]),
            )
            .unwrap();

        assert_eq!(replacements.len(), 1);
        assert_eq!(replacements[0].placeholder_token, IMAGE_TOKEN);
        assert_eq!(
            replacements[0].tokens,
            vec![
                START_ID as TokenId,
                IMAGE_ID as TokenId,
                IMAGE_ID as TokenId,
                IMAGE_ID as TokenId,
                IMAGE_ID as TokenId,
                END_ID as TokenId,
            ]
        );
    }

    #[test]
    fn does_not_match_other_minimax_generations() {
        let tokenizer = tokenizer();
        let config = json!({"model_type": "minimax_m2"});
        let metadata = ModelMetadata {
            model_id: "MiniMaxAI/MiniMax-M2",
            tokenizer: &tokenizer,
            config: &config,
        };
        assert!(ModelRegistry::new().lookup(&metadata).is_none());
    }
}
