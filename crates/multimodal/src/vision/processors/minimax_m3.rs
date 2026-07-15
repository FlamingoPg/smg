//! MiniMax-M3 VL image preprocessing.
//!
//! MiniMax-M3 uses the same merge-grouped patch layout as Qwen2-VL, with a
//! different dynamic-resolution budget.  This wrapper keeps the checkpoint
//! defaults in one place while reusing the shared, tested patchification path.

use std::ops::Deref;

use image::DynamicImage;

use super::qwen_vl_base::{QwenVLConfig, QwenVLProcessorBase, QwenVideoResizeMode};
use crate::vision::{
    preprocessor_config::PreProcessorConfig,
    processor::{PreprocessedEncoderInputs, VisionPreProcessor},
    transforms::TransformError,
};

/// CLIP normalization used by the MiniMax-M3 vision tower.
pub const CLIP_MEAN: [f64; 3] = [0.48145466, 0.4578275, 0.40821073];
pub const CLIP_STD: [f64; 3] = [0.26862954, 0.26130258, 0.27577711];

pub const DEFAULT_PATCH_SIZE: usize = 14;
pub const DEFAULT_TEMPORAL_PATCH_SIZE: usize = 2;
pub const DEFAULT_MERGE_SIZE: usize = 2;
pub const DEFAULT_MIN_PIXELS: usize = 4 * 28 * 28;
pub const DEFAULT_MAX_PIXELS: usize = 672 * 672;

/// Image processor for MiniMax-M3 VL checkpoints.
#[derive(Debug, Clone)]
pub struct MiniMaxM3VLProcessor {
    inner: QwenVLProcessorBase,
}

impl Default for MiniMaxM3VLProcessor {
    fn default() -> Self {
        Self::new()
    }
}

impl MiniMaxM3VLProcessor {
    /// Construct a processor with the official MiniMax-M3 checkpoint defaults.
    pub fn new() -> Self {
        Self::with_config(
            DEFAULT_PATCH_SIZE,
            DEFAULT_MERGE_SIZE,
            DEFAULT_MIN_PIXELS,
            DEFAULT_MAX_PIXELS,
            DEFAULT_TEMPORAL_PATCH_SIZE,
        )
    }

    fn with_config(
        patch_size: usize,
        merge_size: usize,
        min_pixels: usize,
        max_pixels: usize,
        temporal_patch_size: usize,
    ) -> Self {
        Self {
            inner: QwenVLProcessorBase::new(QwenVLConfig {
                patch_size,
                merge_size,
                min_pixels,
                max_pixels,
                video_min_pixels: min_pixels,
                video_max_pixels: max_pixels,
                video_resize_mode: QwenVideoResizeMode::TotalVolume,
                temporal_patch_size,
                mean: CLIP_MEAN,
                std: CLIP_STD,
                model_name: "minimax-m3-vl",
            }),
        }
    }

    /// Apply explicit structural overrides from a HuggingFace preprocessor config.
    pub fn from_preprocessor_config(config: &PreProcessorConfig) -> Self {
        Self::with_config(
            config.get_patch_size(DEFAULT_PATCH_SIZE),
            config.merge_size.unwrap_or(DEFAULT_MERGE_SIZE),
            config.min_pixels.unwrap_or(DEFAULT_MIN_PIXELS),
            config.max_pixels.unwrap_or(DEFAULT_MAX_PIXELS),
            config
                .temporal_patch_size
                .unwrap_or(DEFAULT_TEMPORAL_PATCH_SIZE),
        )
    }

    fn with_preprocessor_config(&self, config: &PreProcessorConfig) -> Self {
        if config.has_structural_overrides() {
            Self::from_preprocessor_config(config)
        } else {
            self.clone()
        }
    }

    pub fn patch_size(&self) -> usize {
        self.inner.patch_size()
    }

    pub fn merge_size(&self) -> usize {
        self.inner.merge_size()
    }

    pub fn min_pixels(&self) -> usize {
        self.inner.min_pixels()
    }

    pub fn max_pixels(&self) -> usize {
        self.inner.max_pixels()
    }

    pub fn temporal_patch_size(&self) -> usize {
        self.inner.temporal_patch_size()
    }

    pub fn smart_resize(
        &self,
        height: usize,
        width: usize,
    ) -> Result<(usize, usize), TransformError> {
        self.inner.smart_resize(height, width)
    }
}

impl Deref for MiniMaxM3VLProcessor {
    type Target = QwenVLProcessorBase;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl VisionPreProcessor for MiniMaxM3VLProcessor {
    fn default_mean(&self) -> [f64; 3] {
        self.inner.default_mean()
    }

    fn default_std(&self) -> [f64; 3] {
        self.inner.default_std()
    }

    fn preprocess(
        &self,
        images: &[DynamicImage],
        config: &PreProcessorConfig,
    ) -> Result<PreprocessedEncoderInputs, TransformError> {
        let processor = self.with_preprocessor_config(config);
        processor.inner.preprocess(images, config)
    }

    fn calculate_num_tokens(&self, width: u32, height: u32, config: &PreProcessorConfig) -> usize {
        let processor = self.with_preprocessor_config(config);
        processor.inner.calculate_num_tokens(width, height, config)
    }

    fn model_name(&self) -> &'static str {
        self.inner.model_name()
    }

    fn get_processed_size(&self, config: &PreProcessorConfig) -> Option<(u32, u32)> {
        self.inner.get_processed_size(config)
    }
}

#[cfg(test)]
mod tests {
    use image::{Rgb, RgbImage};

    use super::*;
    use crate::vision::processor::ModelSpecificValue;

    fn solid_image(width: u32, height: u32, rgb: [u8; 3]) -> DynamicImage {
        DynamicImage::from(RgbImage::from_pixel(width, height, Rgb(rgb)))
    }

    #[test]
    fn checkpoint_defaults_are_exact() {
        let processor = MiniMaxM3VLProcessor::new();
        assert_eq!(processor.patch_size(), 14);
        assert_eq!(processor.temporal_patch_size(), 2);
        assert_eq!(processor.merge_size(), 2);
        assert_eq!(processor.min_pixels(), 3136);
        assert_eq!(processor.max_pixels(), 451_584);
        assert_eq!(processor.default_mean(), CLIP_MEAN);
        assert_eq!(processor.default_std(), CLIP_STD);
    }

    #[test]
    fn dynamic_resolution_matches_checkpoint_budget() {
        let processor = MiniMaxM3VLProcessor::new();

        assert_eq!(processor.smart_resize(10, 10).unwrap(), (56, 56));
        // Official processor output for the 300 x 200 E2E pug fixture.
        assert_eq!(processor.smart_resize(200, 300).unwrap(), (196, 308));
        assert_eq!(processor.smart_resize(420, 840).unwrap(), (420, 840));
        assert_eq!(processor.smart_resize(1400, 1400).unwrap(), (672, 672));
    }

    #[test]
    fn preprocess_emits_flat_merge_grouped_patches() {
        let processor = MiniMaxM3VLProcessor::new();
        let result = processor
            .preprocess(
                &[solid_image(56, 56, [255, 0, 128])],
                &PreProcessorConfig::default(),
            )
            .unwrap();

        // 56 / 14 = 4 patches per axis; each flattened patch contains two
        // duplicated temporal frames: 3 * 2 * 14 * 14 = 1176 values.
        assert_eq!(result.encoder_input.shape(), &[16, 1176]);
        assert_eq!(result.feature_token_counts, vec![4]);

        match result.model_specific.get("image_grid_thw") {
            Some(ModelSpecificValue::IntTensor { data, shape }) => {
                assert_eq!(shape, &[1, 3]);
                assert_eq!(data, &[1, 4, 4]);
            }
            other => panic!("expected image_grid_thw IntTensor, got {other:?}"),
        }
        match result.model_specific.get("patches_per_image") {
            Some(ModelSpecificValue::IntTensor { data, shape }) => {
                assert_eq!(shape, &[1]);
                assert_eq!(data, &[16]);
            }
            other => panic!("expected patches_per_image IntTensor, got {other:?}"),
        }

        let patch = &result.encoder_input.as_slice_memory_order().unwrap()[..1176];
        let expected_red = (1.0 - CLIP_MEAN[0]) / CLIP_STD[0];
        let expected_green = (0.0 - CLIP_MEAN[1]) / CLIP_STD[1];
        let expected_blue = (128.0 / 255.0 - CLIP_MEAN[2]) / CLIP_STD[2];
        for (&actual, expected) in patch[..392].iter().zip(std::iter::repeat(expected_red)) {
            assert!((actual as f64 - expected).abs() < 1e-5);
        }
        for (&actual, expected) in patch[392..784]
            .iter()
            .zip(std::iter::repeat(expected_green))
        {
            assert!((actual as f64 - expected).abs() < 1e-5);
        }
        for (&actual, expected) in patch[784..].iter().zip(std::iter::repeat(expected_blue)) {
            assert!((actual as f64 - expected).abs() < 1e-5);
        }
    }

    #[test]
    fn checkpoint_size_sequence_does_not_disable_dynamic_resize() {
        let config = PreProcessorConfig::from_json(
            r#"{
                "size": [672, 672],
                "patch_size": 14,
                "image_mean": [0.48145466, 0.4578275, 0.40821073],
                "image_std": [0.26862954, 0.26130258, 0.27577711]
            }"#,
        )
        .unwrap();
        let processor = MiniMaxM3VLProcessor::new();
        let result = processor
            .preprocess(&[solid_image(420, 840, [128, 128, 128])], &config)
            .unwrap();

        assert_eq!(result.encoder_input.shape(), &[1800, 1176]);
        assert_eq!(result.feature_token_counts, vec![450]);
    }
}
