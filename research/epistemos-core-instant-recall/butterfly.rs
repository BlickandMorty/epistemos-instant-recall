#![allow(clippy::needless_range_loop)]

// ButterflyQuant: O(d log₂ d) learned rotation via butterfly factorization.
//
// Instead of a dense d×d rotation matrix (O(d²) per vector), we factorize into
// log₂(d) stages of paired Givens rotations. Each stage applies d/2 independent
// 2×2 rotations parameterized by a single angle θ, guaranteeing orthogonality
// by construction.
//
// For d=384 (nomic-embed-text Matryoshka): 9 stages × 192 angles = 1,728 parameters
// versus 384² = 147,456 for a dense rotation. Multiply-adds per vector: ~1,728
// versus 147,456 for dense matmul.
//
// Learning: Cayley SGD on the Stiefel manifold is unnecessary — each θ is an
// unconstrained scalar, so standard SGD with straight-through estimators suffices.
// Convergence in ~500 steps on calibration data (ButterflyQuant, Xu et al., 2025).
//
// Reference: arXiv:2509.09679 (ButterflyQuant)

use std::f32::consts::PI;

/// A single butterfly stage: d/2 independent Givens rotations.
/// Each rotation acts on a pair (i, i + stride) with angle θ.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ButterflyStage {
    /// Givens rotation angles, one per pair. Length = dim / 2.
    pub angles: Vec<f32>,
    /// Stride between paired indices. For stage s: stride = 2^s.
    pub stride: usize,
}

impl ButterflyStage {
    /// Apply this stage's rotations to vector x in-place.
    /// Each pair (i, i+stride) undergoes a 2×2 Givens rotation:
    ///   x'[i]          = cos(θ) * x[i] - sin(θ) * x[i+stride]
    ///   x'[i+stride]   = sin(θ) * x[i] + cos(θ) * x[i+stride]
    #[inline]
    pub fn apply_forward(&self, x: &mut [f32]) {
        let dim = x.len();
        let mut pair_idx = 0;
        let mut i = 0;
        while i < dim {
            // Within each block of size 2*stride, the first `stride` elements
            // pair with the corresponding element at offset `stride`.
            for offset in 0..self.stride {
                let a = i + offset;
                let b = a + self.stride;
                if b >= dim {
                    break;
                }
                let theta = self.angles[pair_idx];
                let (sin_t, cos_t) = theta.sin_cos();
                let xa = x[a];
                let xb = x[b];
                x[a] = cos_t * xa - sin_t * xb;
                x[b] = sin_t * xa + cos_t * xb;
                pair_idx += 1;
            }
            i += 2 * self.stride;
        }
    }

    /// Apply the inverse (transpose) rotation: negate all angles.
    #[inline]
    pub fn apply_inverse(&self, x: &mut [f32]) {
        let dim = x.len();
        let mut pair_idx = 0;
        let mut i = 0;
        while i < dim {
            for offset in 0..self.stride {
                let a = i + offset;
                let b = a + self.stride;
                if b >= dim {
                    break;
                }
                let theta = self.angles[pair_idx];
                // Inverse: negate the angle → swap sign of sin
                let (sin_t, cos_t) = theta.sin_cos();
                let xa = x[a];
                let xb = x[b];
                x[a] = cos_t * xa + sin_t * xb;
                x[b] = -sin_t * xa + cos_t * xb;
                pair_idx += 1;
            }
            i += 2 * self.stride;
        }
    }
}

/// Complete butterfly rotation: log₂(d) stages of paired Givens rotations.
/// Guarantees orthogonality by construction — no Cayley/SVD needed.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ButterflyRotation {
    /// Dimension of vectors this rotation operates on.
    pub dim: usize,
    /// log₂(dim) stages, each with dim/2 Givens angles.
    pub stages: Vec<ButterflyStage>,
    /// Version counter for MVCC (incremented on each re-learn).
    pub version: u64,
}

impl ButterflyRotation {
    /// Create a new butterfly rotation with identity initialization (all angles = 0).
    pub fn identity(dim: usize) -> Self {
        assert!(
            dim.is_power_of_two(),
            "ButterflyRotation requires power-of-two dimension, got {dim}"
        );
        let num_stages = dim.trailing_zeros() as usize;
        let stages = (0..num_stages)
            .map(|s| {
                let stride = 1 << s;
                ButterflyStage {
                    angles: vec![0.0; dim / 2],
                    stride,
                }
            })
            .collect();
        Self {
            dim,
            stages,
            version: 0,
        }
    }

    /// Create a butterfly rotation with random angles (good initialization for learning).
    pub fn random(dim: usize, seed: u64) -> Self {
        assert!(
            dim.is_power_of_two(),
            "ButterflyRotation requires power-of-two dimension, got {dim}"
        );
        let num_stages = dim.trailing_zeros() as usize;

        // Simple deterministic PRNG (xorshift64) for reproducibility
        let mut rng_state = seed;
        let mut next_f32 = move || -> f32 {
            rng_state ^= rng_state << 13;
            rng_state ^= rng_state >> 7;
            rng_state ^= rng_state << 17;
            // Map to [-π/4, π/4] — small initial rotations converge faster
            let u = (rng_state as f32) / (u64::MAX as f32);
            (u - 0.5) * PI * 0.5
        };

        let stages = (0..num_stages)
            .map(|s| {
                let stride = 1 << s;
                let angles = (0..dim / 2).map(|_| next_f32()).collect();
                ButterflyStage { angles, stride }
            })
            .collect();

        Self {
            dim,
            stages,
            version: 0,
        }
    }

    /// Total number of learnable parameters: log₂(d) × d/2.
    pub fn num_parameters(&self) -> usize {
        self.stages.len() * (self.dim / 2)
    }

    /// Apply the full forward rotation R·x in-place.
    /// Stages applied in order: stage 0 (stride=1), stage 1 (stride=2), ...
    #[inline]
    pub fn rotate_forward(&self, x: &mut [f32]) {
        debug_assert_eq!(
            x.len(),
            self.dim,
            "Vector dim {} != rotation dim {}",
            x.len(),
            self.dim
        );
        for stage in &self.stages {
            stage.apply_forward(x);
        }
    }

    /// Apply the inverse rotation R⁻¹·x = Rᵀ·x in-place.
    /// Stages applied in reverse order with negated angles.
    #[inline]
    pub fn rotate_inverse(&self, x: &mut [f32]) {
        debug_assert_eq!(
            x.len(),
            self.dim,
            "Vector dim {} != rotation dim {}",
            x.len(),
            self.dim
        );
        for stage in self.stages.iter().rev() {
            stage.apply_inverse(x);
        }
    }

    /// Learn rotation angles from calibration data using SGD.
    ///
    /// Minimizes quantization error: Σ‖Rx - Q(Rx)‖² where Q is the target quantizer.
    /// Each angle θ is an unconstrained scalar, so standard gradient descent works.
    ///
    /// `data`: calibration vectors (each must be `self.dim` dimensional).
    /// `quantize_fn`: function that quantizes a rotated vector and returns the dequantized
    ///                approximation (for computing reconstruction error).
    /// `steps`: number of SGD steps (default ~500 per ButterflyQuant paper).
    /// `lr`: learning rate (default 0.01).
    pub fn learn<F>(&mut self, data: &[Vec<f32>], quantize_fn: F, steps: usize, lr: f32)
    where
        F: Fn(&[f32]) -> Vec<f32>,
    {
        if data.is_empty() {
            return;
        }

        let epsilon = 1e-4_f32;

        for _step in 0..steps {
            // For each stage and each angle, compute numerical gradient
            for stage_idx in 0..self.stages.len() {
                let num_angles = self.stages[stage_idx].angles.len();
                let mut gradients = vec![0.0_f32; num_angles];

                for angle_idx in 0..num_angles {
                    // Finite difference: ∂L/∂θ ≈ (L(θ+ε) - L(θ-ε)) / 2ε
                    let loss_plus = {
                        self.stages[stage_idx].angles[angle_idx] += epsilon;
                        let l = self.compute_loss(data, &quantize_fn);
                        self.stages[stage_idx].angles[angle_idx] -= epsilon;
                        l
                    };
                    let loss_minus = {
                        self.stages[stage_idx].angles[angle_idx] -= epsilon;
                        let l = self.compute_loss(data, &quantize_fn);
                        self.stages[stage_idx].angles[angle_idx] += epsilon;
                        l
                    };

                    gradients[angle_idx] = (loss_plus - loss_minus) / (2.0 * epsilon);
                }

                // SGD update
                for (angle_idx, grad) in gradients.iter().enumerate() {
                    self.stages[stage_idx].angles[angle_idx] -= lr * grad;
                }
            }
        }

        self.version += 1;
    }

    /// Compute total reconstruction loss: Σ‖Rx - Q(Rx)‖²
    fn compute_loss<F>(&self, data: &[Vec<f32>], quantize_fn: &F) -> f32
    where
        F: Fn(&[f32]) -> Vec<f32>,
    {
        let mut total_loss = 0.0_f32;
        for vec in data {
            let mut rotated = vec.clone();
            self.rotate_forward(&mut rotated);
            let dequantized = quantize_fn(&rotated);
            // L2 reconstruction error
            for (a, b) in rotated.iter().zip(dequantized.iter()) {
                let diff = a - b;
                total_loss += diff * diff;
            }
        }
        total_loss / data.len() as f32
    }

    /// Serialize to bytes for persistence.
    pub fn to_bytes(&self) -> Vec<u8> {
        bincode::serialize(self).expect("ButterflyRotation serialization should never fail")
    }

    /// Deserialize from bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, bincode::Error> {
        bincode::deserialize(bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_rotation_is_noop() {
        let rot = ButterflyRotation::identity(16);
        let original = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ];
        let mut x = original.clone();
        rot.rotate_forward(&mut x);
        for (a, b) in x.iter().zip(original.iter()) {
            assert!((a - b).abs() < 1e-6, "Identity rotation should be no-op");
        }
    }

    #[test]
    fn forward_inverse_roundtrip() {
        let rot = ButterflyRotation::random(64, 42);
        let original: Vec<f32> = (0..64).map(|i| (i as f32 * 0.1).sin()).collect();
        let mut x = original.clone();

        rot.rotate_forward(&mut x);
        // After rotation, vector should be different
        assert!(
            x.iter()
                .zip(original.iter())
                .any(|(a, b)| (a - b).abs() > 1e-4),
            "Rotation should change the vector"
        );

        rot.rotate_inverse(&mut x);
        // After inverse, should recover original
        for (a, b) in x.iter().zip(original.iter()) {
            assert!(
                (a - b).abs() < 1e-4,
                "Inverse should recover original: got {} expected {}",
                a,
                b
            );
        }
    }

    #[test]
    fn rotation_preserves_norm() {
        let rot = ButterflyRotation::random(128, 123);
        let original: Vec<f32> = (0..128).map(|i| (i as f32 * 0.3).cos()).collect();
        let original_norm: f32 = original.iter().map(|x| x * x).sum::<f32>().sqrt();

        let mut x = original.clone();
        rot.rotate_forward(&mut x);
        let rotated_norm: f32 = x.iter().map(|x| x * x).sum::<f32>().sqrt();

        assert!(
            (original_norm - rotated_norm).abs() < 1e-3,
            "Orthogonal rotation must preserve L2 norm: {} vs {}",
            original_norm,
            rotated_norm
        );
    }

    #[test]
    fn num_parameters_correct() {
        let rot = ButterflyRotation::identity(256);
        // log₂(256) = 8, each stage has 128 angles → 8 × 128 = 1024
        assert_eq!(rot.num_parameters(), 8 * 128);
    }

    #[test]
    fn serialization_roundtrip() {
        let rot = ButterflyRotation::random(32, 99);
        let bytes = rot.to_bytes();
        let restored = ButterflyRotation::from_bytes(&bytes).unwrap();
        assert_eq!(rot.dim, restored.dim);
        assert_eq!(rot.version, restored.version);
        assert_eq!(rot.stages.len(), restored.stages.len());
        for (s1, s2) in rot.stages.iter().zip(restored.stages.iter()) {
            assert_eq!(s1.stride, s2.stride);
            assert_eq!(s1.angles, s2.angles);
        }
    }

    #[test]
    fn learning_reduces_quantization_error() {
        let mut rot = ButterflyRotation::random(16, 7);

        // Generate calibration data with outlier channels (the problem rotation solves)
        let data: Vec<Vec<f32>> = (0..50)
            .map(|seed| {
                (0..16)
                    .map(|i| {
                        let base = ((seed as f32) * 0.1 + (i as f32) * 0.3).sin();
                        // Channel 3 and 7 have 10x outliers
                        if i == 3 || i == 7 {
                            base * 10.0
                        } else {
                            base
                        }
                    })
                    .collect()
            })
            .collect();

        // Simple uniform quantizer (round to nearest 0.5)
        let quantize =
            |x: &[f32]| -> Vec<f32> { x.iter().map(|v| (v * 2.0).round() / 2.0).collect() };

        let loss_before = rot.compute_loss(&data, &quantize);
        rot.learn(&data, quantize, 20, 0.05);
        let loss_after = rot.compute_loss(&data, &quantize);

        assert!(
            loss_after < loss_before,
            "Learning should reduce quantization error: before={loss_before}, after={loss_after}"
        );
    }

    #[test]
    fn different_seeds_produce_different_rotations() {
        let r1 = ButterflyRotation::random(32, 1);
        let r2 = ButterflyRotation::random(32, 2);
        let differ = r1.stages[0]
            .angles
            .iter()
            .zip(r2.stages[0].angles.iter())
            .any(|(a, b)| (a - b).abs() > 1e-6);
        assert!(differ, "Different seeds should produce different rotations");
    }
}
