import { covariancesFromScaleRotation } from '../src/index.js';
import type { Scene } from './reference.js';

// mulberry32: a small seeded generator, so every run draws the same scene.
export function random(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Anisotropic Gaussians in a box in front of the camera of `lookAt`. */
export function randomScene(count: number, channels: number, seed = 1): Scene {
  const r = random(seed);
  const positions = new Float32Array(3 * count), scales = new Float32Array(3 * count), rotations = new Float32Array(4 * count);
  const opacities = new Float32Array(count), features = new Float32Array(count * channels);
  for (let i = 0; i < count; i++) {
    positions.set([(r() - 0.5) * 1.2, (r() - 0.5) * 1.2, (r() - 0.5) * 0.8], 3 * i);
    scales.set([0.01 + 0.06 * r(), 0.01 + 0.06 * r(), 0.01 + 0.03 * r()], 3 * i);
    rotations.set([r() - 0.5, r() - 0.5, r() - 0.5, r() - 0.5], 4 * i);
    opacities[i] = 0.05 + 0.9 * r();
    for (let c = 0; c < channels; c++) features[i * channels + c] = r() * 2 - 0.5;
  }
  return { count, channels, positions, covariances: covariancesFromScaleRotation(scales, rotations), opacities, features };
}

/** Column-major view for a camera at distance `d` on -z looking at the origin, OpenCV axes. */
export function lookAt(d: number): Float32Array {
  return new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, d, 1]);
}
