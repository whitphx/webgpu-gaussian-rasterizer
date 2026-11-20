// A per-pixel CPU rasterizer with the conventions the kernels follow (EWA
// projection with 0.3 px dilation, alpha capped at 0.99, contributions under
// 1/255 skipped, a pixel stopping once its transmittance would fall under
// 1e-4) and none of their machinery: no tiles, no key quantization, every
// Gaussian tested at every pixel. Gaussians are composited by depth, ties in
// index order.

import type { Camera } from '../src/index.js';

export interface Scene {
  count: number;
  channels: number;
  positions: Float32Array;
  covariances: Float32Array;
  opacities: Float32Array;
  features: Float32Array;
}

export function referenceImage(
  scene: Scene, camera: Required<Camera>, width: number, height: number,
  { background = null, alpha = false, near = 0.2 }: { background?: number[] | null; alpha?: boolean; near?: number } = {},
): Float32Array {
  const { view: v, fx, fy, cx, cy } = camera;
  const tanX = width / (2 * fx), tanY = height / (2 * fy);
  const splats: { i: number; z: number; mx: number; my: number; ca: number; cb: number; cc: number; op: number }[] = [];
  for (let i = 0; i < scene.count; i++) {
    const [x, y, z] = scene.positions.subarray(3 * i, 3 * i + 3);
    // Column-major: t = V [x y z 1].
    const tx = v[0] * x + v[4] * y + v[8] * z + v[12];
    const ty = v[1] * x + v[5] * y + v[9] * z + v[13];
    const tz = v[2] * x + v[6] * y + v[10] * z + v[14];
    if (tz < near) continue;
    const op = scene.opacities[i];
    const limx = 1.3 * tanX, limy = 1.3 * tanY;
    const txtz = Math.min(limx, Math.max(-limx, tx / tz)) * tz;
    const tytz = Math.min(limy, Math.max(-limy, ty / tz)) * tz;
    // Rows of J (2x3) and of W, the view's rotation.
    const J = [[fx / tz, 0, -fx * txtz / (tz * tz)], [0, fy / tz, -fy * tytz / (tz * tz)]];
    const W = [[v[0], v[4], v[8]], [v[1], v[5], v[9]], [v[2], v[6], v[10]]];
    const T = J.map((row) => [0, 1, 2].map((c) => row[0] * W[0][c] + row[1] * W[1][c] + row[2] * W[2][c]));
    const c6 = scene.covariances.subarray(6 * i, 6 * i + 6);
    const Sg = [[c6[0], c6[1], c6[2]], [c6[1], c6[3], c6[4]], [c6[2], c6[4], c6[5]]];
    const TS = T.map((row) => [0, 1, 2].map((c) => row[0] * Sg[0][c] + row[1] * Sg[1][c] + row[2] * Sg[2][c]));
    const C = (r: number, c: number) => TS[r][0] * T[c][0] + TS[r][1] * T[c][1] + TS[r][2] * T[c][2];
    const a = C(0, 0) + 0.3, b = C(0, 1), c = C(1, 1) + 0.3;
    const det = a * c - b * b;
    if (det === 0) continue;
    splats.push({ i, z: tz, mx: fx * (tx / tz) + cx, my: fy * (ty / tz) + cy, ca: c / det, cb: -b / det, cc: a / det, op });
  }
  splats.sort((p, q) => p.z - q.z || p.i - q.i);
  const C = scene.channels, planes = C + (alpha ? 1 : 0);
  const out = new Float32Array(planes * width * height);
  for (let py = 0; py < height; py++) {
    for (let px = 0; px < width; px++) {
      let T = 1;
      const acc = new Float64Array(C);
      for (const s of splats) {
        const dx = s.mx - px, dy = s.my - py;
        const power = -0.5 * (s.ca * dx * dx + s.cc * dy * dy) - s.cb * dx * dy;
        if (power > 0) continue;
        const al = Math.min(0.99, s.op * Math.exp(power));
        if (al < 1 / 255) continue;
        const next = T * (1 - al);
        if (next < 1e-4) break;
        for (let k = 0; k < C; k++) acc[k] += scene.features[s.i * C + k] * al * T;
        T = next;
      }
      const pix = py * width + px;
      for (let k = 0; k < C; k++) out[k * width * height + pix] = acc[k] + (background ? T * background[k] : 0);
      if (alpha) out[C * width * height + pix] = 1 - T;
    }
  }
  return out;
}

/** PSNR in dB of `actual` against `expected`, with the peak taken as the largest |expected|. */
export function psnr(actual: ArrayLike<number>, expected: ArrayLike<number>): number {
  let se = 0, peak = 0;
  for (let i = 0; i < expected.length; i++) {
    se += (actual[i] - expected[i]) ** 2;
    peak = Math.max(peak, Math.abs(expected[i]));
  }
  return 10 * Math.log10((peak * peak) / (se / expected.length));
}
