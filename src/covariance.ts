/**
 * 3D covariances in the rasterizer's layout (xx, xy, xz, yy, yz, zz per
 * Gaussian) from per-axis scales and rotation quaternions (w, x, y, z).
 *
 * `normalize: false` builds the rotation from the quaternion as given. Some
 * models emit quaternions that are not unit length and were trained against
 * a rasterizer that never normalizes them, so their Gaussians only come out
 * right this way.
 */
export function covariancesFromScaleRotation(
  scales: ArrayLike<number>,
  rotations: ArrayLike<number>,
  { normalize = true }: { normalize?: boolean } = {},
): Float32Array {
  const n = scales.length / 3;
  if (rotations.length !== 4 * n) throw new Error(`${n} scales but ${rotations.length / 4} rotations`);
  const out = new Float32Array(6 * n);
  for (let i = 0; i < n; i++) {
    let w = rotations[4 * i], x = rotations[4 * i + 1], y = rotations[4 * i + 2], z = rotations[4 * i + 3];
    if (normalize) {
      const len = Math.hypot(w, x, y, z) || 1;
      w /= len; x /= len; y /= len; z /= len;
    }
    const r = [
      1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
      2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
      2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ];
    const sx = scales[3 * i], sy = scales[3 * i + 1], sz = scales[3 * i + 2];
    // M = R diag(s), covariance = M M^T.
    const m = [r[0] * sx, r[1] * sy, r[2] * sz, r[3] * sx, r[4] * sy, r[5] * sz, r[6] * sx, r[7] * sy, r[8] * sz];
    const dot = (a: number, b: number) => m[3 * a] * m[3 * b] + m[3 * a + 1] * m[3 * b + 1] + m[3 * a + 2] * m[3 * b + 2];
    out.set([dot(0, 0), dot(0, 1), dot(0, 2), dot(1, 1), dot(1, 2), dot(2, 2)], 6 * i);
  }
  return out;
}
