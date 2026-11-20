const f32 = new Float32Array(1);
const u32 = new Uint32Array(f32.buffer);

/** IEEE half-precision bits of `value`, rounded to nearest even as numpy's `astype(float16)` rounds. */
export function toHalf(value: number): number {
  f32[0] = value;
  const x = u32[0];
  const sign = (x >>> 16) & 0x8000;
  const exp = (x >>> 23) & 0xff;
  let m = x & 0x7fffff;
  if (exp === 0xff) return sign | 0x7c00 | (m ? 0x200 : 0);
  const e = exp - 127 + 15;
  if (e >= 0x1f) return sign | 0x7c00;
  if (e <= 0) {
    if (e < -10) return sign;
    m |= 0x800000;
    const shift = 14 - e;
    const half = 1 << (shift - 1);
    const rem = m & ((1 << shift) - 1);
    let h = m >>> shift;
    if (rem > half || (rem === half && h & 1)) h++;
    return sign | h;
  }
  let h = (e << 10) | (m >>> 13);
  const rem = m & 0x1fff;
  if (rem > 0x1000 || (rem === 0x1000 && h & 1)) h++;
  return sign | h;
}

export function fromHalf(h: number): number {
  const s = h & 0x8000 ? -1 : 1;
  const e = (h >> 10) & 0x1f;
  const m = h & 0x3ff;
  if (e === 0) return s * m * 2 ** -24;
  if (e === 31) return m ? NaN : s * Infinity;
  return s * (1 + m / 1024) * 2 ** (e - 15);
}

/** Halves per Gaussian in the rasterizer's feature buffer: channels padded to whole u32 words. */
export function featureStride(channels: number): number {
  return 2 * Math.ceil(channels / 2);
}

/**
 * Packs `channels` float features per Gaussian (`values`, Gaussian-major) into
 * the half-precision layout `writeFeatures` takes.
 */
export function packFeatures(values: ArrayLike<number>, channels: number): Uint16Array {
  const count = values.length / channels;
  if (!Number.isInteger(count)) throw new Error(`${values.length} values is not a whole number of ${channels}-channel Gaussians`);
  const stride = featureStride(channels);
  const out = new Uint16Array(count * stride);
  for (let i = 0; i < count; i++) {
    for (let c = 0; c < channels; c++) out[i * stride + c] = toHalf(values[i * channels + c]);
  }
  return out;
}
