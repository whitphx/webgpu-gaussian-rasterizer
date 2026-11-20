import { describe, expect, test } from 'vitest';
import { covariancesFromScaleRotation, featureStride, fromHalf, packFeatures, toHalf } from '../src/index.js';

describe('half precision', () => {
  test.each([
    [0, 0x0000], [1, 0x3c00], [-2, 0xc000], [0.5, 0x3800], [65504, 0x7bff], [1e6, 0x7c00],
    [2 ** -24, 0x0001], [2 ** -26, 0x0000],
    // Halfway between 1 and the next half rounds to even, down to 1.
    [1 + 2 ** -11, 0x3c00],
    [1 + 3 * 2 ** -11, 0x3c02],
  ])('toHalf(%f) = %i', (v, bits) => {
    expect(toHalf(v)).toBe(bits);
  });

  test('fromHalf inverts toHalf on every finite half', () => {
    for (let h = 0; h < 0x10000; h++) {
      if ((h & 0x7c00) === 0x7c00) continue;
      expect(toHalf(fromHalf(h))).toBe(h === 0x8000 ? 0x8000 : h);
    }
  });

  test('packFeatures pads odd channel counts to whole words', () => {
    expect(featureStride(3)).toBe(4);
    expect(Array.from(packFeatures([1, 2, 3, -1, -2, -3], 3))).toEqual([0x3c00, 0x4000, 0x4200, 0, 0xbc00, 0xc000, 0xc200, 0]);
    expect(() => packFeatures([1, 2, 3, 4], 3)).toThrow();
  });
});

describe('covariancesFromScaleRotation', () => {
  test('is diag(s^2) without rotation', () => {
    expect(Array.from(covariancesFromScaleRotation([1, 2, 3], [1, 0, 0, 0]))).toEqual([1, 0, 0, 4, 0, 9]);
  });

  test('turns the axes with the quaternion', () => {
    // A quarter turn about z swaps the x and y variances.
    const h = Math.SQRT1_2;
    const c = covariancesFromScaleRotation([1, 2, 3], [h, 0, 0, h]);
    [4, 0, 0, 1, 0, 9].forEach((v, k) => expect(c[k]).toBeCloseTo(v, 6));
  });

  test('keeps a non-unit quaternion as given when asked', () => {
    // (1, 1, 0, 0) normalized is a half turn about x, which leaves diag(s^2) unchanged.
    const unit = covariancesFromScaleRotation([1, 2, 3], [1, 1, 0, 0]);
    const raw = covariancesFromScaleRotation([1, 2, 3], [1, 1, 0, 0], { normalize: false });
    [1, 0, 0, 9, 0, 4].forEach((v, k) => expect(unit[k]).toBeCloseTo(v, 6));
    expect(Array.from(raw)).not.toEqual(Array.from(unit));
  });
});
