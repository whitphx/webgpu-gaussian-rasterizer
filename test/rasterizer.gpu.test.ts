import { afterAll, afterEach, beforeAll, beforeEach, describe, expect, test } from 'vitest';
import { create, globals } from 'webgpu';
import { createGaussianRasterizer, fromHalf, packFeatures, toHalf, type GaussianRasterizer, type RasterizerOptions } from '../src/index.js';
import { psnr, referenceImage, type Scene } from './reference.js';
import { lookAt, randomScene } from './scene.js';

Object.assign(globalThis, globals);

// The GPU object has to outlive the device, or Dawn crashes when it is collected.
const gpu = create([]);
const W = 96, H = 64;
const camera = { view: lookAt(3), fx: 80, fy: 80, cx: (W - 1) / 2, cy: (H - 1) / 2 };

let device: GPUDevice | undefined;
const errors: string[] = [];
// Dawn crashes later in the run when the adapter is requested while the
// test module loads, so the device comes in beforeAll.
beforeAll(async () => {
  device = await (await gpu.requestAdapter())?.requestDevice();
  device?.addEventListener('uncapturederror', (e) => errors.push((e as GPUUncapturedErrorEvent).error.message));
});
// Without a GPU adapter the tests are skipped.
beforeEach(({ skip }) => { if (!device) skip(); });
afterEach(() => expect(errors.splice(0)).toEqual([]));
afterAll(() => device?.destroy());

// The rasterizer stores features in half precision, so the reference gets the same rounding.
function halfRounded(scene: Scene): Scene {
  return { ...scene, features: Float32Array.from(scene.features, (v) => fromHalf(toHalf(v))) };
}

async function upload(scene: Scene, options: Partial<RasterizerOptions> = {}): Promise<GaussianRasterizer> {
  const r = await createGaussianRasterizer(device!, { capacity: scene.count, width: W, height: H, channels: scene.channels, outputFormat: 'float32', ...options });
  r.writePositions(scene.positions);
  r.writeCovariances(scene.covariances);
  r.writeOpacities(scene.opacities);
  r.writeFeatures(packFeatures(scene.features, scene.channels));
  r.setCamera(camera);
  return r;
}

async function readImage(r: GaussianRasterizer): Promise<Float32Array> {
  const bytes = r.planes * W * H * (r.outputFormat === 'float16' ? 2 : 4);
  const read = device!.createBuffer({ size: bytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const enc = device!.createCommandEncoder();
  enc.copyBufferToBuffer(r.buffers.image, 0, read, 0, bytes);
  device!.queue.submit([enc.finish()]);
  await read.mapAsync(GPUMapMode.READ);
  const raw = read.getMappedRange().slice(0);
  read.destroy();
  return r.outputFormat === 'float16' ? Float32Array.from(new Uint16Array(raw), fromHalf) : new Float32Array(raw);
}

describe('createGaussianRasterizer', () => {
  test.each([3, 5, 16, 32])('matches the reference with %i channels', async (channels) => {
    const scene = halfRounded(randomScene(400, channels));
    const r = await upload(scene);
    r.render({ readback: true });
    const image = await readImage(r);
    const stats = await r.readStats();
    expect(stats.overflow).toBe(0);
    expect(stats.pairs).toBeGreaterThan(0);
    expect(psnr(image, referenceImage(scene, camera, W, H))).toBeGreaterThan(100);
    r.destroy();
  });

  test('box tiles drop only the far tails the reference keeps', async () => {
    const scene = halfRounded(randomScene(400, 4));
    const r = await upload(scene, { tileTest: 'box' });
    r.render();
    expect(psnr(await readImage(r), referenceImage(scene, camera, W, H))).toBeGreaterThan(45);
    r.destroy();
  });

  test('float16 output rounds the float32 image', async () => {
    const scene = halfRounded(randomScene(400, 6));
    const r = await upload(scene, { outputFormat: 'float16' });
    r.render();
    expect(psnr(await readImage(r), referenceImage(scene, camera, W, H))).toBeGreaterThan(60);
    r.destroy();
  });

  test('writes the background and an alpha plane', async () => {
    const scene = halfRounded(randomScene(300, 3));
    const background = [0.25, -1, 2];
    const r = await upload(scene, { background, outputAlpha: true });
    expect(r.planes).toBe(4);
    r.render();
    const image = await readImage(r);
    expect(psnr(image, referenceImage(scene, camera, W, H, { background, alpha: true }))).toBeGreaterThan(100);
    // A corner pixel no Gaussian reaches shows the background and zero alpha.
    expect([0, 1, 2, 3].map((k) => image[k * W * H])).toEqual([0.25, -1, 2, 0]);
    r.destroy();
  });

  test('follows per-frame updates and a smaller count', async () => {
    const scene = halfRounded(randomScene(500, 4, 7));
    const r = await upload(scene);
    r.render();
    await readImage(r);
    const moved = { ...scene, positions: scene.positions.map((v, k) => (k % 3 === 0 ? v + 0.1 : v)) };
    r.writePositions(moved.positions);
    r.render();
    expect(psnr(await readImage(r), referenceImage(moved, camera, W, H))).toBeGreaterThan(100);
    r.setCount(200);
    r.render();
    expect(psnr(await readImage(r), referenceImage({ ...moved, count: 200 }, camera, W, H))).toBeGreaterThan(100);
    r.setCount(0);
    r.render();
    expect((await readImage(r)).every((v) => v === 0)).toBe(true);
    r.destroy();
  });

  test('renders into a buffer the caller owns', async () => {
    const scene = halfRounded(randomScene(200, 2));
    const image = device!.createBuffer({ size: 2 * W * H * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const r = await upload(scene, { image });
    expect(r.buffers.image).toBe(image);
    r.render();
    expect(psnr(await readImage(r), referenceImage(scene, camera, W, H))).toBeGreaterThan(100);
    r.destroy();
    image.destroy();
  });

  test('reports pairs beyond its capacity', async () => {
    const scene = halfRounded(randomScene(400, 2));
    const r = await upload(scene, { pairCapacity: 64 });
    r.render({ readback: true });
    const stats = await r.readStats();
    expect(stats.pairs).toBe(64);
    expect(stats.overflow).toBeGreaterThan(64);
    r.destroy();
  });
});
