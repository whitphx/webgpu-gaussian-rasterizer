// Renders the scene parity.py wrote with the built library under Dawn and
// writes the image next to it as image.f32 (planar, float32).
//
//   node render-webgpu.mjs <scene dir>

import { readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { create, globals } from 'webgpu';
import { createGaussianRasterizer, packFeatures } from '../../dist/index.js';

Object.assign(globalThis, globals);

const dir = process.argv[2];
const scene = JSON.parse(readFileSync(join(dir, 'scene.json'), 'utf8'));
const f32 = (name) => {
  const b = readFileSync(join(dir, `${name}.f32`));
  return new Float32Array(b.buffer, b.byteOffset, b.byteLength / 4);
};

const gpu = create([]);
const adapter = await gpu.requestAdapter();
if (!adapter) throw new Error('Dawn found no WebGPU adapter');
const device = await adapter.requestDevice();
const { count, channels, width, height } = scene;
const r = await createGaussianRasterizer(device, {
  capacity: count, width, height, channels, outputFormat: 'float32',
  tileTest: scene.tileTest, background: scene.background, pairCapacity: 1 << 23,
});
r.writePositions(f32('positions'));
r.writeCovariances(f32('covariances'));
r.writeOpacities(f32('opacities'));
r.writeFeatures(packFeatures(f32('features'), channels));
r.setCamera(scene.camera);
r.render({ readback: true });
const stats = await r.readStats();
if (stats.overflow) throw new Error(`${stats.overflow} tile pairs overflowed`);

const bytes = r.planes * width * height * 4;
const read = device.createBuffer({ size: bytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
const enc = device.createCommandEncoder();
enc.copyBufferToBuffer(r.buffers.image, 0, read, 0, bytes);
device.queue.submit([enc.finish()]);
await read.mapAsync(GPUMapMode.READ);
writeFileSync(join(dir, 'image.f32'), new Uint8Array(read.getMappedRange().slice(0)));
read.unmap();
console.log(JSON.stringify({ adapter: adapter.info.description || adapter.info.vendor, pairs: stats.pairs }));
device.destroy();
