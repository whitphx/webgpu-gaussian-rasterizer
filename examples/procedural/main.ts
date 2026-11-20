import { createGaussianRasterizer, createPresenter, packFeatures } from '../../src/index.js';

const SIZE = 512;
const COUNT = 40_000;
const CHANNELS = 8;
const status = document.querySelector<HTMLOutputElement>('#status')!;

async function main() {
  if (!('gpu' in navigator)) throw new Error('This browser has no WebGPU.');
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error('No WebGPU adapter.');
  const device = await adapter.requestDevice();
  const canvas = document.querySelector<HTMLCanvasElement>('#view')!;
  const context = canvas.getContext('webgpu')!;
  const format = navigator.gpu.getPreferredCanvasFormat();
  context.configure({ device, format, alphaMode: 'opaque' });

  // Points on a sphere (a Fibonacci lattice), each with a direction-dependent feature vector.
  const base = new Float32Array(3 * COUNT);
  const features = new Float32Array(CHANNELS * COUNT);
  for (let i = 0; i < COUNT; i++) {
    const y = 1 - (2 * (i + 0.5)) / COUNT, r = Math.sqrt(1 - y * y), phi = i * Math.PI * (3 - Math.sqrt(5));
    const x = r * Math.cos(phi), z = r * Math.sin(phi);
    base.set([x, y, z], 3 * i);
    for (let c = 0; c < CHANNELS; c++) features[i * CHANNELS + c] = Math.sin((c + 1) * 2.1 * x + c * 1.3 * y + (c % 3) * z);
  }
  const s2 = 0.012 ** 2;
  const covariances = new Float32Array(6 * COUNT);
  for (let i = 0; i < COUNT; i++) covariances.set([s2, 0, 0, s2, 0, s2], 6 * i);

  const rasterizer = await createGaussianRasterizer(device, { capacity: COUNT, width: SIZE, height: SIZE, channels: CHANNELS });
  rasterizer.writeCovariances(covariances);
  rasterizer.writeOpacities(new Float32Array(COUNT).fill(0.8));
  rasterizer.writeFeatures(packFeatures(features, CHANNELS));

  // Stands in for a neural decoder: colour = sigmoid(M f) per pixel,
  // reading the rasterizer's float16 planes where they are on the GPU.
  const decoded = device.createBuffer({ size: 3 * SIZE * SIZE * 4, usage: GPUBufferUsage.STORAGE, label: 'decoded' });
  const weights = [
    [1.2, -0.4, 0.3, 0.8, -0.2, 0.5, 0.1, -0.6],
    [-0.3, 1.0, 0.6, -0.5, 0.7, -0.1, 0.4, 0.2],
    [0.5, 0.2, -0.8, 0.3, 0.4, 0.9, -0.5, 0.6],
  ];
  const plane = SIZE * SIZE;
  const decodeModule = device.createShaderModule({
    label: 'decode',
    code: /* wgsl */ `
@group(0) @binding(0) var<storage, read> features: array<u32>;
@group(0) @binding(1) var<storage, read_write> rgb: array<f32>;
fn feature(c: u32, i: u32) -> f32 {
  let k = c * ${plane}u + i;
  let w = unpack2x16float(features[k >> 1u]);
  return select(w.x, w.y, (k & 1u) == 1u);
}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= ${plane}u) { return; }
${weights.map((row, o) => `  rgb[${o * plane}u + i] = 1.0 / (1.0 + exp(-(${row.map((w, c) => `${w.toFixed(2)} * feature(${c}u, i)`).join(' + ')})));`).join('\n')}
}`,
  });
  const decodePipeline = device.createComputePipeline({ layout: 'auto', compute: { module: decodeModule, entryPoint: 'main' } });
  const decodeGroup = device.createBindGroup({
    layout: decodePipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: rasterizer.buffers.image } }, { binding: 1, resource: { buffer: decoded } }],
  });
  const presenters = {
    decoded: createPresenter(device, { source: decoded, width: SIZE, height: SIZE, sourceFormat: 'float32', targetFormat: format }),
    raw: createPresenter(device, { source: rasterizer.buffers.image, width: SIZE, height: SIZE, sourceFormat: 'float16', targetFormat: format, scale: 0.5, bias: 0.5 }),
  };

  const positions = new Float32Array(3 * COUNT);
  let frames = 0, since = performance.now();
  const frame = (now: number) => {
    const t = now / 1000;
    for (let i = 0; i < COUNT; i++) {
      const x = base[3 * i], y = base[3 * i + 1], z = base[3 * i + 2];
      const r = 0.6 * (1 + 0.08 * Math.sin(5 * y + 2 * t) * Math.cos(4 * x - t));
      positions[3 * i] = r * x; positions[3 * i + 1] = r * y; positions[3 * i + 2] = r * z;
    }
    rasterizer.writePositions(positions);
    // The camera circles the sphere at distance 2.4. Its rotation is
    // diag(-1, -1, 1) after a turn about y, which keeps world y up on the
    // screen with OpenCV's y-down camera axes.
    const a = 0.3 * t, c = Math.cos(a), s = Math.sin(a);
    rasterizer.setCamera({ view: [-c, 0, -s, 0, 0, -1, 0, 0, -s, 0, c, 0, 0, 0, 2.4, 1], fx: SIZE * 1.4, fy: SIZE * 1.4 });

    const encoder = device.createCommandEncoder();
    rasterizer.encode(encoder);
    const show = document.querySelector<HTMLInputElement>('input[name="show"]:checked')!.value as keyof typeof presenters;
    if (show === 'decoded') {
      const pass = encoder.beginComputePass();
      pass.setPipeline(decodePipeline);
      pass.setBindGroup(0, decodeGroup);
      pass.dispatchWorkgroups(Math.ceil(plane / 256));
      pass.end();
    }
    presenters[show].encode(encoder, context.getCurrentTexture());
    device.queue.submit([encoder.finish()]);

    frames++;
    if (now - since > 1000) {
      status.value = `${COUNT.toLocaleString()} Gaussians, ${CHANNELS} channels, ${((frames * 1000) / (now - since)).toFixed(0)} fps`;
      frames = 0;
      since = now;
    }
    requestAnimationFrame(frame);
  };
  requestAnimationFrame(frame);
}

main().catch((e) => { status.value = String(e instanceof Error ? e.message : e); });
