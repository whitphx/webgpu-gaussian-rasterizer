export interface PresenterOptions {
  /** A planar image: `planes` height x width planes, as the rasterizer writes it. */
  source: GPUBuffer;
  width: number;
  height: number;
  sourceFormat: 'float16' | 'float32';
  /** Format of the texture drawn into, e.g. `navigator.gpu.getPreferredCanvasFormat()`. */
  targetFormat: GPUTextureFormat;
  /** The source planes shown as red, green and blue. */
  rgb?: readonly [number, number, number];
  /** Shown value = scale * source + bias, clamped to [0, 1]. */
  scale?: number;
  bias?: number;
  /** Unsharp-mask strength on a 3x3 binomial blur; 0 for none. */
  sharpen?: number;
}

export interface Presenter {
  /**
   * Records a render pass that draws the image over the whole of `target`,
   * scaled by nearest-neighbour sampling when the sizes differ.
   */
  encode(encoder: GPUCommandEncoder, target: GPUTexture): void;
}

/** Draws three planes of a planar image, such as the rasterizer's or a network's output, into a texture. */
export function createPresenter(device: GPUDevice, options: PresenterOptions): Presenter {
  const { source, width, height, sourceFormat, targetFormat, rgb = [0, 1, 2], scale = 1, bias = 0, sharpen = 0 } = options;
  const fp16 = sourceFormat === 'float16';
  const plane = width * height;
  const f = (v: number) => (Number.isInteger(v) ? `${v}.0` : `${v}`);
  const code = /* wgsl */ `
@group(0) @binding(0) var<storage, read> src: array<${fp16 ? 'u32' : 'f32'}>;
@group(0) @binding(1) var<uniform> target_size: vec2f;
${fp16
    ? 'fn rd(i: u32) -> f32 { let w = unpack2x16float(src[i >> 1u]); return select(w.x, w.y, (i & 1u) == 1u); }'
    : 'fn rd(i: u32) -> f32 { return src[i]; }'}
fn px(x: i32, y: i32) -> vec3f {
  let i = u32(clamp(y, 0, ${height - 1})) * ${width}u + u32(clamp(x, 0, ${width - 1}));
  let v = vec3f(rd(${rgb[0] * plane}u + i), rd(${rgb[1] * plane}u + i), rd(${rgb[2] * plane}u + i));
  return clamp(${f(scale)} * v + ${f(bias)}, vec3f(0.0), vec3f(1.0));
}
@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
  let p = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
  return vec4f(p * 2.0 - 1.0, 0.0, 1.0);
}
@fragment fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let s = vec2i(floor(pos.xy * vec2f(${width}.0, ${height}.0) / target_size));
  let e = px(s.x, s.y);
${sharpen ? `  var blur = vec3f(0.0);
  for (var dy = -1; dy <= 1; dy++) { for (var dx = -1; dx <= 1; dx++) {
    blur += f32((2 - abs(dy)) * (2 - abs(dx))) * px(s.x + dx, s.y + dy);
  } }
  let o = e + ${f(sharpen)} * (e - blur / 16.0);` : '  let o = e;'}
  return vec4f(clamp(o, vec3f(0.0), vec3f(1.0)), 1.0);
}`;
  const module = device.createShaderModule({ code, label: 'presenter' });
  const pipeline = device.createRenderPipeline({
    label: 'presenter', layout: 'auto',
    vertex: { module, entryPoint: 'vs' },
    fragment: { module, entryPoint: 'fs', targets: [{ format: targetFormat }] },
    primitive: { topology: 'triangle-list' },
  });
  const sizeBuf = device.createBuffer({ size: 8, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, label: 'presenter_size' });
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: source } }, { binding: 1, resource: { buffer: sizeBuf } }],
  });
  let lastW = 0, lastH = 0;
  return {
    encode(encoder, target) {
      if (target.width !== lastW || target.height !== lastH) {
        lastW = target.width; lastH = target.height;
        device.queue.writeBuffer(sizeBuf, 0, new Float32Array([lastW, lastH]));
      }
      const pass = encoder.beginRenderPass({
        colorAttachments: [{ view: target.createView(), loadOp: 'clear', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 1 } }],
      });
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindGroup);
      // One triangle covers the viewport.
      pass.draw(3);
      pass.end();
    },
  };
}
