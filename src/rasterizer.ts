import {
  TILE, SCAN_BLOCK, SORT_BLOCK, RADIX_PASSES, PARAMS_BYTES,
  projectShader, scanLocalShader, scanBlocksShader, scanHistShader, emitShader,
  upsweepShader, downsweepShader, rangesShader, rasterizeShader,
} from './shaders.js';
import { featureStride } from './half.js';

export const STAGES = ['project', 'scan', 'emit', 'sort', 'ranges', 'rasterize'] as const;
export type Stage = (typeof STAGES)[number];

export interface RasterizerOptions {
  /** Most Gaussians a frame can hold; `setCount` picks how many of them are drawn. */
  capacity: number;
  width: number;
  height: number;
  /** Feature channels composited per pixel. */
  channels: number;
  /**
   * How Gaussians are binned into tiles. `'box'` takes the square around
   * each Gaussian's 3-sigma circle, as the reference CUDA rasterizer does.
   * `'ellipse'` keeps only the tiles where its alpha can reach 1/255, which
   * leaves the image as it is and usually halves the work.
   */
  tileTest?: 'ellipse' | 'box';
  /** Element type of the output image. `'float16'` needs an even width. */
  outputFormat?: 'float16' | 'float32';
  /** Adds a plane after the channels with the composited opacity. */
  outputAlpha?: boolean;
  /** Per-channel value seen through transparent pixels. */
  background?: readonly number[];
  /** Most (tile, Gaussian) pairs a frame can hold; `readStats` reports overflow. */
  pairCapacity?: number;
  /**
   * Renders into this buffer instead of one the rasterizer allocates, so a
   * consumer bound to it (an ONNX Runtime Web session, say) keeps its
   * binding when the rasterizer is recreated. Needs STORAGE usage.
   */
  image?: GPUBuffer;
  /** Usage flags added to the image buffer the rasterizer allocates. */
  imageUsage?: GPUBufferUsageFlags;
  /** Per-stage GPU timings, when the device has 'timestamp-query'. */
  timestamps?: boolean;
  /** Camera-space depth below which Gaussians are culled. */
  near?: number;
}

export interface Camera {
  /**
   * World-to-camera transform, a column-major 4x4 matrix for column vectors
   * (WGSL's `mat4x4f` layout). Camera space has x right, y down and z
   * forward, the OpenCV convention.
   */
  view: ArrayLike<number>;
  fx: number;
  fy: number;
  /** Principal point in pixels; pixel (0, 0)'s centre is at (0, 0). Defaults to the image centre. */
  cx?: number;
  cy?: number;
}

export interface FrameStats {
  /** (tile, Gaussian) pairs the frame needed. */
  pairs: number;
  /** Pairs beyond `pairCapacity`, which were dropped; 0 when everything fit. */
  overflow: number;
  depthMin: number;
  depthMax: number;
}

export interface GaussianRasterizer {
  readonly width: number;
  readonly height: number;
  readonly channels: number;
  readonly capacity: number;
  readonly count: number;
  /** Planes in the output image: the channels, plus one with `outputAlpha`. */
  readonly planes: number;
  readonly outputFormat: 'float16' | 'float32';
  readonly timestamps: boolean;
  /**
   * The Gaussians, which a caller's own compute passes may also write:
   * positions as 3 f32, covariances as 6 f32 (xx, xy, xz, yy, yz, zz),
   * opacities as 1 f32 and features as `featureStride(channels)` halves per
   * Gaussian; the image as `planes` planar height x width planes.
   */
  readonly buffers: {
    readonly positions: GPUBuffer;
    readonly covariances: GPUBuffer;
    readonly opacities: GPUBuffer;
    readonly features: GPUBuffer;
    readonly image: GPUBuffer;
  };
  setCount(count: number): void;
  writePositions(data: Float32Array, first?: number): void;
  writeCovariances(data: Float32Array, first?: number): void;
  writeOpacities(data: Float32Array, first?: number): void;
  /** Half-precision bits laid out as `packFeatures` returns them. */
  writeFeatures(data: Uint16Array, first?: number): void;
  setCamera(camera: Camera): void;
  /**
   * Records one frame into `encoder`. The Gaussians, the count and the
   * camera are read when the encoder's commands run; the write* and
   * set* calls go through the queue, so they apply from the next submit.
   * `readback` adds the copies readStats and readTimings need.
   */
  encode(encoder: GPUCommandEncoder, options?: { readback?: boolean }): void;
  /** Encodes and submits one frame. */
  render(options?: { readback?: boolean }): void;
  /** The last frame encoded with readback. */
  readStats(): Promise<FrameStats>;
  /** GPU milliseconds per stage of the last frame encoded with readback, or null without timestamps. */
  readTimings(): Promise<Record<Stage | 'total', number> | null>;
  destroy(): void;
}

export async function createGaussianRasterizer(device: GPUDevice, options: RasterizerOptions): Promise<GaussianRasterizer> {
  const {
    capacity, width, height, channels, tileTest = 'ellipse', outputFormat = 'float16', outputAlpha = false,
    background, pairCapacity = 1 << 21, image = null, imageUsage = 0, timestamps = false, near = 0.2,
  } = options;
  for (const [k, v] of Object.entries({ capacity, width, height, channels, pairCapacity })) {
    if (!Number.isInteger(v) || v <= 0) throw new Error(`${k} must be a positive integer (got ${v})`);
  }
  const outputF16 = outputFormat === 'float16';
  if (outputF16 && width % 2) throw new Error(`float16 output packs pixel pairs, so the width must be even (got ${width})`);
  if (background && background.length !== channels) throw new Error(`background has ${background.length} values for ${channels} channels`);
  const tilesX = Math.ceil(width / TILE), tilesY = Math.ceil(height / TILE);
  const tileBits = Math.max(1, Math.ceil(Math.log2(tilesX * tilesY)));
  // The sort key holds the tile and the quantized depth in 32 bits. Depth
  // gets at most 24 bits: the kernel quantizes it in f32, whose 24-bit
  // significand cannot hold a larger maximum code, which would then round
  // up into the tile bits.
  const depthBits = Math.min(24, 32 - tileBits);
  if (depthBits < 12) throw new Error(`${width}x${height} leaves ${depthBits} bits of the sort key for depth`);
  const planes = channels + (outputAlpha ? 1 : 0);
  const imageBytes = planes * width * height * (outputF16 ? 2 : 4);
  if (image && image.size < imageBytes) throw new Error(`image buffer holds ${image.size} bytes, the frame needs ${imageBytes}`);
  const words = featureStride(channels) / 2;
  const sortBlocksCap = Math.ceil(pairCapacity / SORT_BLOCK);
  const useTimestamps = timestamps && device.features.has('timestamp-query');

  const mk = (bytes: number, usage: GPUBufferUsageFlags, label: string) =>
    device.createBuffer({ size: Math.max(16, Math.ceil(bytes / 16) * 16), usage, label });
  const S = GPUBufferUsage.STORAGE, CD = GPUBufferUsage.COPY_DST, CS = GPUBufferUsage.COPY_SRC;
  const maxScanBlocks = Math.ceil(capacity / SCAN_BLOCK);
  const buf = {
    params: mk(PARAMS_BYTES, GPUBufferUsage.UNIFORM | CD, 'params'),
    positions: mk(capacity * 12, S | CD | CS, 'positions'),
    covariances: mk(capacity * 24, S | CD | CS, 'covariances'),
    opacities: mk(capacity * 4, S | CD | CS, 'opacities'),
    features: mk(capacity * words * 4, S | CD | CS, 'features'),
    proj: mk(capacity * 56, S, 'proj'),
    tileCounts: mk(capacity * 4, S, 'tile_counts'),
    prefix: mk(capacity * 4, S, 'prefix'),
    blockSums: mk(maxScanBlocks * 4, S, 'block_sums'),
    blockOffsets: mk(maxScanBlocks * 4, S, 'block_offsets'),
    counters: mk(16, S | CD | CS, 'counters'),
    countersInit: mk(16, CS | CD, 'counters_init'),
    indirect: mk(24, S | GPUBufferUsage.INDIRECT, 'indirect'),
    keysA: mk(pairCapacity * 4, S, 'keys_a'),
    valsA: mk(pairCapacity * 4, S, 'vals_a'),
    keysB: mk(pairCapacity * 4, S, 'keys_b'),
    valsB: mk(pairCapacity * 4, S, 'vals_b'),
    hist: mk(16 * sortBlocksCap * 4, S, 'hist'),
    histScan: mk(16 * sortBlocksCap * 4, S, 'hist_scan'),
    ranges: mk(tilesX * tilesY * 8, S | CD, 'ranges'),
  };
  const imageBuf = image ?? mk(imageBytes, S | CS | imageUsage, 'image');
  // The depth range starts empty: min at the largest bit pattern, max at 0.
  device.queue.writeBuffer(buf.countersInit, 0, new Uint32Array([0, 0xffffffff, 0, 0]));

  const paramsData = new ArrayBuffer(PARAMS_BYTES);
  const pf = new Float32Array(paramsData), pu = new Uint32Array(paramsData);
  pu[20] = width; pu[21] = height; pu[22] = tilesX; pu[23] = tilesY;
  pu[25] = depthBits; pu[26] = pairCapacity; pu[27] = tileTest === 'ellipse' ? 1 : 0;
  pf[31] = near;
  let count = capacity;
  let cameraSet = false;
  const writeParams = () => device.queue.writeBuffer(buf.params, 0, paramsData);
  const setCount = (n: number) => {
    if (!Number.isInteger(n) || n < 0 || n > capacity) throw new Error(`count must be an integer in [0, ${capacity}] (got ${n})`);
    count = n;
    pu[24] = n;
    pu[30] = Math.ceil(n / SCAN_BLOCK);
    writeParams();
  };
  setCount(capacity);

  const pipe = (code: string, label: string) => device.createComputePipelineAsync({
    label, layout: 'auto', compute: { module: device.createShaderModule({ code, label }), entryPoint: 'main' },
  });
  const [pProject, pScanLocal, pScanBlocks, pScanHist, pEmit, pRanges, pRaster] = await Promise.all([
    pipe(projectShader, 'project'), pipe(scanLocalShader, 'scan_local'), pipe(scanBlocksShader, 'scan_blocks'),
    pipe(scanHistShader, 'scan_hist'), pipe(emitShader, 'emit'), pipe(rangesShader, 'ranges'),
    pipe(rasterizeShader({ channels, outputF16, background: background ?? null, alpha: outputAlpha }), 'rasterize'),
  ]);
  const pUp = await Promise.all([...Array(RADIX_PASSES)].map((_, p) => pipe(upsweepShader(4 * p), `upsweep_${p}`)));
  const pDown = await Promise.all([...Array(RADIX_PASSES)].map((_, p) => pipe(downsweepShader(4 * p), `downsweep_${p}`)));

  // A null entry skips a binding the shader does not reference (auto layouts drop those).
  const bg = (pipeline: GPUComputePipeline, buffers: (GPUBuffer | null)[]) => device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: buffers.flatMap((b, i) => (b ? [{ binding: i, resource: { buffer: b } }] : [])),
  });
  const bgProject = bg(pProject, [buf.params, buf.positions, buf.covariances, buf.opacities, buf.proj, buf.tileCounts, buf.counters]);
  const bgScanLocal = bg(pScanLocal, [buf.params, buf.tileCounts, buf.prefix, buf.blockSums]);
  const bgScanBlocks = bg(pScanBlocks, [buf.params, buf.blockSums, buf.blockOffsets, buf.counters, buf.indirect]);
  const bgScanHist = bg(pScanHist, [null, buf.hist, buf.histScan, buf.counters]);
  const bgEmit = bg(pEmit, [buf.params, buf.proj, buf.tileCounts, buf.prefix, buf.blockOffsets, buf.counters, buf.keysA, buf.valsA]);
  const pingpong = [[buf.keysA, buf.valsA, buf.keysB, buf.valsB], [buf.keysB, buf.valsB, buf.keysA, buf.valsA]];
  const bgUp = pUp.map((p, i) => bg(p, [pingpong[i % 2][0], buf.hist, buf.counters]));
  const bgDown = pDown.map((p, i) => bg(p, [...pingpong[i % 2], buf.histScan, buf.counters]));
  // An even number of passes leaves the sorted keys back in the A buffers.
  const bgRanges = bg(pRanges, [buf.params, buf.keysA, buf.counters, buf.ranges]);
  const bgRaster = bg(pRaster, [buf.params, buf.proj, buf.valsA, buf.ranges, buf.features, imageBuf]);

  let querySet: GPUQuerySet | null = null, queryBuf: GPUBuffer | null = null, queryRead: GPUBuffer | null = null;
  if (useTimestamps) {
    querySet = device.createQuerySet({ type: 'timestamp', count: 2 * STAGES.length });
    queryBuf = device.createBuffer({ size: 16 * STAGES.length, usage: GPUBufferUsage.QUERY_RESOLVE | CS });
    queryRead = device.createBuffer({ size: 16 * STAGES.length, usage: GPUBufferUsage.MAP_READ | CD });
  }
  const countersRead = device.createBuffer({ size: 16, usage: GPUBufferUsage.MAP_READ | CD });

  const checkWrite = (what: string, length: number, per: number, first: number) => {
    if (length % per || !Number.isInteger(first) || first < 0 || first + length / per > capacity) {
      throw new Error(`${what}: ${length} values from Gaussian ${first} do not fit ${capacity} Gaussians of ${per}`);
    }
  };

  function encode(encoder: GPUCommandEncoder, { readback = false }: { readback?: boolean } = {}) {
    if (!cameraSet) throw new Error('setCamera before the first frame');
    encoder.copyBufferToBuffer(buf.countersInit, 0, buf.counters, 0, 16);
    encoder.clearBuffer(buf.ranges);
    let stage = 0;
    const pass = () => {
      const desc: GPUComputePassDescriptor = {};
      if (querySet) desc.timestampWrites = { querySet, beginningOfPassWriteIndex: 2 * stage, endOfPassWriteIndex: 2 * stage + 1 };
      stage++;
      return encoder.beginComputePass(desc);
    };
    const groups = Math.ceil(count / 256), scanBlocks = Math.ceil(count / SCAN_BLOCK);
    let p = pass();
    p.setPipeline(pProject); p.setBindGroup(0, bgProject); if (groups) p.dispatchWorkgroups(groups);
    p.end();
    p = pass();
    p.setPipeline(pScanLocal); p.setBindGroup(0, bgScanLocal); if (scanBlocks) p.dispatchWorkgroups(scanBlocks);
    p.setPipeline(pScanBlocks); p.setBindGroup(0, bgScanBlocks); p.dispatchWorkgroups(1);
    p.end();
    p = pass();
    p.setPipeline(pEmit); p.setBindGroup(0, bgEmit); if (groups) p.dispatchWorkgroups(groups);
    p.end();
    p = pass();
    for (let i = 0; i < RADIX_PASSES; i++) {
      p.setPipeline(pUp[i]); p.setBindGroup(0, bgUp[i]); p.dispatchWorkgroupsIndirect(buf.indirect, 0);
      p.setPipeline(pScanHist); p.setBindGroup(0, bgScanHist); p.dispatchWorkgroups(1);
      p.setPipeline(pDown[i]); p.setBindGroup(0, bgDown[i]); p.dispatchWorkgroupsIndirect(buf.indirect, 0);
    }
    p.end();
    p = pass();
    p.setPipeline(pRanges); p.setBindGroup(0, bgRanges); p.dispatchWorkgroupsIndirect(buf.indirect, 12);
    p.end();
    p = pass();
    p.setPipeline(pRaster); p.setBindGroup(0, bgRaster); p.dispatchWorkgroups(tilesX, tilesY);
    p.end();
    if (readback) {
      if (querySet && queryBuf && queryRead) {
        encoder.resolveQuerySet(querySet, 0, 2 * STAGES.length, queryBuf, 0);
        encoder.copyBufferToBuffer(queryBuf, 0, queryRead, 0, 16 * STAGES.length);
      }
      encoder.copyBufferToBuffer(buf.counters, 0, countersRead, 0, 16);
    }
  }

  return {
    width, height, channels, capacity, planes, outputFormat, timestamps: useTimestamps,
    get count() { return count; },
    buffers: { positions: buf.positions, covariances: buf.covariances, opacities: buf.opacities, features: buf.features, image: imageBuf },
    setCount,
    writePositions(data, first = 0) { checkWrite('positions', data.length, 3, first); device.queue.writeBuffer(buf.positions, first * 12, data); },
    writeCovariances(data, first = 0) { checkWrite('covariances', data.length, 6, first); device.queue.writeBuffer(buf.covariances, first * 24, data); },
    writeOpacities(data, first = 0) { checkWrite('opacities', data.length, 1, first); device.queue.writeBuffer(buf.opacities, first * 4, data); },
    writeFeatures(data, first = 0) {
      checkWrite('features', data.length, 2 * words, first);
      // writeBuffer copies whole 4-byte words, which a whole number of Gaussians always is.
      device.queue.writeBuffer(buf.features, first * words * 4, data);
    },
    setCamera({ view, fx, fy, cx = (width - 1) / 2, cy = (height - 1) / 2 }) {
      if (view.length !== 16) throw new Error(`view has ${view.length} values, not 16`);
      pf.set(Array.from(view), 0);
      pf[16] = fx; pf[17] = fy; pf[18] = cx; pf[19] = cy;
      // The projection's Jacobian is clamped at 1.3 times the half field of view.
      pf[28] = width / (2 * fx); pf[29] = height / (2 * fy);
      cameraSet = true;
      writeParams();
    },
    encode,
    render(opts) {
      const enc = device.createCommandEncoder({ label: 'gaussian_rasterizer' });
      encode(enc, opts);
      device.queue.submit([enc.finish()]);
    },
    async readStats() {
      await countersRead.mapAsync(GPUMapMode.READ);
      const c = new Uint32Array(countersRead.getMappedRange().slice(0));
      countersRead.unmap();
      const f = new Float32Array(c.buffer);
      return { pairs: c[0], overflow: c[3], depthMin: f[1], depthMax: f[2] };
    },
    async readTimings() {
      if (!queryRead) return null;
      await queryRead.mapAsync(GPUMapMode.READ);
      const t = new BigUint64Array(queryRead.getMappedRange().slice(0));
      queryRead.unmap();
      const out = {} as Record<Stage | 'total', number>;
      STAGES.forEach((s, i) => { out[s] = Number(t[2 * i + 1] - t[2 * i]) / 1e6; });
      out.total = Number(t[2 * STAGES.length - 1] - t[0]) / 1e6;
      return out;
    },
    destroy() {
      for (const b of Object.values(buf)) b.destroy();
      if (!image) imageBuf.destroy();
      countersRead.destroy();
      queryBuf?.destroy(); queryRead?.destroy(); querySet?.destroy();
    },
  };
}
