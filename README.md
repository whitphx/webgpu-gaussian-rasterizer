# webgpu-gaussian-rasterizer

A WebGPU compute rasterizer for 3D Gaussians that composites any number of feature channels per pixel and writes the result into a GPU buffer, ready for a neural network or your own compute pass to read without a round trip through the CPU.

Most web Gaussian-splatting renderers draw RGB(A) into a canvas. This one is for neural rendering, where each Gaussian carries a feature vector (16 or 32 channels, say) and a network turns the composited feature image into the final picture. Positions, covariances, opacities and features can change every frame, so Gaussians bound to an animated mesh work as well as static scenes. `docs/design.md` explains how it compares with the existing libraries.

## Usage

```ts
import { createGaussianRasterizer, createPresenter, packFeatures } from 'webgpu-gaussian-rasterizer';

const rasterizer = await createGaussianRasterizer(device, { capacity: n, width: 512, height: 512, channels: 16 });
rasterizer.writePositions(positions);     // Float32Array, x y z per Gaussian
rasterizer.writeCovariances(covariances); // Float32Array, xx xy xz yy yz zz
rasterizer.writeOpacities(opacities);     // Float32Array, after the sigmoid
rasterizer.writeFeatures(packFeatures(features, 16));
rasterizer.setCamera({ view, fx: 600, fy: 600 });

const encoder = device.createCommandEncoder();
rasterizer.encode(encoder);
// rasterizer.buffers.image now holds 16 planes of 512 x 512 float16 values (NCHW).
device.queue.submit([encoder.finish()]);
```

To animate, write new positions or covariances before each frame, or bind `rasterizer.buffers.positions` in a compute pass of your own and write them on the GPU.

To feed a network, bind the image buffer as the input of an ONNX Runtime Web session on the same device (`ort.Tensor.fromGpuBuffer(rasterizer.buffers.image, { dataType: 'float16', dims: [1, 16, 512, 512] })`). Alternatively, pass a buffer the session already holds as the `image` option, and the rasterizer renders straight into it. That keeps a captured graph's bindings valid when the rasterizer is recreated.

To look at the result, `createPresenter` draws three planes of a planar buffer into a canvas texture.

### Conventions

- `view` is the world-to-camera matrix in column-major order for column vectors, the layout of WGSL's `mat4x4f`. Camera space follows OpenCV: x right, y down, z forward. Pixel (0, 0)'s centre is at (0, 0), so the default principal point is `((width - 1) / 2, (height - 1) / 2)`.
- Covariances are 3D world-space covariances. `covariancesFromScaleRotation` builds them from scales and (w, x, y, z) quaternions. Pass `normalize: false` for models trained against a rasterizer that uses the quaternion as given.
- Features are stored in half precision. The output is half or single precision (`outputFormat`); half precision needs an even width.
- Compositing follows the reference CUDA implementation of 3D Gaussian Splatting: 0.3 px of dilation on the 2D covariance, alpha capped at 0.99, contributions under 1/255 skipped, and a pixel stops once its transmittance would fall under 1e-4. With `tileTest: 'box'` the binning is the reference's 3-sigma square, so images match it to rounding. The default `'ellipse'` skips tiles a Gaussian cannot reach above 1/255, which gives the same image as compositing every Gaussian at every pixel and usually halves the work.

## Example

```bash
pnpm install
pnpm dev
```

Open the printed URL in a browser with WebGPU. The page animates 40,000 Gaussians with 8 feature channels and decodes them to colour in a compute pass of its own.

## Tests

```bash
pnpm test
```

The GPU tests run under Node through Dawn (the `webgpu` package) and compare every configuration against a per-pixel CPU rasterizer in `test/reference.ts`. Dawn needs a Vulkan driver on Linux. Without a usable adapter those tests are skipped and only the CPU tests run.

## Licence

MIT. The radix sort's structure follows PlayCanvas's, and the tile test is patterned after Brush's; `THIRD_PARTY_NOTICES.md` credits both and carries their licences.
