# Design notes

## Why another Gaussian rasterizer

The rasterizer exists for neural rendering in the browser. Here, each Gaussian carries a learned feature vector rather than a colour, and a network decodes the composited feature image. Avatar models in the GAGAvatar family work this way, with 32 feature channels per Gaussian and Gaussians that move with a face mesh every frame. Such a renderer needs four things:

1. It composites N generic channels, not RGB or spherical harmonics.
2. Positions and covariances change every frame.
3. The image lands in a GPU buffer, in the planar layout a network takes as input, with no readback.
4. It matches the CUDA rasterizer the network was trained against, closely enough that the network does not notice the difference.

### Survey (November 2026)

None of the web libraries we surveyed does all four. Links are pinned to the versions read.

| Library | GPU path | Channels | Per-frame geometry | Output | Licence |
|---|---|---|---|---|---|
| [PlayCanvas engine](https://github.com/playcanvas/engine/tree/v2.24.2) 2.24 | WebGPU compute sort, then instanced quads (its compute tile renderer was [removed](https://github.com/playcanvas/engine/pull/8921) in 2.20) | RGBA; the fragment stage writes one vec4 | yes (work-buffer modifiers) | engine render target | MIT |
| [Spark](https://github.com/sparkjsdev/spark/tree/v2.4.0) 2.4 | WebGL2, CPU sort in a worker | RGBA | yes (shader graph, skinning) | three.js render target | MIT |
| [Spark Radix](https://github.com/sparkjsdev/radix/tree/a9963abf8a333a3ff8d1580c0271274b6d909ff5) | WebGPU through three.js TSL, quads | at most 4 per projected splat | yes | three.js render target | MIT |
| [luma.gl splats](https://github.com/visgl/luma.gl/tree/94e95a6ab4e5981d4d15e54cca540febd13f2775/modules/splats) 9.4 | WebGPU or WebGL2, GPU sort, quads | RGBA and SH | CPU updates | the caller's render pass | MIT |
| [Babylon.js](https://github.com/BabylonJS/Babylon.js/tree/9.37.1) 9.37 | quads; optional experimental WebGPU sort | SH up to degree 3 | whole-buffer re-upload | render target texture | Apache-2.0 |
| [Visionary](https://github.com/Visionary-Laboratory/visionary/tree/76a2b93cec0e0dfaeb598fd320b6ceabf4b39d48) | WebGPU compute preprocess and sort, quads | 3, 12, 27 or 48 colour coefficients | yes, from ONNX outputs on the GPU | a texture view | Apache-2.0 |
| [GaussianSplats3D](https://github.com/mkkellogg/GaussianSplats3D/tree/eb2fc4593e3ea5e75388296fcdde2459542d1290), [gsplat.js](https://github.com/huggingface/gsplat.js/tree/dbe93f6dd3de81392c829dd82704c03bc1c1a880), [antimatter15/splat](https://github.com/antimatter15/splat/tree/ba182b51b7c2ad5738cdd6741cd63336d27470fb) | WebGL, CPU or WASM sort | RGB or SH | whole-scene transforms only | canvas | MIT |
| [Brush](https://github.com/ArthurBrussee/brush/tree/1388f74c6fe0236f68ee4915564bf00e9d2e3747) | wgpu compute tile rasterizer (Rust) | RGB | training, not animation | its own app; the JS binding trains only | Apache-2.0 |
| [nerfstudio gsplat](https://github.com/nerfstudio-project/gsplat/blob/512d366b67073d77ca099ede742683c165dfc23b/gsplat/rendering.py) | CUDA only | N channels | yes | PyTorch tensors | Apache-2.0 |

Nerfstudio's gsplat composites N channels, but only in CUDA. Visionary is the closest web project in data flow, since it feeds ONNX outputs into a splatter on the GPU, but it composites colour only and draws into a texture. Contributing an N-channel compute path upstream was considered. PlayCanvas, the most likely host, has just removed its compute tile renderer as experimental, and its forward path writes a single vec4. Luma.gl draws through the hardware rasterizer and would need a second renderer added beside the first.

### What this library adds

- **N feature channels in one pass.** 3, 16 and 32 channels are tested, and any count works. Features are half precision and unpacked in the kernel, so `shader-f16` is not required.
- **Output into a GPU buffer the caller can own.** The planar (NCHW) half or single precision image is laid out the way ONNX Runtime Web's WebGPU execution provider binds an input tensor.
- **A compute tile rasterizer that follows the CUDA reference's compositing,** rather than sorted quads through the hardware rasterizer. A network trained on the CUDA output sees the same image.
- **No engine.** It takes a bare `GPUDevice` and records into the caller's command encoder, so it shares a device and a frame loop with ONNX Runtime Web or anything else.

## Pipeline

One frame is six compute passes, 33 dispatches in all, with no CPU round trip between them:

1. **project** transforms each Gaussian, builds its 2D conic, picks the tiles it touches and tracks the frame's depth range.
2. **scan** takes the prefix sum of the per-Gaussian tile counts. It writes the pair count and the indirect dispatch sizes that the later passes read.
3. **emit** writes one 32-bit key per (tile, Gaussian) pair: the tile index above the depth, quantized over the frame's range into at most 24 bits.
4. **sort** is an LSD radix sort, eight passes of 4 bits each. Its structure follows PlayCanvas's portable multipass sort.
5. **ranges** finds where each tile's run of keys starts and ends.
6. **rasterize** runs one 16 x 16 workgroup per tile. It stages 256 Gaussians at a time in workgroup memory, accumulates N channels per pixel, and stops early once every pixel of the tile is opaque.

The `'ellipse'` tile test bounds each Gaussian by where its alpha can reach 1/255, ln(255 * opacity) in Mahalanobis terms (after Brush), and tests each tile's pixel-centre rectangle against that ellipse exactly. Pixels outside the ellipse would be skipped by the compositor anyway, so the image does not change. On avatar scenes this halves the pairs, which is most of the cost.

## Accuracy

The tests in `test/` compare every configuration against a per-pixel CPU rasterizer with the same conventions and no tiles. Single-precision output agrees above 100 dB PSNR, and the box tile test, which truncates at 3 sigma as the CUDA reference does, stays above 45 dB.

Against the CUDA rasterizer itself (`diff-gaussian-rasterization`), the kernels were measured before they were extracted into this library. The test set was four GAGAvatar avatars of 75k to 180k Gaussians, five frames each, at 512 px. With 16 channels the PSNR was 92 to 105 dB, and with 32 channels 92 to 98 dB. The residual is the half-precision output. The same frame with single-precision output reads 124 dB, and the pair counts agree to within one pair per frame. The extraction did not change the arithmetic at those image sizes. A script that reproduces the comparison is not part of this repository yet.

## Speed

Measured before extraction, on an NVIDIA P100 through Dawn and Vulkan: 75k to 123k Gaussians, 16 channels, 512 px, ellipse tiles and half-precision output took 1.9 to 3.3 ms of GPU time per frame. The CUDA rasterizer took 3.8 to 6.3 ms on the same frames, though it always composites 32 channels. The sort and the rasterize pass account for nearly all of the time. Sixteen channels cost about 1.3 times the rasterize time of three. On small GPUs the 33 dispatches also cost CPU time, which a browser's per-dispatch overhead makes visible.

## Choices in the API

- **Avatar-specific deformation stays out.** A model that turns a head on its neck, or skins Gaussians to bones, writes the positions and covariances, either from the CPU or in its own compute pass on `buffers.positions` and `buffers.covariances` before `encode`. This keeps the projection exactly the reference's when no deformation is applied.
- **`encode` records into the caller's encoder.** It resets its counters inside the encoder, so several rasterizers, or a rasterizer and a network, can share one submit. The `write*`, `setCamera` and `setCount` calls go through the queue and apply from the next submit.
- **A caller-owned `image` buffer.** ONNX Runtime Web binds a captured graph to fixed input buffers. Letting the rasterizer render into the session's buffer avoids a copy, and avoids recapturing when the rasterizer is recreated, for example on an avatar switch.
- **Capacity and count.** Buffers are sized for `capacity` Gaussians. `setCount` draws a prefix of them, so a scene can grow and shrink without reallocation. The pair buffer is sized by `pairCapacity`. When a frame needs more pairs, `readStats` reports the overflow, and tiles lose the Gaussians that did not fit.

## Limits and next steps

- The sort uses eight 4-bit passes with no subgroup operations, so it is portable to every WebGPU implementation but not the fastest where subgroups exist.
- Depth is quantized per frame into at most 24 bits over the frame's depth range. Scenes much deeper than they are detailed may need a finer key.
- There are no gradients. The library renders and does not train.
- A browser benchmark page and a CUDA comparison script are planned.
