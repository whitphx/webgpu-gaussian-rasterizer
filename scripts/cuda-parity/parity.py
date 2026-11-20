"""Compares this library's image with the reference CUDA rasterizer's.

    python parity.py [--module diff_gaussian_rasterization] [--channels 3]

Draws a random scene with the CUDA rasterizer of 3D Gaussian Splatting
(graphdeco-inria/diff-gaussian-rasterization, or a fork compiled for more
channels), renders the same scene with the built library under Dawn, and
prints the PSNR and the largest per-pixel difference.

Needs a Python with numpy, PyTorch on CUDA and the rasterizer module, and
`pnpm build` run first. The rasterizer is not a dependency of this
repository and comes under its own licence.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def random_scene(rng: np.random.Generator, count: int, channels: int):
    positions = rng.uniform(-1.0, 1.0, (count, 3)).astype(np.float32)
    scales = np.exp(rng.uniform(np.log(0.01), np.log(0.12), (count, 3)))
    q = rng.normal(size=(count, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    rotation = np.stack(
        [
            1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
            2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
            2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
        ],
        axis=1,
    ).reshape(count, 3, 3)
    m = rotation * scales[:, None, :]
    sigma = m @ m.transpose(0, 2, 1)
    covariances = sigma[:, [0, 0, 0, 1, 1, 2], [0, 1, 2, 1, 2, 2]].astype(np.float32)
    opacities = rng.uniform(0.05, 0.95, count).astype(np.float32)
    # The library stores features in half precision, so both renderers get
    # the same rounded values.
    features = rng.uniform(0.0, 1.0, (count, channels)).astype(np.float16).astype(np.float32)
    return positions, covariances, opacities, features


def look_at(distance: float) -> np.ndarray:
    """World-to-camera (column vectors, OpenCV axes) looking at the origin
    down +z from (0, 0, -distance)."""
    view = np.eye(4)
    view[2, 3] = distance
    return view


def projection(tan_x: float, tan_y: float, near: float, far: float) -> np.ndarray:
    # getProjectionMatrix in graphdeco-inria/gaussian-splatting's
    # utils/graphics_utils.py.
    p = np.zeros((4, 4))
    p[0, 0] = 1 / tan_x
    p[1, 1] = 1 / tan_y
    p[3, 2] = 1.0
    p[2, 2] = far / (far - near)
    p[2, 3] = -(far * near) / (far - near)
    return p


def render_cuda(module, scene, view, width, height, tan_x, tan_y, background):
    import torch

    positions, covariances, opacities, features = (torch.from_numpy(a).cuda() for a in scene)
    view_t = torch.tensor(view.T, dtype=torch.float32, device="cuda")
    proj_t = torch.tensor((projection(tan_x, tan_y, 0.01, 100.0) @ view).T, dtype=torch.float32, device="cuda")
    fields = dict(
        image_height=height, image_width=width, tanfovx=tan_x, tanfovy=tan_y,
        bg=torch.tensor(background, dtype=torch.float32, device="cuda"), scale_modifier=1.0,
        viewmatrix=view_t, projmatrix=proj_t, sh_degree=0,
        campos=torch.tensor(np.linalg.inv(view)[:3, 3], dtype=torch.float32, device="cuda"),
        prefiltered=False, debug=False, antialiasing=False,
    )
    accepted = module.GaussianRasterizationSettings._fields
    settings = module.GaussianRasterizationSettings(**{k: v for k, v in fields.items() if k in accepted})
    rasterizer = module.GaussianRasterizer(raster_settings=settings)
    with torch.no_grad():
        out = rasterizer(
            means3D=positions, means2D=torch.zeros_like(positions), opacities=opacities[:, None],
            colors_precomp=features, cov3D_precomp=covariances,
        )
    # Forks differ in how many extra outputs (radii, depth) follow the image.
    image = out[0] if isinstance(out, tuple) else out
    return image.float().cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--module", default="diff_gaussian_rasterization", help="the CUDA rasterizer's import name")
    parser.add_argument("--channels", type=int, default=3, help="the channel count the module was compiled for")
    parser.add_argument("--count", type=int, default=20000)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-psnr", type=float, default=60.0, help="exit with 1 below this PSNR")
    args = parser.parse_args()

    module = importlib.import_module(args.module)
    rng = np.random.default_rng(args.seed)
    scene = random_scene(rng, args.count, args.channels)
    view = look_at(3.0)
    fx = fy = 0.9 * args.width
    tan_x, tan_y = args.width / (2 * fx), args.height / (2 * fy)
    background = rng.uniform(0.0, 1.0, args.channels).tolist()
    cuda = render_cuda(module, scene, view, args.width, args.height, tan_x, tan_y, background)

    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        for name, array in zip(["positions", "covariances", "opacities", "features"], scene):
            array.astype("<f4").tofile(d / f"{name}.f32")
        (d / "scene.json").write_text(json.dumps({
            "count": args.count, "channels": args.channels, "width": args.width, "height": args.height,
            # The reference bins each Gaussian into the square around its
            # 3-sigma circle, which tileTest 'box' reproduces.
            "tileTest": "box", "background": background,
            "camera": {"view": view.T.reshape(-1).tolist(), "fx": fx, "fy": fy},
        }))
        run = subprocess.run(["node", str(HERE / "render-webgpu.mjs"), tmp], check=True, capture_output=True, text=True)
        webgpu = np.fromfile(d / "image.f32", dtype="<f4").reshape(args.channels, args.height, args.width)

    diff = np.abs(webgpu - cuda)
    mse = float(np.mean(diff**2))
    psnr = math.inf if mse == 0 else 10 * math.log10(1.0 / mse)
    report = {
        "module": args.module, "channels": args.channels, "count": args.count,
        "webgpu": json.loads(run.stdout.strip().splitlines()[-1]),
        "psnr_db": psnr, "max_abs_diff": float(diff.max()), "mean_abs_diff": float(diff.mean()),
    }
    print(json.dumps(report, indent=1))
    sys.exit(0 if psnr >= args.min_psnr else 1)


if __name__ == "__main__":
    main()
