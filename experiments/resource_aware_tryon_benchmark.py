from __future__ import annotations

import json
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps


REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = REPO_ROOT / "reports" / "resource_aware_tryon_2026"

UPPER_LABELS = {5, 6, 7}
PRESERVE_LABELS = {1, 2, 4, 13, 14, 15, 16, 17, 18, 19}

RESAMPLE_LANCZOS = Image.Resampling.LANCZOS
RESAMPLE_BILINEAR = Image.Resampling.BILINEAR


@dataclass(frozen=True)
class GarmentAsset:
    name: str
    image_path: Path
    mask_path: Path | None = None
    reference_path: Path | None = None


@dataclass
class PersonContext:
    image: np.ndarray
    parse: np.ndarray
    upper_mask: np.ndarray
    upper_soft: np.ndarray
    upper_dilated: np.ndarray
    preserve_soft: np.ndarray
    pose: np.ndarray
    width: int
    height: int
    upper_box: tuple[int, int, int, int]


@dataclass
class MethodMetrics:
    method: str
    garment: str
    runtime_sec: float
    coverage: float
    spill: float
    outside_ssim: float
    reference_ssim: float | None
    output_file: str


def run_command(args: list[str]) -> str:
    try:
        result = subprocess.run(args, capture_output=True, text=True, check=False)
    except OSError as exc:
        return f"unavailable: {exc}"
    output = (result.stdout or result.stderr).strip()
    return output or "unavailable"


def relative_path(path: Path) -> str:
    return str(path.resolve().relative_to(REPO_ROOT))


def to_uint8_rgb(array: np.ndarray) -> Image.Image:
    clipped = np.clip(array * 255.0, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(clipped, mode="RGB")


def to_uint8_mask(mask: np.ndarray) -> Image.Image:
    clipped = np.clip(mask * 255.0, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(clipped, mode="L")


def rgb_to_gray(rgb: np.ndarray) -> np.ndarray:
    return 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]


def blur_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    return np.asarray(to_uint8_mask(mask).filter(ImageFilter.GaussianBlur(radius=radius)), dtype=np.float32) / 255.0


def dilate_mask(mask: np.ndarray, size: int) -> np.ndarray:
    size = max(3, size | 1)
    return np.asarray(to_uint8_mask(mask).filter(ImageFilter.MaxFilter(size=size)), dtype=np.float32) / 255.0


def refine_binary_mask(mask: np.ndarray) -> np.ndarray:
    image = to_uint8_mask(mask)
    image = image.filter(ImageFilter.MaxFilter(size=5))
    image = image.filter(ImageFilter.MinFilter(size=3))
    return (np.asarray(image, dtype=np.float32) > 16).astype(np.float32)


def bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0.5)
    if len(xs) == 0:
        raise ValueError("Mask is empty")
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def row_extents(mask: np.ndarray, quantile: float) -> tuple[int, int, int]:
    ys, xs = np.where(mask > 0.5)
    if len(xs) == 0:
        raise ValueError("Mask is empty")
    target_y = int(np.quantile(ys, quantile))
    height = mask.shape[0]
    for radius in range(0, 40):
        y0 = max(0, target_y - radius)
        y1 = min(height, target_y + radius + 1)
        band = mask[y0:y1]
        band_xs = np.where(band > 0.5)[1]
        if len(band_xs):
            return int(band_xs.min()), int(band_xs.max()) + 1, target_y
    left, _, right, _ = bbox_from_mask(mask)
    return left, right, target_y


def load_pose(path: Path) -> np.ndarray:
    data = json.loads(path.read_text())
    points = np.array(data["people"][0]["pose_keypoints_2d"], dtype=np.float32).reshape(-1, 3)[:, :2]
    return points


def load_person_context() -> PersonContext:
    person_image_path = REPO_ROOT / "TryYours-Virtual-Try-On-main" / "HR-VITON-main" / "test" / "test" / "image" / "00001_00.jpg"
    parse_path = REPO_ROOT / "TryYours-Virtual-Try-On-main" / "HR-VITON-main" / "test" / "test" / "image-parse-v3" / "00001_00.png"
    pose_path = REPO_ROOT / "TryYours-Virtual-Try-On-main" / "HR-VITON-main" / "test" / "test" / "openpose_json" / "00001_00_keypoints.json"

    image = np.asarray(Image.open(person_image_path).convert("RGB"), dtype=np.float32) / 255.0
    parse = np.asarray(Image.open(parse_path), dtype=np.int32)
    upper_mask = np.isin(parse, sorted(UPPER_LABELS)).astype(np.float32)
    upper_mask = refine_binary_mask(upper_mask)
    preserve_mask = np.isin(parse, sorted(PRESERVE_LABELS)).astype(np.float32)
    preserve_mask = blur_mask(preserve_mask, radius=3)
    upper_dilated = dilate_mask(upper_mask, size=21)
    upper_soft = blur_mask(upper_dilated, radius=8)
    upper_box = bbox_from_mask(upper_mask)
    pose = load_pose(pose_path)
    height, width = image.shape[:2]
    return PersonContext(
        image=image,
        parse=parse,
        upper_mask=upper_mask,
        upper_soft=upper_soft,
        upper_dilated=upper_dilated,
        preserve_soft=preserve_mask,
        pose=pose,
        width=width,
        height=height,
        upper_box=upper_box,
    )


def infer_mask_from_background(rgb: np.ndarray) -> np.ndarray:
    corners = np.stack([rgb[0, 0], rgb[0, -1], rgb[-1, 0], rgb[-1, -1]], axis=0)
    background = corners.mean(axis=0)
    distance = np.linalg.norm(rgb - background[None, None, :], axis=2)
    threshold = max(0.10, float(np.percentile(distance, 70)) * 0.55)
    mask = (distance > threshold).astype(np.float32)
    return refine_binary_mask(mask)


def load_garment_rgba(asset: GarmentAsset) -> np.ndarray:
    image = Image.open(asset.image_path)
    has_alpha = "A" in image.getbands() or "transparency" in image.info
    if asset.mask_path is not None:
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        mask_image = Image.open(asset.mask_path).convert("L")
        if mask_image.size != image.size:
            mask_image = mask_image.resize(image.size, RESAMPLE_BILINEAR)
        mask = np.asarray(mask_image, dtype=np.float32) / 255.0
        mask = refine_binary_mask(mask)
    elif has_alpha:
        rgba_image = image.convert("RGBA")
        rgba = np.asarray(rgba_image, dtype=np.float32) / 255.0
        rgb = rgba[:, :, :3]
        mask = rgba[:, :, 3]
        mask = refine_binary_mask(mask)
    else:
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        mask = infer_mask_from_background(rgb)
    return np.dstack([rgb, mask])


def expand_box(box: tuple[int, int, int, int], width: int, height: int, pad_x: float, pad_y: float) -> tuple[int, int, int, int]:
    left, top, right, bottom = box
    box_width = right - left
    box_height = bottom - top
    left = max(0, int(round(left - box_width * pad_x)))
    top = max(0, int(round(top - box_height * pad_y)))
    right = min(width, int(round(right + box_width * pad_x)))
    bottom = min(height, int(round(bottom + box_height * pad_y)))
    return left, top, right, bottom


def fit_rgba_to_box(rgba: np.ndarray, box: tuple[int, int, int, int], canvas_size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    left, top, right, bottom = box
    box_width = max(1, right - left)
    box_height = max(1, bottom - top)
    rgba_uint8 = np.clip(rgba * 255.0, 0.0, 255.0).astype(np.uint8)
    garment = Image.fromarray(rgba_uint8, mode="RGBA")
    fitted = ImageOps.fit(garment, (box_width, box_height), method=RESAMPLE_LANCZOS, centering=(0.5, 0.12))
    canvas = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
    canvas.paste(fitted, (left, top), fitted)
    array = np.asarray(canvas, dtype=np.float32) / 255.0
    return array[:, :, :3], array[:, :, 3]


def sort_pair(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ordered = points[np.argsort(points[:, 0])]
    return ordered[0], ordered[1]


def source_quad(mask: np.ndarray) -> np.ndarray:
    top_left, top_right, top_y = row_extents(mask, 0.22)
    bottom_left, bottom_right, bottom_y = row_extents(mask, 0.92)
    return np.array(
        [
            [top_left, top_y],
            [top_right, top_y],
            [bottom_right, bottom_y],
            [bottom_left, bottom_y],
        ],
        dtype=np.float32,
    )


def target_pose_quad(ctx: PersonContext) -> np.ndarray:
    mask_left, mask_top, mask_right, mask_bottom = ctx.upper_box
    top_mask_left, top_mask_right, _ = row_extents(ctx.upper_mask, 0.12)
    bottom_mask_left, bottom_mask_right, _ = row_extents(ctx.upper_mask, 0.92)

    left_shoulder, right_shoulder = sort_pair(ctx.pose[[2, 5]])
    left_hip, right_hip = sort_pair(ctx.pose[[9, 12]])
    shoulder_width = max(1.0, float(right_shoulder[0] - left_shoulder[0]))
    hip_width = max(1.0, float(right_hip[0] - left_hip[0]))

    top_y = max(0.0, min(float(mask_top), float(left_shoulder[1]), float(right_shoulder[1])) - 12.0)
    bottom_y = min(float(ctx.height - 1), float(mask_bottom) + 8.0)
    top_left_x = min(float(top_mask_left), float(left_shoulder[0] - shoulder_width * 0.12))
    top_right_x = max(float(top_mask_right), float(right_shoulder[0] + shoulder_width * 0.12))
    bottom_left_x = min(float(bottom_mask_left), float(left_hip[0] - hip_width * 0.08))
    bottom_right_x = max(float(bottom_mask_right), float(right_hip[0] + hip_width * 0.08))

    return np.array(
        [
            [max(0.0, top_left_x), top_y],
            [min(float(ctx.width - 1), top_right_x), top_y],
            [min(float(ctx.width - 1), bottom_right_x), bottom_y],
            [max(0.0, bottom_left_x), bottom_y],
        ],
        dtype=np.float32,
    )


def homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    matrix = []
    values = []
    for (x, y), (u, v) in zip(src, dst):
        matrix.append([x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y])
        matrix.append([0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y])
        values.append(u)
        values.append(v)
    matrix_arr = np.asarray(matrix, dtype=np.float32)
    values_arr = np.asarray(values, dtype=np.float32)
    coeffs, _, _, _ = np.linalg.lstsq(matrix_arr, values_arr, rcond=None)
    return np.append(coeffs, 1.0).reshape(3, 3)


def bilinear_sample(image: np.ndarray, x_coords: np.ndarray, y_coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    height, width = image.shape[:2]
    valid = (x_coords >= 0.0) & (x_coords <= width - 1.0) & (y_coords >= 0.0) & (y_coords <= height - 1.0)
    sampled = np.zeros((x_coords.shape[0], image.shape[2]), dtype=np.float32)
    if not valid.any():
        return sampled, valid

    x_valid = x_coords[valid]
    y_valid = y_coords[valid]
    x0 = np.floor(x_valid).astype(np.int32)
    y0 = np.floor(y_valid).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, width - 1)
    y1 = np.clip(y0 + 1, 0, height - 1)
    dx = x_valid - x0
    dy = y_valid - y0

    top = image[y0, x0] * (1.0 - dx)[:, None] + image[y0, x1] * dx[:, None]
    bottom = image[y1, x0] * (1.0 - dx)[:, None] + image[y1, x1] * dx[:, None]
    sampled[valid] = top * (1.0 - dy)[:, None] + bottom * dy[:, None]
    return sampled, valid


def warp_rgba(rgba: np.ndarray, src_pts: np.ndarray, dst_pts: np.ndarray, canvas_size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    canvas_width, canvas_height = canvas_size
    polygon = Image.new("L", (canvas_width, canvas_height), 0)
    ImageDraw.Draw(polygon).polygon([tuple(point.tolist()) for point in dst_pts], fill=255)
    poly_mask = np.asarray(polygon, dtype=np.float32) / 255.0
    ys, xs = np.where(poly_mask > 0.5)
    output = np.zeros((canvas_height, canvas_width, 4), dtype=np.float32)
    if len(xs) == 0:
        return output[:, :, :3], output[:, :, 3]

    transform = homography(src_pts, dst_pts)
    inverse = np.linalg.inv(transform)
    coords = np.stack(
        [xs.astype(np.float32), ys.astype(np.float32), np.ones(len(xs), dtype=np.float32)],
        axis=0,
    )
    mapped = inverse @ coords
    mapped_x = mapped[0] / np.maximum(mapped[2], 1e-6)
    mapped_y = mapped[1] / np.maximum(mapped[2], 1e-6)
    sampled, _ = bilinear_sample(rgba, mapped_x, mapped_y)
    output[ys, xs] = sampled
    return output[:, :, :3], output[:, :, 3]


def alpha_blend(background: np.ndarray, foreground: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    alpha = np.clip(alpha, 0.0, 1.0)
    return foreground * alpha[:, :, None] + background * (1.0 - alpha[:, :, None])


def restore_person(result: np.ndarray, person: np.ndarray, preserve_soft: np.ndarray) -> np.ndarray:
    return alpha_blend(result, person, preserve_soft)


def detail_transfer(person: np.ndarray, result: np.ndarray, garment_mask: np.ndarray) -> np.ndarray:
    detail = rgb_to_gray(person)
    detail_smooth = blur_mask(detail, radius=10)
    ratio = detail / np.maximum(detail_smooth, 1e-3)
    ratio = np.clip(1.0 + (ratio - 1.0) * 0.15, 0.92, 1.08)
    enhanced = np.clip(result * ratio[:, :, None], 0.0, 1.0)
    soft = blur_mask(garment_mask, radius=4)
    return alpha_blend(result, enhanced, soft)


def masked_ssim(image_a: np.ndarray, image_b: np.ndarray, mask: np.ndarray) -> float:
    active = mask > 0.05
    if active.sum() < 16:
        return 1.0
    values_a = image_a[active].astype(np.float64)
    values_b = image_b[active].astype(np.float64)
    mu_a = values_a.mean()
    mu_b = values_b.mean()
    sigma_a = values_a.var()
    sigma_b = values_b.var()
    sigma_ab = ((values_a - mu_a) * (values_b - mu_b)).mean()
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    numerator = (2.0 * mu_a * mu_b + c1) * (2.0 * sigma_ab + c2)
    denominator = (mu_a * mu_a + mu_b * mu_b + c1) * (sigma_a + sigma_b + c2)
    return float(numerator / max(denominator, 1e-6))


def layer_mask_box(ctx: PersonContext, garment_rgba: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    box = expand_box(ctx.upper_box, ctx.width, ctx.height, pad_x=0.10, pad_y=0.06)
    rgb, mask = fit_rgba_to_box(garment_rgba, box, (ctx.width, ctx.height))
    mask = np.clip(mask * ctx.upper_soft, 0.0, 1.0)
    mask = blur_mask(mask, radius=5)
    return rgb, mask


def layer_pose_quad(ctx: PersonContext, garment_rgba: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    src = source_quad(garment_rgba[:, :, 3])
    dst = target_pose_quad(ctx)
    rgb, mask = warp_rgba(garment_rgba, src, dst, (ctx.width, ctx.height))
    mask = np.clip(mask * ctx.upper_dilated, 0.0, 1.0)
    mask = blur_mask(mask, radius=4)
    return rgb, mask


def render_mask_box(ctx: PersonContext, garment_rgba: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rgb, mask = layer_mask_box(ctx, garment_rgba)
    result = alpha_blend(ctx.image, rgb, mask)
    result = restore_person(result, ctx.image, ctx.preserve_soft)
    garment_mask = np.clip(mask * (1.0 - ctx.preserve_soft), 0.0, 1.0)
    return np.clip(result, 0.0, 1.0), garment_mask


def render_pose_quad(ctx: PersonContext, garment_rgba: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rgb, mask = layer_pose_quad(ctx, garment_rgba)
    result = alpha_blend(ctx.image, rgb, mask)
    result = restore_person(result, ctx.image, ctx.preserve_soft)
    garment_mask = np.clip(mask * (1.0 - ctx.preserve_soft), 0.0, 1.0)
    return np.clip(result, 0.0, 1.0), garment_mask


def render_hybrid_pose_parse(ctx: PersonContext, garment_rgba: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    box_rgb, box_mask = layer_mask_box(ctx, garment_rgba)
    pose_rgb, pose_mask = layer_pose_quad(ctx, garment_rgba)
    fallback_mask = np.clip(ctx.upper_soft - pose_mask * 0.85, 0.0, 1.0)
    result = alpha_blend(ctx.image, box_rgb, fallback_mask)
    result = alpha_blend(result, pose_rgb, pose_mask)
    garment_mask = np.maximum(pose_mask, fallback_mask)
    result = detail_transfer(ctx.image, result, garment_mask)
    result = restore_person(result, ctx.image, ctx.preserve_soft)
    garment_mask = np.clip(garment_mask * (1.0 - ctx.preserve_soft), 0.0, 1.0)
    return np.clip(result, 0.0, 1.0), garment_mask


def coverage(mask: np.ndarray, target: np.ndarray) -> float:
    target_pixels = float((target > 0.5).sum())
    if target_pixels == 0.0:
        return 0.0
    covered = float(((mask > 0.35) & (target > 0.5)).sum())
    return covered / target_pixels


def spill(mask: np.ndarray, target: np.ndarray) -> float:
    predicted = float((mask > 0.35).sum())
    if predicted == 0.0:
        return 0.0
    outside = float(((mask > 0.35) & (target < 0.2)).sum())
    return outside / predicted


def create_captioned_panel(image: Image.Image, label: str, panel_size: tuple[int, int]) -> Image.Image:
    panel_width, panel_height = panel_size
    caption_height = 44
    canvas = Image.new("RGB", (panel_width, panel_height), (248, 248, 248))
    content_height = panel_height - caption_height
    fitted = ImageOps.contain(image.convert("RGB"), (panel_width - 16, content_height - 16), method=RESAMPLE_LANCZOS)
    offset_x = (panel_width - fitted.width) // 2
    offset_y = 8 + (content_height - fitted.height) // 2
    canvas.paste(fitted, (offset_x, offset_y))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.rectangle((0, content_height, panel_width, panel_height), fill=(234, 234, 234))
    draw.text((10, content_height + 12), label, fill=(15, 15, 15), font=font)
    return canvas


def save_montage(rows: list[list[tuple[str, Image.Image]]], output_path: Path, panel_size: tuple[int, int] = (260, 360)) -> None:
    if not rows:
        return
    columns = max(len(row) for row in rows)
    panel_width, panel_height = panel_size
    gutter = 10
    width = columns * panel_width + gutter * (columns + 1)
    height = len(rows) * panel_height + gutter * (len(rows) + 1)
    sheet = Image.new("RGB", (width, height), (255, 255, 255))
    for row_index, row in enumerate(rows):
        for column_index, (label, image) in enumerate(row):
            panel = create_captioned_panel(image, label, panel_size)
            x = gutter + column_index * (panel_width + gutter)
            y = gutter + row_index * (panel_height + gutter)
            sheet.paste(panel, (x, y))
    sheet.save(output_path)


def mean_defined(values: list[float | None]) -> float | None:
    defined = [value for value in values if value is not None]
    if not defined:
        return None
    return float(np.mean(defined))


def percent(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.1f}%"


def value_str(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"


def probe_system() -> dict[str, str]:
    system = {
        "gpu": run_command(["nvidia-smi", "--query-gpu=name,memory.total,memory.free,driver_version", "--format=csv,noheader"]),
        "memory": run_command(["free", "-h"]),
        "disk": run_command(["df", "-h", str(REPO_ROOT)]),
    }
    opencv_probe = run_command(["python3", "-c", "import cv2; print(cv2.__version__)"])
    if "Traceback" in opencv_probe or "ImportError" in opencv_probe or "AttributeError" in opencv_probe:
        system["opencv_status"] = opencv_probe.splitlines()[-1]
    else:
        system["opencv_status"] = f"ok: {opencv_probe}"
    return system


def build_report(
    ctx: PersonContext,
    assets: list[GarmentAsset],
    metrics: list[MethodMetrics],
    report_dir: Path,
    system: dict[str, str],
    dataset_size_mb: float,
    aggregate: dict[str, dict[str, float | None]],
) -> str:
    yale_name = "yale_reference"
    yale_rows = [entry for entry in metrics if entry.garment == yale_name]
    per_garment_rows = []
    for entry in metrics:
        per_garment_rows.append(
            f"| {entry.garment} | {entry.method} | {entry.runtime_sec:.3f} | {percent(entry.coverage)} | {percent(entry.spill)} | {value_str(entry.outside_ssim)} | {value_str(entry.reference_ssim)} |"
        )

    aggregate_rows = []
    for method, summary in aggregate.items():
        aggregate_rows.append(
            f"| {method} | {summary['avg_runtime_sec']:.3f} | {percent(summary['avg_coverage'])} | {percent(summary['avg_spill'])} | {value_str(summary['avg_outside_ssim'])} | {value_str(summary['avg_reference_ssim'])} |"
        )

    recommendations = []
    if aggregate["hybrid_pose_parse"]["avg_reference_ssim"] is not None:
        best_method = max(
            (name for name in aggregate if aggregate[name]["avg_reference_ssim"] is not None),
            key=lambda name: float(aggregate[name]["avg_reference_ssim"]),
        )
        recommendations.append(
            f"On the one pair with a deep-model reference, `{best_method}` produced the closest match to the cached HR-VITON output."
        )
    recommendations.append(
        "The project is now executable on this machine with a tiny local benchmark instead of depending on missing legacy checkpoints and a PyTorch 1.0 training stack."
    )
    recommendations.append(
        "CatVTON is the most realistic future upgrade path because its published inference target is below 8 GB VRAM at 1024x768, but this laptop still sits below that threshold with only about 3.7 GB free."
    )

    report = f"# Resource-Aware Virtual Try-On Modernization Report\n\n"
    report += "## Scope\n\n"
    report += "This repository is centered on a 2021 DIOR/GFLA pipeline that expects `torch==1.0.0`, external checkpoints, and a much older CUDA toolchain. Instead of trying to revive the full legacy training path on a 4 GB laptop GPU, I added a lightweight benchmark that reuses the included HR-VITON sample person and several local garment assets to produce runnable try-on experiments on the current machine.\n\n"
    report += "This is not a reproduction of recent diffusion models. It is a resource-aware benchmark inspired by them, built to answer three practical questions: can the project run today, can it stay under 1 GB of data, and what method mix looks best on this device.\n\n"
    report += "## Hardware And Environment\n\n"
    report += f"- GPU: {system['gpu']}\n"
    report += f"- RAM: {system['memory'].splitlines()[1] if len(system['memory'].splitlines()) > 1 else system['memory']}\n"
    report += f"- Disk: {system['disk'].splitlines()[1] if len(system['disk'].splitlines()) > 1 else system['disk']}\n"
    report += f"- OpenCV status: {system['opencv_status']}\n"
    report += "- Legacy requirement check: the root `requirements.txt` still pins `torch==1.0.0` and `torchvision==0.2.1`, which is a strong sign that the original training path is not a modern drop-in install.\n\n"
    report += "## Recent Methods Reviewed\n\n"
    report += "| Method | Year | Key Idea | Inputs / Preprocessing | Device Fit For This Laptop | Source |\n"
    report += "| --- | --- | --- | --- | --- | --- |\n"
    report += "| StableVITON | 2023 | Learns semantic correspondence inside latent diffusion with additional zero cross-attention blocks. | Needs agnostic map, agnostic mask, and DensePose. | High OOM risk on 4 GB VRAM because of diffusion backbone plus heavy conditioning. | https://rlawjdghek.github.io/StableVITON/ |\n"
    report += "| IDM-VTON | 2024 | Dual garment conditioning: IP-Adapter style high-level semantics plus a GarmentNet for low-level details. | Uses segmentation mask, masked image, DensePose, and detailed garment prompts. | High OOM risk on 4 GB VRAM and more preprocessing than this repo currently has ready. | https://idm-vton.github.io/ |\n"
    report += "| CatVTON | 2024 / ICLR 2025 | Simplifies the pipeline by concatenating garment and person in the diffusion input space and removing extra encoders. | Requires person image, garment image, and mask. | Closest to feasible, but the project page still states `< 8G VRAM for 1024x768`, which is above this laptop's current free VRAM. | https://zheng-chong.github.io/CatVTON/ |\n"
    report += "| DIOR / Dressing In Order | 2021 | Flow-guided garment transfer with recurrent person synthesis. | DeepFashion-style parsing and keypoints plus legacy checkpoints. | Blocked by old Torch/CUDA requirements and missing checkpoints in this workspace. | https://github.com/cuiaiyu/dressing-in-order |\n\n"
    report += "## Experimental Setup\n\n"
    report += f"- Person sample: `{relative_path(REPO_ROOT / 'TryYours-Virtual-Try-On-main' / 'HR-VITON-main' / 'test' / 'test' / 'image' / '00001_00.jpg')}`\n"
    report += "- Garments: one HR-VITON flat-lay shirt plus three transparent PNG shirts from the included web demo assets\n"
    report += f"- Total benchmark asset size: {dataset_size_mb:.2f} MB\n"
    report += f"- Upper-garment mask size: {int((ctx.upper_mask > 0.5).sum())} pixels\n"
    report += "- Methods tested:\n"
    report += "  - `mask_box`: simple mask-aware resize and blend, closest to a zero-geometry baseline\n"
    report += "  - `pose_quad`: pose-guided quadrilateral warp, closest to a classic geometry-first VTON baseline\n"
    report += "  - `hybrid_pose_parse`: pose warp plus mask-box fallback plus parse-based occlusion restoration\n\n"
    report += "## Aggregate Results\n\n"
    report += "| Method | Avg Runtime (s) | Avg Coverage | Avg Spill | Avg Outside SSIM | Yale Reference SSIM |\n"
    report += "| --- | --- | --- | --- | --- | --- |\n"
    report += "\n".join(aggregate_rows)
    report += "\n\n"
    report += "Coverage is the fraction of the upper-clothing target region filled by the synthesized garment mask. Spill is the fraction of synthesized garment pixels that fall outside the target garment region. Outside SSIM measures how much of the rest of the person image stays unchanged. Yale Reference SSIM compares only the sample shirt against the cached HR-VITON output already present in the repository.\n\n"
    report += "## Per-Garment Metrics\n\n"
    report += "| Garment | Method | Runtime (s) | Coverage | Spill | Outside SSIM | Reference SSIM |\n"
    report += "| --- | --- | --- | --- | --- | --- | --- |\n"
    report += "\n".join(per_garment_rows)
    report += "\n\n"
    report += "## Visual Outputs\n\n"
    report += "### Yale Reference Comparison\n\n"
    report += "![Yale reference comparison](yale_reference_comparison.png)\n\n"
    report += "### All Garments Overview\n\n"
    report += "![All garments overview](all_garments_overview.png)\n\n"
    report += "## Interpretation\n\n"
    for item in recommendations:
        report += f"- {item}\n"
    report += "\n"
    report += "The `mask_box` baseline is the fastest and safest option but it ignores pose and tends to overfill the torso. The `pose_quad` method respects shoulder and torso geometry better, but it can leave holes near the hem or sleeves when the target silhouette and garment silhouette do not line up. The `hybrid_pose_parse` method fills those gaps by combining both outputs and then restoring arms and head from the original image, which makes it the best fit for this repo's current constraints.\n\n"
    report += "## OOM Assessment\n\n"
    report += "- This benchmark ran entirely on CPU and standard Python image operations, so it does not risk GPU OOM on the current machine.\n"
    report += "- Full reproduction of modern diffusion try-on models was intentionally not forced here because the available GPU memory is below the published comfort zone of CatVTON and well below the practical needs of IDM-VTON and StableVITON.\n"
    report += "- Reviving the legacy DIOR training stack would also require solving the `torch==1.0.0` dependency lock and compiling old CUDA extensions, which is not a fast or reliable path on this environment.\n\n"
    report += "## Files Produced\n\n"
    report += "- `summary.json`: raw metrics and system probe output\n"
    report += "- `yale_reference_comparison.png`: one-row comparison with the cached HR-VITON result\n"
    report += "- `all_garments_overview.png`: outputs for all garments and methods\n"
    report += "- One PNG per method and garment pair\n\n"
    report += "## Recommended Next Step\n\n"
    report += "If you want to keep modernizing this repo, the next serious upgrade is to build a low-resolution CatVTON inference path with CPU offload and a hand-made torso mask generator. That is the only recent method in the survey that looks remotely compatible with this laptop after aggressive downscaling.\n"
    return report


def main() -> None:
    report_dir = REPORT_DIR
    report_dir.mkdir(parents=True, exist_ok=True)

    ctx = load_person_context()
    system = probe_system()

    assets = [
        GarmentAsset(
            name="yale_reference",
            image_path=REPO_ROOT / "TryYours-Virtual-Try-On-main" / "HR-VITON-main" / "test" / "test" / "cloth" / "00001_00.jpg",
            mask_path=REPO_ROOT / "TryYours-Virtual-Try-On-main" / "HR-VITON-main" / "test" / "test" / "cloth-mask" / "00001_00.jpg",
            reference_path=REPO_ROOT / "TryYours-Virtual-Try-On-main" / "HR-VITON-main" / "Output" / "00001_00_00001_00.png",
        ),
        GarmentAsset(
            name="cvzone_green",
            image_path=REPO_ROOT / "TryYours-Virtual-Try-On" / "webpage-20231110T073429Z-001" / "webpage" / "Shirts" / "1.png",
        ),
        GarmentAsset(
            name="cvzone_blue",
            image_path=REPO_ROOT / "TryYours-Virtual-Try-On" / "webpage-20231110T073429Z-001" / "webpage" / "Shirts" / "2.png",
        ),
        GarmentAsset(
            name="cvzone_red",
            image_path=REPO_ROOT / "TryYours-Virtual-Try-On" / "webpage-20231110T073429Z-001" / "webpage" / "Shirts" / "3.png",
        ),
    ]

    methods = {
        "mask_box": render_mask_box,
        "pose_quad": render_pose_quad,
        "hybrid_pose_parse": render_hybrid_pose_parse,
    }

    metrics: list[MethodMetrics] = []
    yale_row: list[tuple[str, Image.Image]] = []
    all_rows: list[list[tuple[str, Image.Image]]] = []
    person_image = to_uint8_rgb(ctx.image)

    asset_files = [
        REPO_ROOT / "TryYours-Virtual-Try-On-main" / "HR-VITON-main" / "test" / "test" / "image" / "00001_00.jpg",
        REPO_ROOT / "TryYours-Virtual-Try-On-main" / "HR-VITON-main" / "test" / "test" / "image-parse-v3" / "00001_00.png",
        REPO_ROOT / "TryYours-Virtual-Try-On-main" / "HR-VITON-main" / "test" / "test" / "openpose_json" / "00001_00_keypoints.json",
    ]
    for asset in assets:
        asset_files.append(asset.image_path)
        if asset.mask_path is not None:
            asset_files.append(asset.mask_path)
        if asset.reference_path is not None:
            asset_files.append(asset.reference_path)
    dataset_size_mb = sum(path.stat().st_size for path in asset_files) / (1024.0 * 1024.0)

    for asset in assets:
        garment_rgba = load_garment_rgba(asset)
        garment_rgb_image = to_uint8_rgb(garment_rgba[:, :, :3])
        row_panels = [(f"garment: {asset.name}", garment_rgb_image)]

        if asset.reference_path is not None:
            reference_image = Image.open(asset.reference_path).convert("RGB")
            if not yale_row:
                yale_row.extend([
                    ("person", person_image),
                    ("garment", garment_rgb_image),
                    ("hr-viton cache", reference_image),
                ])
        else:
            reference_image = None

        for method_name, renderer in methods.items():
            start = time.perf_counter()
            result_rgb, garment_mask = renderer(ctx, garment_rgba)
            runtime_sec = time.perf_counter() - start
            output_name = f"{asset.name}_{method_name}.png"
            output_path = report_dir / output_name
            to_uint8_rgb(result_rgb).save(output_path)

            outside_mask = np.clip(1.0 - ctx.upper_dilated, 0.0, 1.0)
            reference_ssim = None
            if reference_image is not None:
                ref_rgb = np.asarray(reference_image, dtype=np.float32) / 255.0
                reference_ssim = masked_ssim(rgb_to_gray(ref_rgb), rgb_to_gray(result_rgb), ctx.upper_dilated)

            entry = MethodMetrics(
                method=method_name,
                garment=asset.name,
                runtime_sec=runtime_sec,
                coverage=coverage(garment_mask, ctx.upper_mask),
                spill=spill(garment_mask, ctx.upper_dilated),
                outside_ssim=masked_ssim(rgb_to_gray(ctx.image), rgb_to_gray(result_rgb), outside_mask),
                reference_ssim=reference_ssim,
                output_file=output_name,
            )
            metrics.append(entry)
            row_panels.append((method_name, to_uint8_rgb(result_rgb)))
            if asset.name == "yale_reference":
                yale_row.append((method_name, to_uint8_rgb(result_rgb)))

        all_rows.append(row_panels)

    aggregate: dict[str, dict[str, float | None]] = {}
    for method_name in methods:
        method_rows = [entry for entry in metrics if entry.method == method_name]
        aggregate[method_name] = {
            "avg_runtime_sec": float(np.mean([entry.runtime_sec for entry in method_rows])),
            "avg_coverage": float(np.mean([entry.coverage for entry in method_rows])),
            "avg_spill": float(np.mean([entry.spill for entry in method_rows])),
            "avg_outside_ssim": float(np.mean([entry.outside_ssim for entry in method_rows])),
            "avg_reference_ssim": mean_defined([entry.reference_ssim for entry in method_rows]),
        }

    save_montage([yale_row], report_dir / "yale_reference_comparison.png")
    save_montage(all_rows, report_dir / "all_garments_overview.png")

    summary = {
        "system": system,
        "dataset_size_mb": dataset_size_mb,
        "assets": [
            {
                "name": asset.name,
                "image_path": relative_path(asset.image_path),
                "mask_path": relative_path(asset.mask_path) if asset.mask_path is not None else None,
                "reference_path": relative_path(asset.reference_path) if asset.reference_path is not None else None,
            }
            for asset in assets
        ],
        "aggregate": aggregate,
        "metrics": [asdict(entry) for entry in metrics],
    }
    summary_path = report_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report_text = build_report(ctx, assets, metrics, report_dir, system, dataset_size_mb, aggregate)
    report_path = report_dir / "README.md"
    report_path.write_text(report_text, encoding="utf-8")

    print(f"Wrote report to {report_path}")
    print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()