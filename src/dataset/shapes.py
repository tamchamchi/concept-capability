"""Geometry and anti-aliased mask rendering for the six shape primitives."""

from __future__ import annotations

import math
from collections.abc import Sequence

from PIL import Image, ImageDraw


Point = tuple[float, float]
SUPPORTED_SHAPES = frozenset(
    {"circle", "square", "triangle", "diamond", "cross", "star"}
)


def _rotate_point(x: float, y: float, angle_degrees: float) -> Point:
    angle = math.radians(angle_degrees)
    cosine, sine = math.cos(angle), math.sin(angle)
    return x * cosine - y * sine, x * sine + y * cosine


def _translate_and_rotate(
    points: Sequence[Point], center: Point, angle_degrees: float
) -> list[Point]:
    center_x, center_y = center
    return [
        (center_x + rotated_x, center_y + rotated_y)
        for x, y in points
        for rotated_x, rotated_y in [_rotate_point(x, y, angle_degrees)]
    ]


def _regular_polygon(
    center: Point,
    radius: float,
    sides: int,
    rotation_degrees: float,
) -> list[Point]:
    return [
        (
            center[0] + radius * math.cos(math.radians(rotation_degrees + i * 360 / sides)),
            center[1] + radius * math.sin(math.radians(rotation_degrees + i * 360 / sides)),
        )
        for i in range(sides)
    ]


def shape_points(
    shape_name: str,
    center: Point,
    size: float,
    rotation_degrees: float,
) -> list[Point] | None:
    """Return polygon vertices; circles return None and use their bounding box."""
    if shape_name == "circle":
        return None
    if shape_name == "square":
        half_width = size
        local = [
            (-half_width, -half_width),
            (half_width, -half_width),
            (half_width, half_width),
            (-half_width, half_width),
        ]
        return _translate_and_rotate(local, center, rotation_degrees)
    if shape_name == "triangle":
        return _regular_polygon(center, size, 3, rotation_degrees - 90.0)
    if shape_name == "diamond":
        return _regular_polygon(center, size, 4, rotation_degrees)
    if shape_name == "cross":
        radius = size
        half_arm_width = size * 0.34
        local = [
            (-half_arm_width, -radius),
            (half_arm_width, -radius),
            (half_arm_width, -half_arm_width),
            (radius, -half_arm_width),
            (radius, half_arm_width),
            (half_arm_width, half_arm_width),
            (half_arm_width, radius),
            (-half_arm_width, radius),
            (-half_arm_width, half_arm_width),
            (-radius, half_arm_width),
            (-radius, -half_arm_width),
            (-half_arm_width, -half_arm_width),
        ]
        return _translate_and_rotate(local, center, rotation_degrees)
    if shape_name == "star":
        points = []
        for index in range(10):
            radius = size if index % 2 == 0 else size * 0.45
            angle = rotation_degrees - 90.0 + index * 36.0
            points.append(
                (
                    center[0] + radius * math.cos(math.radians(angle)),
                    center[1] + radius * math.sin(math.radians(angle)),
                )
            )
        return points
    raise ValueError(f"Unsupported shape: {shape_name!r}")


def shape_bounds(
    shape_name: str,
    size: float,
    rotation_degrees: float,
) -> tuple[float, float, float, float]:
    """Return bounds relative to a center at (0, 0)."""
    if shape_name == "circle":
        return -size, -size, size, size
    points = shape_points(shape_name, (0.0, 0.0), size, rotation_degrees)
    assert points is not None
    x_values = [point[0] for point in points]
    y_values = [point[1] for point in points]
    return min(x_values), min(y_values), max(x_values), max(y_values)


def draw_shape_mask(
    shape_name: str,
    image_size: int,
    center: Point,
    size: float,
    rotation_degrees: float,
    supersampling_factor: int = 4,
    downsampling_filter: str = "lanczos",
    canvas_padding: float = 0.0,
) -> Image.Image:
    """Render a grayscale shape mask with supersampling-based anti-aliasing."""
    if shape_name not in SUPPORTED_SHAPES:
        raise ValueError(f"Unsupported shape: {shape_name!r}")
    if supersampling_factor < 1:
        raise ValueError("supersampling_factor must be at least 1")

    scale = supersampling_factor
    high_resolution_size = image_size * scale
    mask = Image.new("L", (high_resolution_size, high_resolution_size), 0)
    draw = ImageDraw.Draw(mask)
    scaled_center = (center[0] * scale, center[1] * scale)

    if shape_name == "circle":
        scaled_radius = size * scale
        draw.ellipse(
            (
                scaled_center[0] - scaled_radius,
                scaled_center[1] - scaled_radius,
                scaled_center[0] + scaled_radius,
                scaled_center[1] + scaled_radius,
            ),
            fill=255,
        )
    else:
        points = shape_points(shape_name, center, size, rotation_degrees)
        assert points is not None
        draw.polygon([(x * scale, y * scale) for x, y in points], fill=255)

    if scale != 1:
        filters = {
            "nearest": Image.Resampling.NEAREST,
            "bilinear": Image.Resampling.BILINEAR,
            "bicubic": Image.Resampling.BICUBIC,
            "lanczos": Image.Resampling.LANCZOS,
        }
        try:
            resampling_filter = filters[downsampling_filter.lower()]
        except KeyError as exc:
            raise ValueError(
                f"Unsupported downsampling filter: {downsampling_filter}"
            ) from exc
        mask = mask.resize((image_size, image_size), resampling_filter)

    # Lanczos can introduce tiny ringing pixels beyond the mathematical boundary.
    # Clear the configured safety margin so "not cropped" also holds pixel-wise.
    clear_pixels = math.ceil(canvas_padding)
    if clear_pixels > 0:
        if clear_pixels * 2 >= image_size:
            raise ValueError("canvas_padding leaves no drawable canvas")
        border = ImageDraw.Draw(mask)
        border.rectangle((0, 0, image_size - 1, clear_pixels - 1), fill=0)
        border.rectangle(
            (0, image_size - clear_pixels, image_size - 1, image_size - 1), fill=0
        )
        border.rectangle((0, 0, clear_pixels - 1, image_size - 1), fill=0)
        border.rectangle(
            (image_size - clear_pixels, 0, image_size - 1, image_size - 1), fill=0
        )
    return mask
