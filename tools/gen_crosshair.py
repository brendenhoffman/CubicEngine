#!/usr/bin/env python3
"""Regenerate assets/ui/crosshair.png — the default engine crosshair.

Drawn at 4x supersampling then downscaled for clean anti-aliased edges.
Deliberately shipped high-res (native 256x256) relative to its typical
on-screen display size (cfg.ui.crosshair_size, default 32px) so it stays
crisp, and so anyone swapping in their own crosshair.png has plenty of
headroom to work with. Classic 4-segment "gap" crosshair: white fill with a
dark outline so it stays visible against both bright and dark backgrounds.
"""

from PIL import Image, ImageDraw

SIZE = 256
SUPERSAMPLE = 4
canvas = SIZE * SUPERSAMPLE
center = canvas / 2

gap = 20 * SUPERSAMPLE  # distance from center to the start of each arm
length = 50 * SUPERSAMPLE  # arm length
thickness = 8 * SUPERSAMPLE
outline = 3 * SUPERSAMPLE  # extra half-width added for the dark outline

img = Image.new("RGBA", (canvas, canvas), (0, 0, 0, 0))
draw = ImageDraw.Draw(img)

arms = [
    # (x0, y0, x1, y1) for the fill rect, before the outline pass
    (center - thickness / 2, center - gap - length, center + thickness / 2, center - gap),  # up
    (center - thickness / 2, center + gap, center + thickness / 2, center + gap + length),  # down
    (center - gap - length, center - thickness / 2, center - gap, center + thickness / 2),  # left
    (center + gap, center - thickness / 2, center + gap + length, center + thickness / 2),  # right
]

for x0, y0, x1, y1 in arms:
    draw.rectangle((x0 - outline, y0 - outline, x1 + outline, y1 + outline), fill=(0, 0, 0, 200))
for x0, y0, x1, y1 in arms:
    draw.rectangle((x0, y0, x1, y1), fill=(255, 255, 255, 255))

img = img.resize((SIZE, SIZE), Image.LANCZOS)
img.save("assets/ui/crosshair.png")
print(f"wrote assets/ui/crosshair.png ({SIZE}x{SIZE})")
