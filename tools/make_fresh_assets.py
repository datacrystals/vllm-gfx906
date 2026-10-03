#!/usr/bin/env python3
"""Regenerate test assets with different shapes/colors to dodge prefix cache."""
from PIL import Image, ImageDraw

img = Image.new("RGB", (512, 512), (255, 220, 0))
dr = ImageDraw.Draw(img)
dr.polygon([(256, 80), (80, 420), (432, 420)], fill=(0, 180, 0))
img.save("/data/tmp/omni_assets/test_image2.png")
print("wrote test_image2.png (green triangle on yellow)")
