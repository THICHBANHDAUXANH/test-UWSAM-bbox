#!/usr/bin/env python3
"""
Simple script to render the ASCII diagram into a PNG using Pillow.
If Pillow is not installed, the script prints an instruction to install it.
"""
import os
import sys

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception as e:
    print("Pillow is not installed. Install with: pip install pillow")
    sys.exit(2)

here = os.path.dirname(__file__)
txt_path = os.path.join(here, 'usis_sam_arch.txt')
png_path = os.path.join(here, 'usis_sam_arch.png')

if not os.path.exists(txt_path):
    print('ASCII diagram file not found:', txt_path)
    sys.exit(1)

with open(txt_path, 'r', encoding='utf-8') as f:
    lines = f.read().splitlines()

# Load default font
font = ImageFont.load_default()
# Compute image size
maxw = 0
line_h = font.getsize('A')[1] + 4
for line in lines:
    w = font.getsize(line)[0]
    if w > maxw:
        maxw = w

pad_x = 20
pad_y = 20
img_w = maxw + pad_x * 2
img_h = line_h * max(1, len(lines)) + pad_y * 2

img = Image.new('RGB', (img_w, img_h), color='white')
d = ImageDraw.Draw(img)

y = pad_y
for line in lines:
    d.text((pad_x, y), line, font=font, fill=(10, 10, 10))
    y += line_h

img.save(png_path)
print('Wrote PNG:', png_path)
