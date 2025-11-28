"""Generate thumbnail for MPT-7B MoE Training blog post."""

from PIL import Image, ImageDraw, ImageFont
import os

# Create image with dark blue gradient background
width, height = 1200, 630
img = Image.new('RGB', (width, height), color='#1a1a2e')

# Create a gradient background
draw = ImageDraw.Draw(img)
for i in range(height):
    # Gradient from dark blue to slightly lighter blue
    r = int(26 + (i / height) * 20)
    g = int(26 + (i / height) * 40)
    b = int(46 + (i / height) * 60)
    draw.rectangle([(0, i), (width, i+1)], fill=(r, g, b))

# Add some accent shapes
draw.rectangle([(0, 0), (20, height)], fill='#3498db')  # Left accent bar
draw.rectangle([(0, height-20), (width, height)], fill='#2c3e50')  # Bottom bar

# Try to load a nice font, fall back to default if not available
try:
    title_font = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', 80)
    subtitle_font = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', 40)
    detail_font = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', 30)
except:
    try:
        title_font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 80)
        subtitle_font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 40)
        detail_font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 30)
    except:
        title_font = ImageFont.load_default()
        subtitle_font = ImageFont.load_default()
        detail_font = ImageFont.load_default()

# Add text content
draw = ImageDraw.Draw(img)

# Title
title = "MPT-7B MoE"
title_bbox = draw.textbbox((0, 0), title, font=title_font)
title_width = title_bbox[2] - title_bbox[0]
draw.text(((width - title_width) // 2, 120), title, fill='#ffffff', font=title_font)

# Subtitle
subtitle = "Training with Accelerate"
subtitle_bbox = draw.textbbox((0, 0), subtitle, font=subtitle_font)
subtitle_width = subtitle_bbox[2] - subtitle_bbox[0]
draw.text(((width - subtitle_width) // 2, 220), subtitle, fill='#ecf0f1', font=subtitle_font)

# Details
details = [
    "DeepSpeed ZeRO Stage 2",
    "2x H100 GPUs • 4 Experts",
    "2-2.5 hours/epoch"
]

y_offset = 320
for detail in details:
    detail_bbox = draw.textbbox((0, 0), detail, font=detail_font)
    detail_width = detail_bbox[2] - detail_bbox[0]
    draw.text(((width - detail_width) // 2, y_offset), detail, fill='#3498db', font=detail_font)
    y_offset += 50

# Add decorative elements
# Draw some neural network-like nodes
node_color = '#3498db'
connection_color = '#2c3e50'

# Left side nodes
nodes_left = [(100, 150), (100, 300), (100, 450)]
nodes_right = [(1100, 150), (1100, 300), (1100, 450)]

# Draw connections
for node in nodes_left:
    for node2 in nodes_right:
        draw.line([node, node2], fill=connection_color, width=1)

# Draw nodes
for node in nodes_left + nodes_right:
    draw.ellipse([node[0]-15, node[1]-15, node[0]+15, node[1]+15],
                  fill=node_color, outline='#ffffff', width=2)

# Save the image
output_path = '/Users/vijayv/PycharmProjects/large-scale-training/posts/thumbnail.png'
img.save(output_path, 'PNG', quality=95)
print(f"Thumbnail saved to: {output_path}")