from PIL import Image, ImageDraw

def make_icon():
    size = 64  # draw at 2x then downsample for crisp edges
    img = Image.new("RGBA", (size, size), (124, 109, 250, 255))
    d = ImageDraw.Draw(img)

    # Scale helpers (drawing coords assume a 32px canvas, scale=2)
    s = size / 32

    # Stem: x=24-27, y=8-22 (in 32px space)
    d.rectangle([24*s, 8*s, 27*s, 22*s], fill=(255, 255, 255, 255))

    # Flag: x=24-32, y=8-13
    d.rectangle([24*s, 8*s, 32*s, 13*s], fill=(255, 255, 255, 255))

    # Note head: circle centred at (20,22), radius 5
    cx, cy, r = 20*s, 22*s, 5*s
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(255, 255, 255, 255))

    # Export multi-size .ico
    sizes = [16, 32, 48]
    frames = []
    for sz in sizes:
        frames.append(img.resize((sz, sz), Image.LANCZOS))

    frames[0].save(
        "icon.ico",
        format="ICO",
        sizes=[(sz, sz) for sz in sizes],
        append_images=frames[1:],
    )
    print("icon.ico created.")

if __name__ == "__main__":
    make_icon()
