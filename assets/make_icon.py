"""Generate assets/icon.ico — a microphone glyph on a rounded indigo tile.

Run:  python assets/make_icon.py
Regenerate only when the design changes; icon.ico is committed.
"""
import pathlib

from PIL import Image, ImageDraw

OUT = pathlib.Path(__file__).parent / "icon.ico"
S = 256
BG1 = (79, 70, 229)    # indigo
BG2 = (139, 92, 246)   # violet
WHITE = (255, 255, 255)


def render(size):
    scale = 4
    n = size * scale
    img = Image.new("RGBA", (n, n), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    # rounded tile with a simple vertical gradient
    radius = int(n * 0.22)
    tile = Image.new("RGBA", (n, n), (0, 0, 0, 0))
    grad = Image.new("RGBA", (1, n))
    for y in range(n):
        t = y / n
        grad.putpixel((0, y), tuple(int(BG1[i] + (BG2[i] - BG1[i]) * t) for i in range(3)) + (255,))
    grad = grad.resize((n, n))
    mask = Image.new("L", (n, n), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, n - 1, n - 1], radius=radius, fill=255)
    tile.paste(grad, (0, 0), mask)
    img = Image.alpha_composite(img, tile)
    d = ImageDraw.Draw(img)

    # microphone capsule
    cx = n // 2
    body_w = int(n * 0.26)
    body_top = int(n * 0.22)
    body_bot = int(n * 0.56)
    d.rounded_rectangle([cx - body_w // 2, body_top, cx + body_w // 2, body_bot],
                        radius=body_w // 2, fill=WHITE)
    # stand arc
    arc_box = [cx - int(n * 0.20), int(n * 0.34), cx + int(n * 0.20), int(n * 0.70)]
    d.arc(arc_box, start=20, end=160, fill=WHITE, width=int(n * 0.045))
    # stem + base
    d.line([cx, int(n * 0.66), cx, int(n * 0.80)], fill=WHITE, width=int(n * 0.05))
    d.line([cx - int(n * 0.11), int(n * 0.80), cx + int(n * 0.11), int(n * 0.80)],
           fill=WHITE, width=int(n * 0.05))

    return img.resize((size, size), Image.LANCZOS)


def main():
    sizes = [16, 24, 32, 48, 64, 128, 256]
    imgs = [render(s) for s in sizes]
    imgs[-1].save(OUT, format="ICO", sizes=[(s, s) for s in sizes],
                  append_images=imgs[:-1])
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
