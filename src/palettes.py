import colorsys, os, random

random.seed(0)

# mood: (hue_lo, hue_hi, sat_lo, sat_hi)
MOODS = {
    "calm and trustworthy": (0.52, 0.62, 0.45, 0.70),
    "warm and friendly": (0.02, 0.10, 0.60, 0.85),
    "fresh and natural": (0.25, 0.40, 0.40, 0.65),
    "bold and playful": (0.10, 0.17, 0.75, 0.95),
    "elegant and minimal": (0.55, 0.65, 0.15, 0.35),
    "serious and professional": (0.58, 0.66, 0.30, 0.50),
}
SITES = ["portfolio", "restaurant site", "online store", "blog", "dashboard",
         "fitness app", "startup landing page", "school website", "photography site"]
SCHEMES = {"analogous": 0.08, "complementary": 0.5, "split-complementary": 0.42, "triadic": 0.33}


def hsl(h, s, l):
    r, g, b = colorsys.hls_to_rgb(h % 1.0, l, s)
    return "#%02x%02x%02x" % (round(r * 255), round(g * 255), round(b * 255))


def lum(c):
    v = [int(c[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    v = [x / 12.92 if x <= 0.03928 else ((x + 0.055) / 1.055) ** 2.4 for x in v]
    return 0.2126 * v[0] + 0.7152 * v[1] + 0.0722 * v[2]


def contrast(a, b):
    hi, lo = sorted((lum(a), lum(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def fit(h, s, bg, target, step):
    l = 0.5
    c = hsl(h, s, l)
    while contrast(c, bg) < target and 0.05 < l < 0.95:
        l += step
        c = hsl(h, s, l)
    return c


def example():
    mood, (h0, h1, s0, s1) = random.choice(list(MOODS.items()))
    site = random.choice(SITES)
    dark = random.random() < 0.4
    h, s = random.uniform(h0, h1), random.uniform(s0, s1)
    scheme, off = random.choice(list(SCHEMES.items()))

    bg = hsl(h, 0.15, 0.10 if dark else 0.98)
    surface = hsl(h, 0.15, 0.15 if dark else 0.94)
    text = hsl(h, 0.10, 0.93 if dark else 0.12)
    step = 0.02 if dark else -0.02
    primary = fit(h, s, bg, 4.5, step)
    accent = fit(h + off, s, bg, 3.0, step)
    on_primary = max(("#ffffff", "#111111"), key=lambda c: contrast(c, primary))

    theme = "dark" if dark else "light"
    return (
        f"<|user|>Pick a color palette for a {mood} {site}. Use a {theme} theme.<|assistant|>\n"
        ":root {\n"
        f"  --bg: {bg};\n  --surface: {surface};\n  --text: {text};\n"
        f"  --primary: {primary};\n  --on-primary: {on_primary};\n  --accent: {accent};\n}}\n"
        f"/* {scheme} scheme. About 60% background and surface, 30% text, 10% primary and accent.\n"
        f"   Text on background: {contrast(text, bg):.1f}:1. Primary on background: {contrast(primary, bg):.1f}:1.\n"
        f"   Solid colors, no gradients. */<|end|>"
    )


def main(n=8000, out="design/generated.txt"):
    os.makedirs("design", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n<|eos|>\n".join(example() for _ in range(n)))
    print("wrote", n, "examples to", out)