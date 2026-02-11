"""Generate a 1024x1024 flowsurface app icon using Pillow.

Dark background with stylized candlestick/range bar chart.
Green (buy) and red (sell) bars on dark surface.
"""
from PIL import Image, ImageDraw, ImageFont
import random

SIZE = 1024
PADDING = 80
CORNER_RADIUS = 180  # macOS rounded rect

# Colors matching flowsurface dark theme
BG_COLOR = (24, 26, 32)          # dark charcoal
BG_INNER = (30, 33, 40)          # slightly lighter inner
GREEN = (72, 199, 142)           # success/buy
RED = (234, 87, 89)              # danger/sell
GREEN_DIM = (50, 140, 100)       # dimmed green
RED_DIM = (160, 60, 62)          # dimmed red
GRID_COLOR = (45, 48, 58)        # subtle grid
ACCENT = (100, 120, 160)         # subtle accent for text


def rounded_rectangle(draw, xy, radius, fill):
    """Draw a rounded rectangle."""
    x0, y0, x1, y1 = xy
    draw.rectangle([x0 + radius, y0, x1 - radius, y1], fill=fill)
    draw.rectangle([x0, y0 + radius, x1, y1 - radius], fill=fill)
    draw.pieslice([x0, y0, x0 + 2 * radius, y0 + 2 * radius], 180, 270, fill=fill)
    draw.pieslice([x1 - 2 * radius, y0, x1, y0 + 2 * radius], 270, 360, fill=fill)
    draw.pieslice([x0, y1 - 2 * radius, x0 + 2 * radius, y1], 90, 180, fill=fill)
    draw.pieslice([x1 - 2 * radius, y1 - 2 * radius, x1, y1], 0, 90, fill=fill)


def draw_candle(draw, x, width, open_y, close_y, high_y, low_y, is_bull):
    """Draw a single candlestick."""
    color = GREEN if is_bull else RED
    dim_color = GREEN_DIM if is_bull else RED_DIM

    body_top = min(open_y, close_y)
    body_bottom = max(open_y, close_y)
    body_height = max(body_bottom - body_top, 4)

    # Wick
    wick_x = x + width // 2
    draw.line([(wick_x, high_y), (wick_x, low_y)], fill=dim_color, width=max(3, width // 8))

    # Body
    draw.rectangle([x, body_top, x + width, body_top + body_height], fill=color)


def main():
    img = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Background rounded rect
    rounded_rectangle(draw, (0, 0, SIZE - 1, SIZE - 1), CORNER_RADIUS, BG_COLOR)

    # Inner area
    inner_pad = 40
    rounded_rectangle(
        draw,
        (inner_pad, inner_pad, SIZE - inner_pad - 1, SIZE - inner_pad - 1),
        CORNER_RADIUS - 30,
        BG_INNER,
    )

    # Chart area
    chart_left = PADDING + 20
    chart_right = SIZE - PADDING - 20
    chart_top = PADDING + 100
    chart_bottom = SIZE - PADDING - 80
    chart_height = chart_bottom - chart_top
    chart_width = chart_right - chart_left

    # Horizontal grid lines
    for i in range(5):
        y = chart_top + int(chart_height * i / 4)
        draw.line([(chart_left, y), (chart_right, y)], fill=GRID_COLOR, width=2)

    # Generate candlestick data (ascending trend with pullbacks)
    random.seed(42)  # deterministic
    n_candles = 14
    candle_width = int(chart_width / n_candles * 0.65)
    candle_spacing = chart_width / n_candles

    # Price series: trending up with volatility
    prices = [0.45]
    for i in range(n_candles):
        trend = 0.015  # slight uptrend
        noise = random.gauss(0, 0.04)
        prices.append(max(0.1, min(0.9, prices[-1] + trend + noise)))

    for i in range(n_candles):
        x = int(chart_left + i * candle_spacing + (candle_spacing - candle_width) / 2)

        open_price = prices[i]
        close_price = prices[i + 1]
        is_bull = close_price >= open_price

        # Wicks extend beyond body
        wick_extend = random.uniform(0.01, 0.06)
        high_price = max(open_price, close_price) + wick_extend
        low_price = min(open_price, close_price) - random.uniform(0.01, 0.05)

        # Map to pixel coordinates (inverted Y)
        open_y = int(chart_top + (1 - open_price) * chart_height)
        close_y = int(chart_top + (1 - close_price) * chart_height)
        high_y = int(chart_top + (1 - high_price) * chart_height)
        low_y = int(chart_top + (1 - low_price) * chart_height)

        draw_candle(draw, x, candle_width, open_y, close_y, high_y, low_y, is_bull)

    # Volume bars at bottom (smaller, inside chart)
    vol_top = chart_bottom - int(chart_height * 0.18)
    vol_bottom = chart_bottom

    for i in range(n_candles):
        x = int(chart_left + i * candle_spacing + (candle_spacing - candle_width) / 2)
        is_bull = prices[i + 1] >= prices[i]
        vol = random.uniform(0.3, 1.0)
        bar_height = int((vol_bottom - vol_top) * vol)
        color = (*GREEN, 100) if is_bull else (*RED, 100)

        # Need to use a temporary image for alpha volume bars
        vol_bar = Image.new("RGBA", img.size, (0, 0, 0, 0))
        vol_draw = ImageDraw.Draw(vol_bar)
        vol_draw.rectangle(
            [x, vol_bottom - bar_height, x + candle_width, vol_bottom],
            fill=color,
        )
        img = Image.alpha_composite(img, vol_bar)

    # Redraw on final composite
    draw = ImageDraw.Draw(img)

    # "FLOWSURFACE" text at top
    try:
        font_large = ImageFont.truetype("/System/Library/Fonts/SFCompact.ttf", 52)
        font_small = ImageFont.truetype("/System/Library/Fonts/SFCompact.ttf", 36)
    except (OSError, IOError):
        try:
            font_large = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 52)
            font_small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 36)
        except (OSError, IOError):
            font_large = ImageFont.load_default()
            font_small = ImageFont.load_default()

    # Title
    draw.text(
        (SIZE // 2, PADDING + 50),
        "FLOWSURFACE",
        fill=ACCENT,
        font=font_large,
        anchor="mm",
    )

    # Subtle "BPR25" label
    draw.text(
        (SIZE // 2, SIZE - PADDING - 30),
        "BPR25",
        fill=(*ACCENT, 150) if len(ACCENT) == 3 else ACCENT,
        font=font_small,
        anchor="mm",
    )

    # Save
    output_path = "/Users/terryli/fork-tools/flowsurface/assets/icon_1024.png"
    img.save(output_path, "PNG")
    print(f"Icon saved to {output_path}")
    print(f"Size: {img.size}")


if __name__ == "__main__":
    main()
