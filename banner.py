"""Generate the Now Playing banner image.

Builds a 1:1 square image with the YouTube thumbnail as background, dark
overlay for readability, and song title + artist + BandiBot mark composited
on top. Returns PNG bytes ready to attach to a Discord embed.
"""

import io
import logging
from typing import Optional

import aiohttp
from PIL import Image, ImageDraw, ImageFont, ImageFilter

logger = logging.getLogger(__name__)

# Image dimensions (square)
SIZE = 230

# Color palette — matches the embed's dark/moody theme
OVERLAY_RGBA = (15, 5, 25, 200)         # near-black purple, ~78% opacity
TITLE_COLOR = (255, 255, 255)           # white for the song title
ARTIST_COLOR = (212, 175, 55)           # gold for the artist line
WATERMARK_COLOR = (160, 100, 200, 200)  # soft magenta, semi-transparent

# Font search order — tries common system fonts, falls back to Pillow default
TITLE_FONT_CANDIDATES = (
    "C:/Windows/Fonts/segoeuib.ttf",     # Segoe UI Bold (Windows)
    "C:/Windows/Fonts/arial.ttf",        # Arial (Windows fallback)
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",   # macOS
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",  # Linux
)
BODY_FONT_CANDIDATES = (
    "C:/Windows/Fonts/segoeui.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
)


def _load_font(candidates: tuple, size: int) -> ImageFont.ImageFont:
    """Try each font path in order; fall back to Pillow's default."""
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def _wrap_text(draw: ImageDraw.ImageDraw, text: str, font, max_width: int) -> list[str]:
    """Greedy line-wrap text to fit within max_width pixels.

    Pillow doesn't have built-in word wrap, so we measure word by word.
    """
    words = text.split()
    if not words:
        return [""]

    lines = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        bbox = draw.textbbox((0, 0), candidate, font=font)
        width = bbox[2] - bbox[0]
        if width <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


async def _fetch_thumbnail(url: str) -> Optional[Image.Image]:
    """Download the thumbnail URL into a Pillow Image."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status != 200:
                    return None
                data = await resp.read()
                return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as e:
        logger.error(f"[banner] thumbnail fetch failed: {e}")
        return None


def _crop_to_square(img: Image.Image) -> Image.Image:
    """Center-crop a rectangular image to a square."""
    w, h = img.size
    if w == h:
        return img
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    return img.crop((left, top, left + side, top + side))


async def generate_banner(
    title: str,
    artist: Optional[str],
    thumbnail_url: Optional[str],
) -> bytes:
    """Generate a square Now Playing banner. Returns PNG bytes.

    Falls back to a solid-color background if the thumbnail can't be fetched.
    """
    if thumbnail_url:
        bg = await _fetch_thumbnail(thumbnail_url)
    else:
        bg = None

    if bg:
        bg = _crop_to_square(bg).resize((SIZE, SIZE), Image.Resampling.LANCZOS)
        bg = bg.filter(ImageFilter.GaussianBlur(radius=2))
    else:
        bg = Image.new("RGB", (SIZE, SIZE), (30, 10, 50))

    canvas = bg.convert("RGBA")
    overlay = Image.new("RGBA", canvas.size, OVERLAY_RGBA)
    canvas = Image.alpha_composite(canvas, overlay)

    draw = ImageDraw.Draw(canvas)

    title_font = _load_font(TITLE_FONT_CANDIDATES, 18)
    artist_font = _load_font(BODY_FONT_CANDIDATES, 12)
    watermark_font = _load_font(BODY_FONT_CANDIDATES, 10)

    padding = 16
    max_text_width = SIZE - (padding * 2)

    title_lines = _wrap_text(draw, title, title_font, max_text_width)
    if len(title_lines) > 3:
        title_lines = title_lines[:3]
        title_lines[-1] += "…"

    line_height = 22
    title_block_height = line_height * len(title_lines)
    artist_height = 18 if artist else 0
    gap = 10 if artist else 0
    total_block = title_block_height + gap + artist_height
    start_y = (SIZE - total_block) // 2 - 10

    y = start_y
    for line in title_lines:
        bbox = draw.textbbox((0, 0), line, font=title_font)
        line_width = bbox[2] - bbox[0]
        x = (SIZE - line_width) // 2
        draw.text((x, y), line, font=title_font, fill=TITLE_COLOR)
        y += line_height

    if artist:
        y += gap
        artist_text = f"by {artist}"
        bbox = draw.textbbox((0, 0), artist_text, font=artist_font)
        line_width = bbox[2] - bbox[0]
        x = (SIZE - line_width) // 2
        draw.text((x, y), artist_text, font=artist_font, fill=ARTIST_COLOR)

    watermark_text = "• Brought you by BandiBot"
    bbox = draw.textbbox((0, 0), watermark_text, font=watermark_font)
    wm_width = bbox[2] - bbox[0]
    wm_height = bbox[3] - bbox[1]
    draw.text(
        (SIZE - wm_width - padding, SIZE - wm_height - padding),
        watermark_text,
        font=watermark_font,
        fill=WATERMARK_COLOR,
    )

    output = io.BytesIO()
    canvas.convert("RGB").save(output, format="PNG", optimize=True)
    return output.getvalue()