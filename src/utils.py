import os
from PIL import ImageFont

def load_font(font_name="DejaVuSans.ttf", size=18):
    """
    Loads a TrueType font, falling back to default if unavailable.
    """
    try:
        font = ImageFont.truetype(font_name, size)
        print(f"Font '{font_name}' loaded successfully (size {size}).")
        return font
    except IOError:
        print(f"Warning: Could not load TTF font '{font_name}'. Using default.")
        try:
            return ImageFont.load_default(size)
        except AttributeError:
             # Fallback for older Pillow versions
            return ImageFont.load_default()
