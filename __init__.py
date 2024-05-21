
from .main import apply_overlay_image

WEB_DIRECTORY = "js"
NODE_CLASS_MAPPINGS = {

    "Image Overlay": apply_overlay_image,
}
CC_VERSION = 2.0

__all__ = ['NODE_CLASS_MAPPINGS']
