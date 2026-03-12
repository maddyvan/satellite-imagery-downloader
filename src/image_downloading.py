import cv2
import requests
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor, as_completed


def download_tile(url, headers, channels, timeout=15, max_retries=3):
    """Download a single tile with retries and exponential backoff."""
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()
            arr = np.asarray(bytearray(response.content), dtype=np.uint8)
            if channels == 3:
                return cv2.imdecode(arr, 1)
            return cv2.imdecode(arr, -1)
        except requests.RequestException as e:
            if attempt < max_retries - 1:
                time.sleep(0.5 * (2 ** attempt))  # 0.5s, 1s, 2s backoff
            else:
                print(f"Failed to download {url}: {e}")
                return None
    return None


# Mercator projection 
# https://developers.google.com/maps/documentation/javascript/examples/map-coordinates
def project_with_scale(lat, lon, scale):
    siny = np.sin(lat * np.pi / 180)
    siny = min(max(siny, -0.9999), 0.9999)
    x = scale * (0.5 + lon / 360)
    y = scale * (0.5 - np.log((1 + siny) / (1 - siny)) / (4 * np.pi))
    return x, y


def download_image(lat1: float, lon1: float, lat2: float, lon2: float,
    zoom: int, url: str, headers: dict, tile_size: int = 256, channels: int = 3) -> np.ndarray:
    """
    Downloads a map region. Returns an image stored as a `numpy.ndarray` in BGR or BGRA, depending on the number
    of `channels`.

    Parameters
    ----------
    `(lat1, lon1)` - Coordinates (decimal degrees) of the top-left corner of a rectangular area

    `(lat2, lon2)` - Coordinates (decimal degrees) of the bottom-right corner of a rectangular area

    `zoom` - Zoom level

    `url` - Tile URL with {x}, {y} and {z} in place of its coordinate and zoom values

    `headers` - Dictionary of HTTP headers

    `tile_size` - Tile size in pixels

    `channels` - Number of channels in the output image. Also affects how the tiles are converted into numpy arrays.
    """

    scale = 1 << zoom

    # Find the pixel coordinates and tile coordinates of the corners
    tl_proj_x, tl_proj_y = project_with_scale(lat1, lon1, scale)
    br_proj_x, br_proj_y = project_with_scale(lat2, lon2, scale)

    tl_pixel_x = int(tl_proj_x * tile_size)
    tl_pixel_y = int(tl_proj_y * tile_size)
    br_pixel_x = int(br_proj_x * tile_size)
    br_pixel_y = int(br_proj_y * tile_size)

    tl_tile_x = int(tl_proj_x)
    tl_tile_y = int(tl_proj_y)
    br_tile_x = int(br_proj_x)
    br_tile_y = int(br_proj_y)

    img_w = abs(tl_pixel_x - br_pixel_x)
    img_h = br_pixel_y - tl_pixel_y
    img = np.zeros((img_h, img_w, channels), np.uint8)

    # Limit concurrent requests to avoid timeouts and rate limiting (default: 8 workers)
    max_workers = 8

    def place_tile(tile_y, tile_x):
        tile = download_tile(url.format(x=tile_x, y=tile_y, z=zoom), headers, channels)
        if tile is None:
            return None
        # Find the pixel coordinates of the new tile relative to the image
        tl_rel_x = tile_x * tile_size - tl_pixel_x
        tl_rel_y = tile_y * tile_size - tl_pixel_y
        br_rel_x = tl_rel_x + tile_size
        br_rel_y = tl_rel_y + tile_size
        img_x_l = max(0, tl_rel_x)
        img_x_r = min(img_w + 1, br_rel_x)
        img_y_l = max(0, tl_rel_y)
        img_y_r = min(img_h + 1, br_rel_y)
        cr_x_l = max(0, -tl_rel_x)
        cr_x_r = tile_size + min(0, img_w - br_rel_x)
        cr_y_l = max(0, -tl_rel_y)
        cr_y_r = tile_size + min(0, img_h - br_rel_y)
        return (img_y_l, img_y_r, img_x_l, img_x_r, tile[cr_y_l:cr_y_r, cr_x_l:cr_x_r])

    # Progress tracking
    total_tiles = (br_tile_y - tl_tile_y + 1) * (br_tile_x - tl_tile_x + 1)
    completed_tiles = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(place_tile, tile_y, tile_x): (tile_y, tile_x)
            for tile_y in range(tl_tile_y, br_tile_y + 1)
            for tile_x in range(tl_tile_x, br_tile_x + 1)
        }
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                img_y_l, img_y_r, img_x_l, img_x_r, tile_data = result
                img[img_y_l:img_y_r, img_x_l:img_x_r] = tile_data

            completed_tiles += 1
            # Print progress every 10 tiles or on completion
            if completed_tiles % 10 == 0 or completed_tiles == total_tiles:
                percent = completed_tiles * 100 // total_tiles
                print(f"Downloading tiles: {completed_tiles}/{total_tiles} ({percent}%)")

    return img


def image_size(lat1: float, lon1: float, lat2: float,
    lon2: float, zoom: int, tile_size: int = 256):
    """ Calculates the size of an image without downloading it. Returns the width and height in pixels as a tuple. """

    scale = 1 << zoom
    tl_proj_x, tl_proj_y = project_with_scale(lat1, lon1, scale)
    br_proj_x, br_proj_y = project_with_scale(lat2, lon2, scale)

    tl_pixel_x = int(tl_proj_x * tile_size)
    tl_pixel_y = int(tl_proj_y * tile_size)
    br_pixel_x = int(br_proj_x * tile_size)
    br_pixel_y = int(br_proj_y * tile_size)

    return abs(tl_pixel_x - br_pixel_x), br_pixel_y - tl_pixel_y
