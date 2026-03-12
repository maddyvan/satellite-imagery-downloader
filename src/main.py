import os
import json
import re
import cv2
from datetime import datetime

from image_downloading import download_image, image_size

file_dir = os.path.dirname(__file__)
prefs_path = os.path.join(file_dir, 'preferences.json')
default_prefs = {
        'url': 'https://mt.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
        'tile_size': 256,
        'channels': 3,
        'dir': os.path.join(file_dir, 'images'),
        'headers': {
            'cache-control': 'max-age=0',
            'sec-ch-ua': '" Not A;Brand";v="99", "Chromium";v="99", "Google Chrome";v="99"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"Windows"',
            'sec-fetch-dest': 'document',
            'sec-fetch-mode': 'navigate',
            'sec-fetch-site': 'none',
            'sec-fetch-user': '?1',
            'upgrade-insecure-requests': '1',
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/99.0.4844.82 Safari/537.36'
        },
        'tl': '',
        'br': '',
        'zoom': ''
    }


def take_input(messages):
    inputs = []
    print('Enter "r" to reset or "q" to exit.')
    for message in messages:
        inp = input(message)
        if inp == 'q' or inp == 'Q':
            return None
        if inp == 'r' or inp == 'R':
            return take_input(messages)
        inputs.append(inp)
    return inputs


def download_centered_square(lat_c, lon_c, zoom, size_px, url, headers, tile_size, channels):
    """Download a square image of given pixel size centered at (lat_c, lon_c)."""
    # Find a square lat/lon region that yields at least size_px x size_px pixels,
    # using repeated calls to `image_size` (which is cheap and offline).
    delta = 0.001  # initial half-size in degrees
    lat1 = lat2 = lon1 = lon2 = None

    for _ in range(25):
        lat1 = lat_c + delta
        lat2 = lat_c - delta
        lon1 = lon_c - delta
        lon2 = lon_c + delta
        w, h = image_size(lat1, lon1, lat2, lon2, zoom, tile_size)
        if w >= size_px and h >= size_px:
            break
        delta *= 1.5

    print(f"Using bounding box tl=({lat1}, {lon1}), br=({lat2}, {lon2}) "
          f"to generate centered {size_px}x{size_px} image.")

    img = download_image(lat1, lon1, lat2, lon2, zoom, url, headers, tile_size, channels)

    # Crop a centered square of the requested size
    img_h, img_w = img.shape[:2]
    cx = img_w // 2
    cy = img_h // 2
    half = size_px // 2

    x1 = max(0, cx - half)
    y1 = max(0, cy - half)
    x2 = min(img_w, x1 + size_px)
    y2 = min(img_h, y1 + size_px)

    cropped = img[y1:y2, x1:x2]
    if cropped.shape[0] != size_px or cropped.shape[1] != size_px:
        cropped = cv2.resize(cropped, (size_px, size_px), interpolation=cv2.INTER_AREA)

    return cropped


def run():
    with open(os.path.join(file_dir, 'preferences.json'), 'r', encoding='utf-8') as f:
        prefs = json.loads(f.read())

    if not os.path.isdir(prefs['dir']):
        os.mkdir(prefs['dir'])

    if (prefs['tl'] == '') or (prefs['br'] == '') or (prefs['zoom'] == ''):
        print('Select input mode:')
        print('1 - Top-left and bottom-right corners')
        print('2 - Centered square (single lat/lon + size in pixels)')
        mode = input('Mode (1/2, default 1): ').strip() or '1'

        if mode == '2':
            messages = [
                'Center coordinates (lat, lon): ',
                'Zoom level: ',
                'Square size in pixels (e.g. 2040): '
            ]
            inputs = take_input(messages)
            if inputs is None:
                return
            center_str, zoom_str, size_str = inputs
            lat_c, lon_c = re.findall(r'[+-]?\d*\.\d+|d+', center_str)
            zoom = int(zoom_str)
            size_px = int(size_str)
            channels = int(prefs['channels'])
            tile_size = int(prefs['tile_size'])

            lat_c = float(lat_c)
            lon_c = float(lon_c)

            img = download_centered_square(
                lat_c, lon_c, zoom, size_px, prefs['url'],
                prefs['headers'], tile_size, channels
            )
        else:
            messages = ['Top-left corner: ', 'Bottom-right corner: ', 'Zoom level: ']
            inputs = take_input(messages)
            if inputs is None:
                return
            prefs['tl'], prefs['br'], prefs['zoom'] = inputs

            lat1, lon1 = re.findall(r'[+-]?\d*\.\d+|d+', prefs['tl'])
            lat2, lon2 = re.findall(r'[+-]?\d*\.\d+|d+', prefs['br'])

            zoom = int(prefs['zoom'])
            channels = int(prefs['channels'])
            tile_size = int(prefs['tile_size'])
            lat1 = float(lat1)
            lon1 = float(lon1)
            lat2 = float(lat2)
            lon2 = float(lon2)

            img = download_image(lat1, lon1, lat2, lon2, zoom, prefs['url'],
                prefs['headers'], tile_size, channels)
    else:
        lat1, lon1 = re.findall(r'[+-]?\d*\.\d+|d+', prefs['tl'])
        lat2, lon2 = re.findall(r'[+-]?\d*\.\d+|d+', prefs['br'])

        zoom = int(prefs['zoom'])
        channels = int(prefs['channels'])
        tile_size = int(prefs['tile_size'])
        lat1 = float(lat1)
        lon1 = float(lon1)
        lat2 = float(lat2)
        lon2 = float(lon2)

        img = download_image(lat1, lon1, lat2, lon2, zoom, prefs['url'],
            prefs['headers'], tile_size, channels)

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    name = f'img_{timestamp}.png'
    cv2.imwrite(os.path.join(prefs['dir'], name), img)
    print(f'Saved as {name}')


if os.path.isfile(prefs_path):
    run()
else:
    with open(prefs_path, 'w', encoding='utf-8') as f:
        json.dump(default_prefs, f, indent=2, ensure_ascii=False)

    print(f'Preferences file created in {prefs_path}')
