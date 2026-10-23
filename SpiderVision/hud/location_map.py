"""Opt-in, nonblocking OpenStreetMap overlay for an approximate supplied location.

The caller must gate render() on a verified face and an inactive menu. Calling
set_location follows user consent to show/send a rounded area to the tile host;
this module never discovers a device location or stores exact coordinates.
Cached OSM tile filenames do reveal an approximate region until cleared.
"""

import json
import math
import os
import re
import threading
import time
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import cv2
import numpy as np


ZOOM = 10  # regional view; tile requests should never identify a street address
TILE_SIZE = 256
CACHE_SECONDS = 7 * 86400
USER_AGENT = 'SpiderVision-LocationMapHUD/1.0 (OpenStreetMap tile viewer)'
# Entire panel fits the top-right of a 1280x720 BGR image.
PANEL = (940, 60, 1255, 280)
MAP = (950, 91, 1245, 218)


class LocationMapHUD:
    """UI-only approximate map. ``state``: inactive/loading/ready/unavailable.

    render() returns the same BGR array (painted in place) and performs no I/O.
    The first render starts a daemon worker for only tiles intersecting the view.
    Set location to None to clear it; call render only while the parent authorizes
    display, so hidden panels never initiate tile requests.
    """

    def __init__(self, *, fetch=None, cache_dir=None, clock=None):
        self._fetch = fetch or urlopen
        self._clock = clock or time.time
        self._cache = Path(cache_dir) if cache_dir is not None else Path.home() / '.cache' / 'spidervision' / 'osm-tiles'
        self._location = None
        self._tiles = {}
        self._worker = None
        self._generation = 0
        self._lock = threading.Lock()
        self.state = 'inactive'

    def set_location(self, lat, lon, label='', accuracy_m=None):
        """Supply a user-authorized approximate area (never infer from camera)."""
        if lat is None and lon is None:
            with self._lock:
                self._generation += 1
                self._location = None
                self._tiles = {}
                self.state = 'inactive'
            return
        try:
            latitude, longitude = float(lat), float(lon)
        except (ValueError, TypeError) as exc:
            raise ValueError('invalid latitude/longitude') from exc
        if not all(math.isfinite(v) for v in (latitude, longitude)) or not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
            raise ValueError('latitude/longitude out of bounds')
        if accuracy_m is not None:
            try:
                accuracy_m = float(accuracy_m)
            except (ValueError, TypeError) as exc:
                raise ValueError('invalid accuracy') from exc
            if not math.isfinite(accuracy_m) or accuracy_m < 0:
                raise ValueError('invalid accuracy')
        # Quantize before storing or deriving network tile keys; only allow
        # fixed source labels, never an arbitrary address from the caller.
        coarse_lat = round(latitude * 10) / 10
        coarse_lon = round(longitude * 10) / 10
        coarse_accuracy = max(10_000, accuracy_m or 0)
        with self._lock:
            self._generation += 1
            source = label if label in ('IP ESTIMATE', 'USER CHOSEN') else 'AREA APPROX'
            self._location = (coarse_lat, coarse_lon, source, coarse_accuracy)
            self._tiles = {}
            self.state = 'loading'

    @staticmethod
    def _position(lat, lon):
        lat = max(-85.05112878, min(85.05112878, lat))
        scale = TILE_SIZE * (1 << ZOOM)
        return ((lon + 180) / 360 * scale,
                (1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * scale)

    def _visible_tiles(self, location):
        px, py = self._position(*location[:2])
        left = px - (MAP[2] - MAP[0]) / 2
        top = py - (MAP[3] - MAP[1]) / 2
        result = []
        x_first, x_last = math.floor(left / TILE_SIZE), math.floor((left + MAP[2] - MAP[0] - 1) / TILE_SIZE)
        y_first, y_last = math.floor(top / TILE_SIZE), math.floor((top + MAP[3] - MAP[1] - 1) / TILE_SIZE)
        n = 1 << ZOOM
        for ty in range(y_first, y_last + 1):
            for tx in range(x_first, x_last + 1):
                if 0 <= ty < n:
                    result.append(((ZOOM, tx % n, ty), int(round(MAP[0] + tx*TILE_SIZE-left)),
                                   int(round(MAP[1] + ty*TILE_SIZE-top))))
        return result

    def _read_tile(self, key):
        z, x, y = key
        stem = self._cache / str(z) / str(x) / str(y)
        for directory in (self._cache, self._cache / str(z), stem.parent):
            try:
                directory.mkdir(mode=0o700, parents=True, exist_ok=True)
                directory.chmod(0o700)
            except OSError:
                pass
        meta = {}
        try:
            meta = json.loads(stem.with_suffix('.json').read_text())
            raw = stem.with_suffix('.png').read_bytes()
            cached = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
            if cached is None or cached.shape[:2] != (256, 256):
                cached = None
        except (OSError, ValueError, TypeError):
            cached = None
        if cached is not None and self._clock() < meta.get('expires', 0):
            return cached
        headers = {'User-Agent': USER_AGENT}
        if cached is not None and meta.get('etag'):
            headers['If-None-Match'] = meta['etag']
        if cached is not None and meta.get('last_modified'):
            headers['If-Modified-Since'] = meta['last_modified']
        request = Request(f'https://tile.openstreetmap.org/{z}/{x}/{y}.png', headers=headers)
        try:
            with self._fetch(request, timeout=3) as response:
                if getattr(response, 'status', 200) == 304:
                    if cached is None:
                        return None
                    raw = None
                else:
                    raw = response.read()
                response_headers = response.headers
        except HTTPError as exc:
            if exc.code != 304 or cached is None:
                return cached
            raw, response_headers = None, exc.headers
        except (OSError, ValueError):
            return cached
        if raw is not None:
            image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
            if image is None or image.shape[:2] != (256, 256):
                return cached
        else:
            image = cached
        directive = re.search(r'max-age=(\d+)', response_headers.get('Cache-Control', ''), re.I)
        lifetime = int(directive.group(1)) if directive else CACHE_SECONDS
        if not directive and response_headers.get('Expires'):
            try:
                lifetime = max(0, parsedate_to_datetime(response_headers['Expires']).timestamp() - self._clock())
            except (ValueError, TypeError, OverflowError):
                pass
        # At least seven days when no usable freshness metadata is available.
        expiry = self._clock() + lifetime
        try:
            stem.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            if raw is not None:
                with os.fdopen(os.open(stem.with_suffix('.png'), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), 'wb') as file:
                    file.write(raw)
                stem.with_suffix('.png').chmod(0o600)
            with os.fdopen(os.open(stem.with_suffix('.json'), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), 'w') as file:
                file.write(json.dumps({
                'expires': expiry, 'etag': response_headers.get('ETag', meta.get('etag')),
                'last_modified': response_headers.get('Last-Modified', meta.get('last_modified'))}))
            stem.with_suffix('.json').chmod(0o600)
        except OSError:
            pass  # In-memory tile still works with a read-only cache.
        return image

    def _load(self, generation, tiles):
        for key, _, _ in tiles:
            with self._lock:
                if self._generation != generation:
                    return
            image = self._read_tile(key)
            with self._lock:
                if self._generation != generation:
                    return
                if image is not None:
                    self._tiles[key] = image
        with self._lock:
            if self._generation == generation:
                self.state = 'ready' if self._tiles else 'unavailable'

    def wait_for_idle(self, timeout=3):
        """Test/diagnostic helper; never call on the camera render thread."""
        worker = self._worker
        if worker is not None:
            worker.join(timeout)

    def render(self, bgr):
        """Paint map and explicit approximate-location legend, without blocking I/O."""
        with self._lock:
            location, generation = self._location, self._generation
            tiles = dict(self._tiles)
            state = self.state
            if location is not None and (self._worker is None or not self._worker.is_alive()):
                if state == 'loading':
                    visible = self._visible_tiles(location)
                    self._worker = threading.Thread(target=self._load, args=(generation, visible), daemon=True)
                    self._worker.start()
        if bgr.shape[0] < PANEL[3] or bgr.shape[1] < PANEL[2]:
            return bgr
        x1, y1, x2, y2 = PANEL
        cv2.rectangle(bgr, (x1, y1), (x2, y2), (12, 25, 33), -1)
        cv2.rectangle(bgr, (x1, y1), (x2, y2), (85, 203, 242), 1)
        cv2.putText(bgr, 'LOCAL AREA  /  APPROXIMATE', (x1+9, y1+20), cv2.FONT_HERSHEY_SIMPLEX, .42, (180, 230, 248), 1, cv2.LINE_AA)
        cv2.rectangle(bgr, (MAP[0], MAP[1]), (MAP[2], MAP[3]), (28, 42, 49), -1)
        if location is None:
            cv2.putText(bgr, 'LOCATION OFF / NO LOCATION', (MAP[0]+13, MAP[1]+68),
                        cv2.FONT_HERSHEY_SIMPLEX, .48, (203, 210, 221), 1, cv2.LINE_AA)
            cv2.putText(bgr, 'No area shared', (x1+10, 238), cv2.FONT_HERSHEY_SIMPLEX,
                        .40, (214, 224, 234), 1, cv2.LINE_AA)
            return bgr
        for key, tx, ty in self._visible_tiles(location):
            tile = tiles.get(key)
            if tile is None:
                continue
            ax, ay = max(MAP[0], tx), max(MAP[1], ty)
            bx, by = min(MAP[2], tx+256), min(MAP[3], ty+256)
            if ax < bx and ay < by:
                bgr[ay:by, ax:bx] = tile[ay-ty:by-ty, ax-tx:bx-tx]
        if state != 'ready':
            text = 'MAP UNAVAILABLE / OFFLINE' if state == 'unavailable' else 'MAP LOADING...'
            cv2.putText(bgr, text, (MAP[0]+20, MAP[1]+23), cv2.FONT_HERSHEY_SIMPLEX, .48, (203, 210, 221), 1, cv2.LINE_AA)
        # Schematic area indicator; no exact-position pin (not to scale).
        cx, cy = (MAP[0]+MAP[2])//2, (MAP[1]+MAP[3])//2
        cv2.circle(bgr, (cx, cy), 26, (237, 205, 83), 2, cv2.LINE_AA)
        label = location[2][:25] or 'Supplied area'
        accuracy = f'~{location[3]/1000:.0f} km area (min)'
        cv2.putText(bgr, f'{label}  /  {accuracy}'[:49], (x1+10, 238), cv2.FONT_HERSHEY_SIMPLEX, .36, (214, 224, 234), 1, cv2.LINE_AA)
        # Hershey has no © glyph; draw its circled C explicitly.
        cv2.circle(bgr, (x1+16, 261), 6, (200, 214, 229), 1, cv2.LINE_AA)
        cv2.putText(bgr, 'C', (x1+12, 265), cv2.FONT_HERSHEY_SIMPLEX, .28, (200, 214, 229), 1, cv2.LINE_AA)
        cv2.putText(bgr, 'OpenStreetMap contributors', (x1+25, 265), cv2.FONT_HERSHEY_SIMPLEX, .38, (200, 214, 229), 1, cv2.LINE_AA)
        return bgr
