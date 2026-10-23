"""Opt-in coarse area estimate from the network's public IP, never GPS.

Only call request_location(opt_in=True) from a worker after explaining that an
external service sees the public IP. No request is made at import or start-up.
The returned data excludes IP, postal code and street information.
"""
import json
import math
from urllib.request import Request, urlopen


class IPAreaProvider:
    ENDPOINT = 'https://ipapi.co/json/'
    USER_AGENT = 'SpiderVision-IPArea/1.0 (opt-in coarse-area lookup)'

    def __init__(self, fetch=None):
        self._fetch = fetch or urlopen

    def request_location(self, *, opt_in=False):
        if not opt_in:
            return {'lat': None, 'lon': None, 'accuracy_m': None,
                    'error': 'consent_required'}
        request = Request(self.ENDPOINT, headers={'User-Agent': self.USER_AGENT})
        try:
            with self._fetch(request, timeout=5) as response:
                data = json.loads(response.read(16_384).decode('utf-8'))
            if data.get('error'):
                raise ValueError('provider error')
            lat, lon = float(data['latitude']), float(data['longitude'])
            if not math.isfinite(lat) or not math.isfinite(lon) or not (-90 <= lat <= 90 and -180 <= lon <= 180):
                raise ValueError('invalid area')
            return {'lat': round(lat, 1), 'lon': round(lon, 1),
                    'accuracy_m': None, 'error': None}
        except (OSError, ValueError, TypeError, KeyError, AttributeError):
            return {'lat': None, 'lon': None, 'accuracy_m': None,
                    'error': 'unavailable'}
