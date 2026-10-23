"""Privacy-first opt-in location providers.

Native CoreLocation auto-detection is *unsupported*, even on macOS: a Swift
CLI probe showed location services enabled but no NSLocationUsageDescription
in its bundle, and no real permission grant or denied path could be verified.
Apple's guidance requires an appropriate usage description and permission
request before collecting sensitive location data:
https://developer.apple.com/documentation/corelocation/configuring-your-app-to-use-location-services
https://developer.apple.com/documentation/corelocation/requesting-authorization-to-use-location-services

Integration (from an explicit voice/menu action only)::

    result = ManualLocationProvider(user_lat, user_lon).request_location()
    if result["error"] is None:
        hud.set_location(result["lat"], result["lon"], "Your chosen area", 500)

Manual input is a *user-chosen point*, not verified current device position;
accuracy_m is None. Never auto-populate coordinates from IP or a cached
sample. Do not upload coordinates here. Do not call any provider from a GUI
frame loop; even though these providers return immediately, the UI owns the
explicit action and any future native implementation must run off the loop.
"""


import math
from numbers import Real


class DeviceLocationProvider:
    """Native detection is unavailable until a verified, permission-capable app exists.

    Call request_location(opt_in=True) only from an explicit user menu/voice
    action. This class never starts CoreLocation or requests permissions. The
    unsupported result is immediate, including on the GUI/main thread.
    """

    def __init__(self):
        self.state = "idle"

    def request_location(self, *, opt_in=False, timeout_s=5.0):
        if (not isinstance(timeout_s, Real) or isinstance(timeout_s, bool)
                or not math.isfinite(timeout_s) or timeout_s <= 0):
            raise ValueError("timeout_s must be positive and finite")
        self.state = "unsupported" if opt_in else "idle"
        return {"lat": None, "lon": None, "accuracy_m": None,
                "error": "unsupported" if opt_in else "consent_required"}


class ManualLocationProvider:
    def __init__(self, lat, lon):
        if any(not isinstance(value, Real) or isinstance(value, bool)
               for value in (lat, lon)):
            raise ValueError("Coordinates must be real numbers")
        if not math.isfinite(lat) or not math.isfinite(lon) or not (-90 <= lat <= 90 and -180 <= lon <= 180):
            raise ValueError("Coordinates out of range")
        self.lat = lat
        self.lon = lon

    def request_location(self):
        return {"lat": self.lat, "lon": self.lon, "accuracy_m": None, "error": None}
