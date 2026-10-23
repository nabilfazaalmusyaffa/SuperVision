import tempfile
import stat
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import Mock

import cv2
import numpy as np

from hud.location_map import LocationMapHUD


class LocationMapTests(unittest.TestCase):
    def test_no_location_displays_off_state_without_requesting_tiles(self):
        fetch = Mock()
        hud = LocationMapHUD(fetch=fetch)
        image = np.zeros((720, 1280, 3), np.uint8)
        output = hud.render(image.copy())
        self.assertGreater(np.count_nonzero(output[60:280, 940:1255]), 1000)
        self.assertEqual(np.count_nonzero(output[:, :935]), 0)
        self.assertEqual(hud.state, 'inactive')
        fetch.assert_not_called()

    def test_supplied_position_is_quantized_before_tile_selection_and_not_exposed(self):
        hud = LocationMapHUD(fetch=lambda req, timeout: (_ for _ in ()).throw(OSError('offline')))
        hud.set_location(37.77491, -122.41939, '123 Exact Home Street', 5)
        self.assertEqual(hud._location[:2], (37.8, -122.4))
        self.assertNotIn('123 Exact Home Street', repr(hud._location))
        image = np.zeros((720, 1280, 3), np.uint8)
        result = hud.render(image)
        self.assertGreater(np.count_nonzero(result[60:280, 940:1255]), 1000)
        self.assertGreaterEqual(hud._location[3], 10_000)

    def test_source_label_distinguishes_ip_estimate_from_user_chosen_area(self):
        hud = LocationMapHUD(fetch=Mock())
        hud.set_location(41.9, 12.5, 'IP ESTIMATE')
        self.assertEqual(hud._location[2], 'IP ESTIMATE')
        hud.set_location(41.9, 12.5, 'USER CHOSEN')
        self.assertEqual(hud._location[2], 'USER CHOSEN')
        hud.set_location(41.9, 12.5, '123 Exact Home Street')
        self.assertEqual(hud._location[2], 'AREA APPROX')

    def test_invalid_coordinates_are_rejected_without_network(self):
        fetch = Mock()
        hud = LocationMapHUD(fetch=fetch)
        for lat, lon in ((91, 0), (0, 181), (float('nan'), 0), (0, float('inf'))):
            with self.assertRaises(ValueError):
                hud.set_location(lat, lon, 'Place', 500)
        self.assertEqual(hud.state, 'inactive')
        fetch.assert_not_called()


    def test_offline_location_is_labelled_approximate_and_stays_in_panel(self):
        hud = LocationMapHUD(fetch=lambda request, timeout: (_ for _ in ()).throw(OSError('offline')))
        hud.set_location(37.7, -122.4, 'Nearby area', 800)
        image = np.zeros((720, 1280, 3), np.uint8)
        result = hud.render(image.copy())
        self.assertGreater(np.count_nonzero(result[60:280, 940:1255]), 1000)
        self.assertEqual(np.count_nonzero(result[:55]), 0)
        self.assertEqual(np.count_nonzero(result[:, :935]), 0)
        self.assertTrue(np.array_equal(image, np.zeros_like(image)))

    def test_visible_tiles_are_requested_asynchronously_with_identification_and_cached(self):
        png = cv2.imencode('.png', np.full((256, 256, 3), 120, np.uint8))[1].tobytes()
        requests = []
        completed = threading.Event()

        class Response:
            headers = {'Cache-Control': 'public, max-age=604800', 'ETag': '"test"'}
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def read(self): return png

        def fetch(request, timeout):
            requests.append(request)
            completed.set()
            return Response()

        with tempfile.TemporaryDirectory() as cache:
            hud = LocationMapHUD(fetch=fetch, cache_dir=cache)
            hud.set_location(37.7, -122.4, 'Nearby', 500)
            blank = np.zeros((720, 1280, 3), np.uint8)
            start = time.monotonic()
            hud.render(blank.copy())
            self.assertLess(time.monotonic() - start, 0.15)
            self.assertTrue(completed.wait(2))
            hud.wait_for_idle(3)
            self.assertTrue(requests)
            self.assertLessEqual(len(requests), 6)
            self.assertTrue(all(r.full_url.startswith('https://tile.openstreetmap.org/') for r in requests))
            self.assertTrue(all('SpiderVision' in r.get_header('User-agent') for r in requests))
            self.assertGreater(np.count_nonzero(hud.render(blank.copy())[90:220, 960:1230]), 0)
            count = len(requests)
            second = LocationMapHUD(fetch=fetch, cache_dir=cache)
            second.set_location(37.7, -122.4, 'Nearby', 500)
            second.render(blank.copy())
            second.wait_for_idle(3)
            self.assertEqual(len(requests), count)
            self.assertEqual(second.state, 'ready')

    def test_tile_cache_permissions_do_not_expose_the_area_to_other_users(self):
        png = cv2.imencode('.png', np.full((256, 256, 3), 100, np.uint8))[1].tobytes()
        class Response:
            headers = {'Cache-Control': 'public, max-age=604800'}
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def read(self): return png
        with tempfile.TemporaryDirectory() as cache:
            hud = LocationMapHUD(fetch=lambda request, timeout: Response(), cache_dir=cache)
            hud.set_location(41.9, 12.5, 'Test area')
            hud.render(np.zeros((720, 1280, 3), np.uint8))
            hud.wait_for_idle(3)
            paths = list(Path(cache).rglob('*.png'))
            self.assertTrue(paths)
            for path in paths:
                self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
                self.assertEqual(stat.S_IMODE(path.parent.stat().st_mode), 0o700)

    def test_expired_cache_is_revalidated_conditionally(self):
        png = cv2.imencode('.png', np.full((256, 256, 3), 80, np.uint8))[1].tobytes()
        requests = []
        clock = [1000000.0]

        class Response:
            headers = {'ETag': '"cache-version"'}
            status = 200
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def read(self): return png

        def fetch(request, timeout):
            requests.append(request)
            return Response()

        with tempfile.TemporaryDirectory() as cache:
            for now in (1000000.0, 1000000.0 + 6*86400, 1000000.0 + 8*86400):
                clock[0] = now
                hud = LocationMapHUD(fetch=fetch, cache_dir=cache, clock=lambda: clock[0])
                hud.set_location(0, 0, 'Area', 1000)
                hud.render(np.zeros((720, 1280, 3), np.uint8))
                hud.wait_for_idle(3)
                if now == 1000000.0: first_count = len(requests)
                elif now == 1000000.0 + 6*86400: self.assertEqual(len(requests), first_count)
            self.assertGreater(len(requests), first_count)
            self.assertTrue(any(r.get_header('If-none-match') == '"cache-version"' for r in requests[first_count:]))

    def test_http_expires_and_last_modified_are_honoured(self):
        png = cv2.imencode('.png', np.full((256, 256, 3), 80, np.uint8))[1].tobytes()
        requests = []
        now = [1_000_000.0]
        from email.utils import formatdate

        class Response:
            status = 200
            headers = {'Expires': formatdate(now[0] + 10*86400, usegmt=True),
                       'Last-Modified': 'Wed, 01 Jan 2020 00:00:00 GMT'}
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def read(self): return png

        def fetch(request, timeout):
            requests.append(request)
            return Response()

        with tempfile.TemporaryDirectory() as cache:
            for timestamp in (1_000_000.0, 1_000_000.0 + 8*86400, 1_000_000.0 + 11*86400):
                now[0] = timestamp
                hud = LocationMapHUD(fetch=fetch, cache_dir=cache, clock=lambda: now[0])
                hud.set_location(0, 0, 'Area', 1000)
                hud.render(np.zeros((720, 1280, 3), np.uint8))
                hud.wait_for_idle(3)
                if timestamp == 1_000_000.0: first_count = len(requests)
                elif timestamp == 1_000_000.0 + 8*86400: self.assertEqual(len(requests), first_count)
            self.assertGreater(len(requests), first_count)
            self.assertTrue(any(r.get_header('If-modified-since') == 'Wed, 01 Jan 2020 00:00:00 GMT'
                                for r in requests[first_count:]))


if __name__ == '__main__':
    unittest.main()
