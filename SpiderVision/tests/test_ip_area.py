"""Tests for explicit opt-in, approximate network-area lookup."""
import io
import json
import unittest
from unittest.mock import Mock

from location.ip_area import IPAreaProvider


class IPAreaTests(unittest.TestCase):
    def test_no_consent_means_no_network_request(self):
        fetch = Mock()
        provider = IPAreaProvider(fetch=fetch)
        self.assertEqual(provider.request_location()['error'], 'consent_required')
        fetch.assert_not_called()

    def test_approved_lookup_returns_only_coarse_region_not_ip_or_postal(self):
        payload = {'ip': '198.51.100.20', 'postal': '94107', 'city': 'Sample City',
                   'latitude': 37.7697, 'longitude': -122.3933}
        seen = []
        class Response:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def read(self, size): return json.dumps(payload).encode()
        def fetch(request, timeout):
            seen.append(request)
            self.assertLessEqual(timeout, 5)
            return Response()
        result = IPAreaProvider(fetch=fetch).request_location(opt_in=True)
        self.assertEqual(result, {'lat': 37.8, 'lon': -122.4,
                                  'accuracy_m': None, 'error': None})
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0].full_url, 'https://ipapi.co/json/')
        self.assertIn('SpiderVision', seen[0].get_header('User-agent'))
        self.assertNotIn('198.51.100.20', repr(result))

    def test_lookup_failure_or_bad_coordinates_never_fabricates_location(self):
        for payload in ({'error': True, 'reason': 'Rate limited'},
                        {'latitude': 900, 'longitude': 0},
                        {'latitude': 'none', 'longitude': 0}):
            class Response:
                def __enter__(self): return self
                def __exit__(self, *args): pass
                def read(self, size): return json.dumps(payload).encode()
            with self.subTest(payload=payload):
                result = IPAreaProvider(fetch=lambda *_a, **_kw: Response()).request_location(opt_in=True)
                self.assertIsNone(result['lat'])
                self.assertEqual(result['error'], 'unavailable')
        result = IPAreaProvider(fetch=lambda *_a, **_kw: (_ for _ in ()).throw(OSError('offline'))).request_location(opt_in=True)
        self.assertIsNone(result['lat'])
        self.assertEqual(result['error'], 'unavailable')


if __name__ == '__main__':
    unittest.main()
