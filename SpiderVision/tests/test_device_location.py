"""Opt-in location provider contracts; no OS permission request in tests."""

import unittest

from location.device import DeviceLocationProvider, ManualLocationProvider


class ManualLocationTests(unittest.TestCase):
    def test_explicit_coordinates_are_returned_without_guessing_accuracy(self):
        result = ManualLocationProvider(37.5, -122.2).request_location()
        self.assertEqual(result, {"lat": 37.5, "lon": -122.2, "accuracy_m": None, "error": None})

    def test_invalid_coordinates_are_rejected(self):
        for lat, lon in ((91, 0), (-91, 0), (0, 181), (0, -181),
                         (float("nan"), 0), (0, float("inf")), ("37", 2), (True, 0)):
            with self.subTest(lat=lat, lon=lon), self.assertRaises(ValueError):
                ManualLocationProvider(lat, lon)


class DeviceLocationTests(unittest.TestCase):
    def test_no_explicit_action_does_not_attempt_detection(self):
        provider = DeviceLocationProvider()
        self.assertEqual(provider.request_location(),
                         {"lat": None, "lon": None, "accuracy_m": None, "error": "consent_required"})
        self.assertEqual(provider.state, "idle")

    def test_explicit_action_reports_unsupported_not_a_position(self):
        provider = DeviceLocationProvider()
        self.assertEqual(provider.request_location(opt_in=True),
                         {"lat": None, "lon": None, "accuracy_m": None, "error": "unsupported"})
        self.assertEqual(provider.state, "unsupported")

    def test_subsequent_unconsented_call_remains_idle(self):
        provider = DeviceLocationProvider()
        provider.request_location(opt_in=True)
        self.assertEqual(provider.request_location(),
                         {"lat": None, "lon": None, "accuracy_m": None, "error": "consent_required"})
        self.assertEqual(provider.state, "idle")


    def test_timeout_must_be_positive_and_finite(self):
        provider = DeviceLocationProvider()
        for timeout in (0, -1, float("nan"), float("inf"), "5", True):
            with self.subTest(timeout=timeout), self.assertRaises(ValueError):
                provider.request_location(opt_in=True, timeout_s=timeout)
        self.assertEqual(provider.state, "idle")


if __name__ == "__main__":
    unittest.main()
