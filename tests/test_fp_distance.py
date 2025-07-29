import base64
from near_duplicate_detector import fingerprint_distance


def test_fingerprint_distance_identical_base64():
    data = b'\x00\x01\x02\x03\x04\x05'
    s = base64.urlsafe_b64encode(data).decode('ascii').rstrip('=')
    assert fingerprint_distance(s, s) == 0.0

