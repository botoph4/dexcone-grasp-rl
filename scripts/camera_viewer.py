"""Camera stream window in its own process (plain python).

Reads length-prefixed JPEG frames from stdin (4-byte big-endian length +
payload) and shows them with cv2.  Running the window in a child process
lets it coexist with the MuJoCo popup: the popup needs mjpython's Cocoa
main thread, cv2 needs a plain interpreter's main thread -- two
processes, one window each.  Esc/q or closed stdin exits.
"""
import struct
import sys

import cv2
import numpy as np

WINDOW = "p24 camera"


def main() -> int:
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    try:
        while True:
            header = sys.stdin.buffer.read(4)
            if len(header) < 4:
                break
            (size,) = struct.unpack(">I", header)
            payload = sys.stdin.buffer.read(size)
            if len(payload) < size:
                break
            frame = cv2.imdecode(np.frombuffer(payload, np.uint8),
                                 cv2.IMREAD_COLOR)
            if frame is None:
                continue
            cv2.imshow(WINDOW, frame)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):  # Esc or q
                break
    finally:
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
