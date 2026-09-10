"""End-to-end check against a running faceshim: register -> recognize -> delete.

    python selfcheck.py --url http://localhost:5002 --image face.jpg
    python selfcheck.py --image auto      # inside the container: uses insightface's bundled test photo

Exit code 0 when the round trip works, 1 otherwise. Standard library only.
"""

import argparse
import json
import sys
import urllib.request
import uuid


TIMEOUT = 60


def post(url: str, fields: dict, file: tuple | None = None) -> dict:
    boundary = uuid.uuid4().hex
    body = b""
    for k, v in fields.items():
        body += f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode()
    if file:
        name, data = file
        body += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{name}\"\r\n"
                 f"Content-Type: image/jpeg\r\n\r\n").encode() + data + b"\r\n"
    body += f"--{boundary}--\r\n".encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        return json.load(r)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:5000")
    ap.add_argument("--image", required=True, help="JPEG with one clear face, or 'auto'")
    ap.add_argument("--timeout", type=float, default=60, help="seconds per request (raise under emulation)")
    args = ap.parse_args()
    global TIMEOUT
    TIMEOUT = args.timeout

    if args.image == "auto":
        import cv2
        from insightface.data import get_image
        ok, buf = cv2.imencode(".jpg", get_image("t1"))
        assert ok
        data = buf.tobytes()
    else:
        data = open(args.image, "rb").read()

    user = f"selfcheck-{uuid.uuid4().hex[:6]}"
    base = args.url.rstrip("/") + "/v1/vision/face"

    health = json.load(urllib.request.urlopen(args.url, timeout=TIMEOUT))
    assert health.get("success"), health
    print("health:", health)

    # Leftovers from an aborted run would tie with the new user (identical embedding); clear them first.
    for stale in [u for u in health.get("users", []) if u.startswith("selfcheck-")]:
        post(f"{base}/delete", {"userid": stale})
        print("removed stale user:", stale)

    reg = post(f"{base}/register", {"userid": user}, ("face.jpg", data))
    assert reg.get("success"), reg
    print("register:", reg)

    rec = post(f"{base}/recognize", {}, ("face.jpg", data))
    assert rec.get("success"), rec
    hits = [p for p in rec["predictions"] if p["userid"] == user]
    print("recognize:", json.dumps(rec, indent=1))
    assert hits, "enrolled face not recognised in the same image"
    assert hits[0]["confidence"] > 0.9, f"self-match confidence unexpectedly low: {hits[0]}"

    lst = post(f"{base}/list", {})
    assert user in lst.get("faces", []), lst

    dele = post(f"{base}/delete", {"userid": user})
    assert dele.get("success"), dele
    lst = post(f"{base}/list", {})
    assert user not in lst.get("faces", []), lst

    print(f"OK - round trip works, self-match confidence {hits[0]['confidence']:.3f} (similarity {hits[0]['similarity']:.3f})")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except AssertionError as e:
        print("FAIL:", e)
        sys.exit(1)
