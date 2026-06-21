"""
Download images from thispersondoesnotexist.com for fine-tuning PixelGuard's
GAN-fake detector against this specific generator's artifact fingerprint.

Usage:
    python scrape_tpdne.py --count 300 --out data/tpdne_fakes

Run this on your own machine (not in a sandboxed environment) since it needs
unrestricted internet access to thispersondoesnotexist.com.
"""

import argparse
import os
import time
import requests

URL = "https://thispersondoesnotexist.com"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/124.0.0.0 Safari/537.36"
}


def download_images(count: int, out_dir: str, delay: float = 1.0):
    os.makedirs(out_dir, exist_ok=True)

    existing = len([f for f in os.listdir(out_dir) if f.endswith(".jpg")])
    print(f"Found {existing} existing images in {out_dir}, starting from there.")

    success = 0
    attempts = 0
    max_attempts = count * 3  # allow for some failed requests

    while success < count and attempts < max_attempts:
        attempts += 1
        try:
            resp = requests.get(URL, headers=HEADERS, timeout=10)
            resp.raise_for_status()

            if not resp.content or len(resp.content) < 1000:
                print(f"  [skip] attempt {attempts}: response too small, retrying")
                time.sleep(delay)
                continue

            idx = existing + success
            filepath = os.path.join(out_dir, f"tpdne_{idx:04d}.jpg")
            with open(filepath, "wb") as f:
                f.write(resp.content)

            success += 1
            if success % 10 == 0:
                print(f"  Downloaded {success}/{count}")

        except requests.RequestException as e:
            print(f"  [error] attempt {attempts}: {e}, retrying after delay")

        time.sleep(delay)  # be polite, avoid hammering the server

    print(f"\nDone. Downloaded {success} new images to {out_dir}")
    if success < count:
        print(f"Warning: only got {success}/{count} requested — "
              f"consider re-running to top up, or increasing --delay if "
              f"you were hitting errors.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=300,
                         help="Number of images to download")
    parser.add_argument("--out", type=str, default="data/tpdne_fakes",
                         help="Output directory")
    parser.add_argument("--delay", type=float, default=1.0,
                         help="Delay in seconds between requests (be polite)")
    args = parser.parse_args()

    download_images(args.count, args.out, args.delay)