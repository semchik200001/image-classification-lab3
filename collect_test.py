"""Скачивание тестовой выборки по списку data/test/sources.csv.

Изображения отобраны вручную на Wikimedia Commons и размечены по классам.
В sources.csv для каждого файла указаны класс, страница на Commons, автор
и лицензия. Изображения уже лежат в репозитории, скрипт нужен, чтобы
выборку можно было собрать заново:

    python3 collect_test.py            # скачать недостающие файлы
    python3 collect_test.py --force    # скачать все заново
"""

import argparse
import csv
import time
import urllib.request
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageOps

ROOT = Path("data/test")
UA = {"User-Agent": "itmo-lab3-test-collector/1.0 (https://github.com/semchik200001)"}
MAX_SIDE = 512


def download(url, retries=4):
    for i in range(retries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
                return r.read()
        except OSError as e:
            if i == retries - 1:
                raise
            print(f"  повтор через {3 * (i + 1)} с: {e}")
            time.sleep(3 * (i + 1))


def main():
    ap = argparse.ArgumentParser(description="Скачать тестовую выборку с Wikimedia Commons")
    ap.add_argument("--force", action="store_true", help="перекачать уже скачанные файлы")
    args = ap.parse_args()
    with open(ROOT / "sources.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        dst = ROOT / row["class"] / row["file"]
        if dst.exists() and not args.force:
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        img = ImageOps.exif_transpose(Image.open(BytesIO(download(row["url"])))).convert("RGB")
        img.thumbnail((MAX_SIDE, MAX_SIDE))
        img.save(dst, quality=90)
        print(f"{dst}  ←  {row['page']}")
        time.sleep(0.5)
    print(f"В выборке {len(rows)} изображений")


if __name__ == "__main__":
    main()
