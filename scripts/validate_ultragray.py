import argparse
from pathlib import Path

import numpy as np


EXPECTED = {
    "pig_shoulder_v2": {
        "images_train.npy": ((219, 732, 746, 3), np.uint8),
        "images_val.npy": ((104, 732, 746, 3), np.uint8),
        "poses_train.npy": ((219, 4, 4), np.float64),
        "poses_val.npy": ((104, 4, 4), np.float64),
    },
    "spine_phantom": {
        "images_train.npy": ((1845, 658, 372, 3), np.uint8),
        "images_val_v2.npy": ((93, 658, 372, 3), np.uint8),
        "poses_train.npy": ((1845, 4, 4), np.float64),
        "poses_val_v2.npy": ((93, 4, 4), np.float64),
    },
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()

    failed = False
    for dataset, files in EXPECTED.items():
        for name, (shape, dtype) in files.items():
            path = args.root / dataset / name
            if not path.exists():
                print(f"MISSING {path}")
                failed = True
                continue
            array = np.load(path, mmap_mode="r")
            ok = array.shape == shape and array.dtype == dtype
            print(f"{'OK' if ok else 'INVALID'} {path}: {array.shape} {array.dtype}")
            failed |= not ok
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
