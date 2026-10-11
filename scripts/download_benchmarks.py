from pathlib import Path

import gdown


OUT = Path("datasets/ultragray")
OUT.mkdir(parents=True, exist_ok=True)

FOLDERS = {
    "pig_shoulder_v2": "19EgYOSbLj-CTFf6uGhArL6wZ_bChe3SV",
    "spine_phantom": "1VrLEIuexBCeqz3Nt-9e80gEZcy26sImF",
}

for name, folder_id in FOLDERS.items():
    target = OUT / name
    target.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {name} to {target}")
    gdown.download_folder(
        id=folder_id,
        output=str(target),
        quiet=False,
        use_cookies=False,
    )
