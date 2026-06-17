# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules


REPO_ROOT = Path.cwd()
STAGE4_DIR = REPO_ROOT / "stage-4"
STAGE3_DIR = REPO_ROOT / "stage-3"
STAGE4_CODE_DIR = STAGE4_DIR / "code"


datas = [
    (str(STAGE4_DIR / "configs"), "stage-4/configs"),
    (str(STAGE4_DIR / "README_STAGE4.md"), "stage-4"),
    (str(STAGE4_DIR / "requirements-stage4.txt"), "stage-4"),
    (str(STAGE3_DIR / "code" / "task9"), "stage-3/code/task9"),
    (str(STAGE3_DIR / "configs" / "task9_effects"), "stage-3/configs/task9_effects"),
    (str(STAGE3_DIR / "reports" / "task9" / "assets" / "stickers"), "stage-3/reports/task9/assets/stickers"),
]

summaries_dir = STAGE4_DIR / "reports" / "summaries"
if summaries_dir.exists():
    datas.append((str(summaries_dir), "stage-4/reports/summaries"))

datas += collect_data_files("mediapipe")

binaries = []
binaries += collect_dynamic_libs("cv2")
binaries += collect_dynamic_libs("mediapipe")

hiddenimports = [
    "PySide6.QtCore",
    "PySide6.QtGui",
    "PySide6.QtWidgets",
    "cv2",
    "numpy",
    "mediapipe",
    "PIL",
    "PIL.Image",
    "stage4_app_main",
    "stage4_desktop_app",
    "stage4_run_cli",
    "stage4_process_image_cli",
    "stage4_live_camera_worker",
    "stage4_write_report",
    "stage4_backend",
    "stage4_common",
    "stage4_packaging_utils",
]
hiddenimports += collect_submodules("mediapipe")


a = Analysis(
    [str(STAGE4_CODE_DIR / "stage4_app_main.py")],
    pathex=[str(STAGE4_CODE_DIR)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Stage4FaceEffects",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="Stage4FaceEffects",
)

app = BUNDLE(
    coll,
    name="Stage4FaceEffects.app",
    icon=None,
    bundle_identifier="local.stage4.faceeffects",
    info_plist={
        "CFBundleName": "Stage4FaceEffects",
        "CFBundleDisplayName": "Stage4FaceEffects",
        "NSCameraUsageDescription": "Stage4FaceEffects needs camera access for realtime face effects preview and recording.",
        "NSMicrophoneUsageDescription": "Stage4FaceEffects does not record audio; this permission is not required for current features.",
        "NSHighResolutionCapable": True,
    },
)
