#!/usr/bin/env python3
"""Stage4 UI V3 desktop app.

The GUI process deliberately imports only standard-library modules and Qt.
All cv2/numpy/mediapipe/Task9 work runs in child CLI/worker processes.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from stage4_packaging_utils import (
    bundled_or_source_path,
    current_executable_command,
    is_frozen,
    source_repo_root,
    source_script_command,
    source_stage4_root,
    user_data_dir,
)

try:
    from PySide6.QtCore import QProcess, QSize, QTimer, Qt
    from PySide6.QtGui import QPixmap
    from PySide6.QtWidgets import (
        QApplication,
        QCheckBox,
        QFileDialog,
        QFrame,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QScrollArea,
        QSizePolicy,
        QSlider,
        QSpinBox,
        QPlainTextEdit,
        QStackedWidget,
        QVBoxLayout,
        QWidget,
    )

    QT_BINDING = "PySide6"
except ModuleNotFoundError:
    try:
        from PyQt6.QtCore import QProcess, QSize, QTimer, Qt
        from PyQt6.QtGui import QPixmap
        from PyQt6.QtWidgets import (
            QApplication,
            QCheckBox,
            QFileDialog,
            QFrame,
            QGridLayout,
            QGroupBox,
            QHBoxLayout,
            QLabel,
            QLineEdit,
            QMainWindow,
            QMessageBox,
            QPushButton,
            QScrollArea,
            QSizePolicy,
            QSlider,
            QSpinBox,
            QPlainTextEdit,
            QStackedWidget,
            QVBoxLayout,
            QWidget,
        )

        QT_BINDING = "PyQt6"
    except ModuleNotFoundError:
        print(
            "ERROR: Missing desktop GUI dependency. Install PySide6 with "
            "`python -m pip install -r stage-4/requirements-stage4.txt`, "
            "or install PyQt6.",
            file=sys.stderr,
        )
        raise SystemExit(1)


EFFECT_CHOICES = ("glasses", "hat", "smooth", "whiten", "lipstick")
EFFECT_LABELS = {
    "glasses": "眼镜 glasses",
    "hat": "帽子 hat",
    "smooth": "磨皮 smooth",
    "whiten": "美白 whiten",
    "lipstick": "口红 lipstick",
}
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv"}

FROZEN_MODE = is_frozen()
SOURCE_REPO_ROOT = source_repo_root()
SOURCE_STAGE4_ROOT = source_stage4_root()
RESOURCE_STAGE4_ROOT = bundled_or_source_path("stage-4")
USER_DATA_ROOT = user_data_dir()
PROCESS_WORK_DIR = USER_DATA_ROOT if FROZEN_MODE else SOURCE_REPO_ROOT
REPORT_DIR = USER_DATA_ROOT
ASSET_DIR = REPORT_DIR / "assets"
SUMMARY_DIR = REPORT_DIR / "summaries"
DEFAULT_INPUT_VIDEO = bundled_or_source_path(
    "stage-3/reports/task9/assets/videos/task9_dynamic_effects_demo.mp4"
)

LOCAL_VIDEO_OUTPUT = ASSET_DIR / "videos" / "stage4_desktop_export.mp4"
LOCAL_VIDEO_DIR = ASSET_DIR / "videos"
LOCAL_VIDEO_SUMMARY = SUMMARY_DIR / "stage4_desktop_export_summary.json"
LOCAL_IMAGE_OUTPUT = ASSET_DIR / "images" / "stage4_image_effects_export.jpg"
LOCAL_IMAGE_DIR = ASSET_DIR / "images"
LOCAL_IMAGE_SUMMARY = SUMMARY_DIR / "stage4_image_effects_summary.json"

RUNTIME_DIR = REPORT_DIR / "runtime"
LIVE_CONTROLS = RUNTIME_DIR / "live_controls.json"
LIVE_PREVIEW = RUNTIME_DIR / "live_preview.jpg"
LIVE_STATUS = RUNTIME_DIR / "live_status.json"
SCREENSHOT_DIR = ASSET_DIR / "screenshots"
RECORDINGS_DIR = ASSET_DIR / "recordings"

UI_CHECK_SUMMARY = SUMMARY_DIR / "stage4_ui_check_summary.json"
UI_V3_SUMMARY = SUMMARY_DIR / "stage4_ui_v3_summary.json"
ERROR_LOG = SUMMARY_DIR / "stage4_desktop_error_log.txt"


APP_STYLE = """
QMainWindow, QWidget {
  background: #f4f7fb;
  color: #172033;
  font-family: "PingFang SC", "Helvetica Neue", Arial;
}
QGroupBox, QFrame#card {
  background: rgba(255, 255, 255, 0.92);
  border: 1px solid #dbe4f0;
  border-radius: 8px;
  margin-top: 12px;
}
QGroupBox::title {
  subcontrol-origin: margin;
  left: 18px;
  padding: 0 8px;
  color: #526079;
  font-weight: 600;
}
QPushButton {
  background: #e9eef8;
  border: 1px solid #d4ddec;
  border-radius: 8px;
  padding: 10px 16px;
  font-weight: 600;
}
QPushButton:hover {
  background: #dee8fb;
}
QPushButton#primaryButton {
  background: #6c63ff;
  color: white;
  border: none;
}
QPushButton#primaryButton:hover {
  background: #5d55eb;
}
QPushButton#dangerButton {
  background: #ffe8ee;
  color: #b4234b;
  border-color: #f6becd;
}
QLineEdit, QSpinBox, QPlainTextEdit {
  background: #ffffff;
  border: 1px solid #d8e0ed;
  border-radius: 8px;
  padding: 8px;
}
QCheckBox {
  spacing: 8px;
  padding: 5px 2px;
}
QSlider::groove:horizontal {
  height: 6px;
  border-radius: 3px;
  background: #dce5f3;
}
QSlider::handle:horizontal {
  width: 18px;
  height: 18px;
  margin: -6px 0;
  border-radius: 9px;
  background: #6c63ff;
}
"""


def ensure_dirs() -> None:
    for path in [
        REPORT_DIR,
        ASSET_DIR,
        LOCAL_VIDEO_OUTPUT.parent,
        LOCAL_IMAGE_OUTPUT.parent,
        LOCAL_VIDEO_SUMMARY.parent,
        SCREENSHOT_DIR,
        RECORDINGS_DIR,
        RUNTIME_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def rel(path: Path) -> str:
    base = USER_DATA_ROOT if FROZEN_MODE else SOURCE_REPO_ROOT
    try:
        return os.path.relpath(str(path), str(base))
    except ValueError:
        return str(path)


def atomic_write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def read_json(path: Path) -> Dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def quote_command(command: List[str]) -> str:
    return shlex.join([str(part) for part in command])


def local_cli_base(is_image: bool) -> List[str]:
    if FROZEN_MODE:
        return current_executable_command("run-cli")
    script_name = "stage4_process_image_cli.py" if is_image else "stage4_run_cli.py"
    return source_script_command(script_name)


def live_worker_base() -> List[str]:
    if FROZEN_MODE:
        return current_executable_command("live-worker")
    return source_script_command("stage4_live_camera_worker.py")


def default_effects() -> set[str]:
    return {"glasses", "whiten"}


def timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def uniquify_path(path: Path) -> Path:
    path = path.expanduser()
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    for index in range(1, 1000):
        candidate = parent / "{}_{:02d}{}".format(stem, index, suffix)
        if not candidate.exists():
            return candidate
    return parent / "{}_{}{}".format(stem, timestamp(), suffix)


def write_ui_check_summary(command: List[str], safe_mode: bool, note: str) -> None:
    atomic_write_json(
        UI_CHECK_SUMMARY,
        {
            "desktop_app_entry": quote_command(current_executable_command("gui"))
            if FROZEN_MODE
            else str(SOURCE_STAGE4_ROOT / "code" / "stage4_desktop_app.py"),
            "gui_subprocess_mode": True,
            "frozen_mode": FROZEN_MODE,
            "imports_cv2_in_gui_process": False,
            "imports_numpy_in_gui_process": False,
            "imports_mediapipe_in_gui_process": False,
            "imports_backend_in_gui_process": False,
            "safe_mode_supported": True,
            "safe_mode": safe_mode,
            "cli_export_command": quote_command(command) if command else None,
            "manual_ui_check_required": True,
            "notes": [
                note,
                "Stage4 UI V3 keeps cv2, numpy, mediapipe, stage4_backend, and Stage3 Task9 out of the GUI process.",
                "Realtime preview is embedded through preview image polling from a worker subprocess.",
                "Packaged mode starts CLI and worker subprocesses through the current app executable and stage4_app_main modes.",
            ],
        },
    )


def write_ui_v3_summary(command: List[str], note: str) -> None:
    atomic_write_json(
        UI_V3_SUMMARY,
        {
            "app_version": "stage4_ui_v3",
            "home_page": True,
            "local_import_page": True,
            "realtime_camera_page": True,
            "embedded_realtime_preview": True,
            "realtime_worker_subprocess": True,
            "frozen_mode": FROZEN_MODE,
            "user_data_dir": str(USER_DATA_ROOT),
            "packaged_subprocess_entry": quote_command(current_executable_command("run-cli")) if FROZEN_MODE else None,
            "gui_imports_cv2": False,
            "gui_imports_mediapipe": False,
            "gui_imports_backend": False,
            "realtime_effect_toggle_supported": True,
            "realtime_strength_adjust_supported": True,
            "resizable_window_supported": True,
            "height_resize_supported": True,
            "right_panel_scroll_area": True,
            "preview_area_expanding": True,
            "fixed_height_removed": True,
            "realtime_recording_supported": True,
            "recording_control_via_live_controls": True,
            "recording_saved_to_local": True,
            "recording_output_dir": str(RECORDINGS_DIR),
            "recording_writer_in_worker": True,
            "local_export_full_length_by_default": True,
            "local_export_original_resolution_by_default": True,
            "fast_preview_optional": True,
            "user_selectable_local_output_path": True,
            "user_selectable_recording_output_path": True,
            "default_max_frames": None,
            "default_fast_mode": False,
            "default_process_width": None,
            "camera_permission_note": "实时视频需要摄像头权限。首次进入实时视频时，macOS 可能弹出权限请求。",
            "worker_script": quote_command(live_worker_base()),
            "controls_path": str(LIVE_CONTROLS),
            "preview_path": str(LIVE_PREVIEW),
            "status_path": str(LIVE_STATUS),
            "last_command_preview": quote_command(command) if command else None,
            "known_limitations": [
                "Realtime preview uses file polling, so latency depends on disk and CPU load.",
                "The current UI is a course-project interaction demo, not commercial-grade realtime beautification.",
                "Camera access may require macOS privacy permission for Terminal or the active Python launcher.",
                "Changing camera index while running requires reopening the worker.",
            ],
            "manual_check_required": True,
            "notes": [note],
        },
    )


def load_home_status() -> Dict[str, str]:
    summary_path = SUMMARY_DIR / "stage4_integration_summary.json"
    report_path = REPORT_DIR / "stage4_project_integration_report.md"
    readme_path = RESOURCE_STAGE4_ROOT / "README_STAGE4.md"
    status = {
        "smoke": "CLI smoke test: not found",
        "environment": "Environment: not checked",
        "summary": rel(summary_path),
        "report": rel(report_path),
        "readme": str(readme_path),
    }
    data = read_json(summary_path)
    if data:
        frames = data.get("processed_frame_count")
        fps = data.get("processing_fps")
        if frames:
            status["smoke"] = "CLI smoke test: passed, frames={}, fps={:.3f}".format(frames, float(fps or 0))
        env = data.get("environment", {})
        status["environment"] = "Environment: Python {}, cv2 {}, mediapipe {}, {}".format(
            env.get("python_version", "unknown"),
            env.get("cv2_version", "unknown"),
            env.get("mediapipe_version", "unknown"),
            QT_BINDING,
        )
    return status


def make_card() -> QFrame:
    card = QFrame()
    card.setObjectName("card")
    return card


def make_scroll_area(widget: QWidget) -> QScrollArea:
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
    scroll.setFrameShape(QFrame.Shape.NoFrame)
    scroll.setWidget(widget)
    return scroll


def set_primary(button: QPushButton) -> QPushButton:
    button.setObjectName("primaryButton")
    return button


def set_danger(button: QPushButton) -> QPushButton:
    button.setObjectName("dangerButton")
    return button


class StrengthSlider(QWidget):
    def __init__(self, title: str, default_value: float, on_change) -> None:
        super().__init__()
        self.label = QLabel()
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 100)
        self.slider.setValue(int(default_value * 100))
        self.slider.valueChanged.connect(self._update_label)
        self.slider.valueChanged.connect(on_change)
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.label)
        layout.addWidget(self.slider)
        self.setLayout(layout)
        self.title = title
        self._update_label()

    def _update_label(self, *_args) -> None:
        self.label.setText("{} {:.2f}".format(self.title, self.value()))

    def value(self) -> float:
        return self.slider.value() / 100.0


class EffectsPanel(QGroupBox):
    def __init__(self, include_strengths: bool, on_change) -> None:
        super().__init__("特效与强度")
        self.checks: Dict[str, QCheckBox] = {}
        self.smooth = StrengthSlider("磨皮强度", 0.55, on_change)
        self.whiten = StrengthSlider("美白强度", 0.35, on_change)
        self.lipstick = StrengthSlider("口红强度", 0.45, on_change)

        layout = QVBoxLayout()
        checks = QGridLayout()
        for index, effect in enumerate(EFFECT_CHOICES):
            checkbox = QCheckBox(EFFECT_LABELS[effect])
            checkbox.setChecked(effect in default_effects())
            checkbox.stateChanged.connect(on_change)
            self.checks[effect] = checkbox
            checks.addWidget(checkbox, index // 2, index % 2)
        layout.addLayout(checks)
        if include_strengths:
            layout.addWidget(self.smooth)
            layout.addWidget(self.whiten)
            layout.addWidget(self.lipstick)
        self.setLayout(layout)

    def selected(self) -> List[str]:
        return [effect for effect, checkbox in self.checks.items() if checkbox.isChecked()]

    def strengths(self) -> Dict[str, float]:
        return {
            "smooth_strength": self.smooth.value(),
            "whiten_strength": self.whiten.value(),
            "lipstick_alpha": self.lipstick.value(),
        }


class HomePage(QWidget):
    def __init__(self, window: "MainWindow") -> None:
        super().__init__()
        self.window = window
        root = QVBoxLayout()
        root.setContentsMargins(48, 36, 48, 36)
        root.setSpacing(22)

        hero = make_card()
        hero_layout = QVBoxLayout()
        hero_layout.setContentsMargins(34, 30, 34, 30)
        title = QLabel("Stage4 人脸视觉特效应用")
        title.setStyleSheet("font-size: 30px; font-weight: 800; color: #172033;")
        subtitle = QLabel("本地图片/视频处理 · 实时摄像头特效 · 可选美颜与贴纸")
        subtitle.setStyleSheet("font-size: 16px; color: #526079;")
        permission = QLabel("实时视频需要摄像头权限。首次进入实时视频时，macOS 可能弹出权限请求。")
        permission.setStyleSheet("color: #6c63ff; font-weight: 600;")
        hero_layout.addWidget(title)
        hero_layout.addWidget(subtitle)
        hero_layout.addSpacing(6)
        hero_layout.addWidget(permission)
        hero.setLayout(hero_layout)
        root.addWidget(hero)

        cards = QHBoxLayout()
        cards.setSpacing(18)
        cards.addWidget(self._entry_card("本地导入", "上传图片或视频，选择特效后导出结果。", "进入本地导入", window.show_local_page))
        cards.addWidget(self._entry_card("实时视频", "调用摄像头，在应用内实时预览人脸特效。", "进入实时视频", window.show_realtime_page))
        root.addLayout(cards, stretch=1)

        status_card = make_card()
        status_layout = QVBoxLayout()
        status_layout.setContentsMargins(24, 20, 24, 20)
        self.status_label = QLabel()
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("color: #526079; line-height: 1.4;")
        status_layout.addWidget(self.status_label)
        status_card.setLayout(status_layout)
        root.addWidget(status_card)
        self.setLayout(root)
        self.refresh_status()

    def _entry_card(self, title: str, detail: str, button_text: str, callback) -> QFrame:
        card = make_card()
        layout = QVBoxLayout()
        layout.setContentsMargins(28, 28, 28, 28)
        title_label = QLabel(title)
        title_label.setStyleSheet("font-size: 24px; font-weight: 800;")
        detail_label = QLabel(detail)
        detail_label.setWordWrap(True)
        detail_label.setStyleSheet("font-size: 14px; color: #526079;")
        button = set_primary(QPushButton(button_text))
        button.setMinimumHeight(48)
        button.clicked.connect(callback)
        layout.addWidget(title_label)
        layout.addWidget(detail_label)
        layout.addStretch(1)
        layout.addWidget(button)
        card.setLayout(layout)
        card.setMinimumHeight(220)
        return card

    def refresh_status(self) -> None:
        status = load_home_status()
        self.status_label.setText(
            "{}\n{}\nReport: {}\nREADME: {}\nSummary: {}".format(
                status["smoke"],
                status["environment"],
                status["report"],
                status["readme"],
                status["summary"],
            )
        )


class ProcessPage(QWidget):
    def __init__(self, window: "MainWindow") -> None:
        super().__init__()
        self.window = window
        self.process = QProcess(self)
        PROCESS_WORK_DIR.mkdir(parents=True, exist_ok=True)
        self.process.setWorkingDirectory(str(PROCESS_WORK_DIR))
        self.process.readyReadStandardOutput.connect(self.append_stdout)
        self.process.readyReadStandardError.connect(self.append_stderr)
        self.process.started.connect(self.on_started)
        self.process.finished.connect(self.on_finished)
        self.process.errorOccurred.connect(self.on_error)
        self.current_command: List[str] = []
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(2500)
        self.log_view.setMinimumHeight(78)
        self.log_view.setMaximumHeight(140)
        self.command_preview = QPlainTextEdit()
        self.command_preview.setReadOnly(True)
        self.command_preview.setMaximumBlockCount(200)
        self.command_preview.setMinimumHeight(56)
        self.command_preview.setMaximumHeight(100)

    def append_log(self, text: str) -> None:
        text = text.rstrip()
        if text:
            self.log_view.appendPlainText(text)

    def append_stdout(self) -> None:
        data = bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace")
        self.append_log(data)

    def append_stderr(self) -> None:
        data = bytes(self.process.readAllStandardError()).decode("utf-8", errors="replace")
        self.append_log(data)

    def start_process(self, command: List[str], note: str) -> bool:
        if self.process.state() != QProcess.ProcessState.NotRunning:
            QMessageBox.warning(self, "Stage4", "A subprocess is already running.")
            return False
        self.current_command = command
        self.log_view.clear()
        self.append_log("$ {}".format(quote_command(command)))
        write_ui_check_summary(command, self.window.safe_mode, note)
        write_ui_v3_summary(command, note)
        self.process.start(command[0], command[1:])
        return True

    def terminate_process(self) -> None:
        if self.process.state() != QProcess.ProcessState.NotRunning:
            self.append_log("Terminating subprocess...")
            self.process.terminate()
            self.window.statusBar().showMessage("terminating")

    def on_started(self) -> None:
        self.window.statusBar().showMessage("running")

    def on_finished(self, exit_code: int, _exit_status) -> None:
        if exit_code == 0:
            self.window.statusBar().showMessage("finished")
            self.append_log("Subprocess finished successfully.")
        else:
            self.window.statusBar().showMessage("failed")
            self.append_log("Subprocess failed with exit code {}.".format(exit_code))
            ERROR_LOG.parent.mkdir(parents=True, exist_ok=True)
            ERROR_LOG.write_text(self.log_view.toPlainText() + "\n", encoding="utf-8")
            self.append_log("Error log: {}".format(ERROR_LOG))

    def on_error(self, error) -> None:
        self.window.statusBar().showMessage("failed")
        self.append_log("QProcess error: {}".format(error))
        ERROR_LOG.parent.mkdir(parents=True, exist_ok=True)
        ERROR_LOG.write_text(self.log_view.toPlainText() + "\n", encoding="utf-8")

    def close_process(self) -> None:
        if self.process.state() != QProcess.ProcessState.NotRunning:
            self.process.terminate()
            if not self.process.waitForFinished(1500):
                self.process.kill()


class LocalImportPage(ProcessPage):
    def __init__(self, window: "MainWindow") -> None:
        super().__init__(window)
        self.input_edit = QLineEdit(str(DEFAULT_INPUT_VIDEO if DEFAULT_INPUT_VIDEO.exists() else ""))
        self.user_output_path: Optional[Path] = None
        self.output_path: Optional[Path] = None
        self.output_kind: Optional[str] = None
        self.output_edit = QLineEdit()
        self.summary_edit = QLineEdit()
        self.output_edit.setReadOnly(True)
        self.summary_edit.setReadOnly(True)
        self.file_info = QLabel()
        self.file_info.setWordWrap(True)
        self.effects = EffectsPanel(include_strengths=True, on_change=self.update_outputs_and_command)
        self.fast_mode = QCheckBox("快速预览模式（降低分辨率/限制帧数）")
        self.fast_mode.setChecked(False)
        self.process_width = QSpinBox()
        self.process_width.setRange(160, 4096)
        self.process_width.setValue(720)
        self.process_width.setEnabled(False)
        self.max_frames = QLineEdit("30")
        self.max_frames.setPlaceholderText("仅快速预览模式生效")
        self.max_frames.setEnabled(False)
        self.start_button = set_primary(QPushButton("开始处理"))
        self.stop_button = set_danger(QPushButton("停止处理"))
        self.open_output_button = QPushButton("打开输出目录")
        self.choose_output_button = QPushButton("选择保存位置")
        self.build_ui()
        self.connect_ui()
        self.update_outputs_and_command()

    def build_ui(self) -> None:
        root = QVBoxLayout()
        root.setContentsMargins(28, 24, 28, 24)
        top = QHBoxLayout()
        back = QPushButton("返回首页")
        back.clicked.connect(self.window.show_home_page)
        title = QLabel("本地导入")
        title.setStyleSheet("font-size: 24px; font-weight: 800;")
        top.addWidget(back)
        top.addWidget(title)
        top.addStretch(1)
        root.addLayout(top)

        content = QHBoxLayout()
        content.setSpacing(18)

        left = make_card()
        left_layout = QVBoxLayout()
        left_layout.setContentsMargins(22, 22, 22, 22)
        file_title = QLabel("文件预览 / 文件信息")
        file_title.setStyleSheet("font-size: 18px; font-weight: 700;")
        choose_row = QHBoxLayout()
        choose_image = QPushButton("选择图片")
        choose_video = QPushButton("选择视频")
        choose_image.clicked.connect(lambda: self.choose_input("image"))
        choose_video.clicked.connect(lambda: self.choose_input("video"))
        choose_row.addWidget(choose_image)
        choose_row.addWidget(choose_video)
        left_layout.addWidget(file_title)
        left_layout.addWidget(self.input_edit)
        left_layout.addLayout(choose_row)
        left_layout.addWidget(self.file_info, stretch=1)
        left_layout.addWidget(QLabel("输出文件"))
        output_row = QHBoxLayout()
        output_row.addWidget(self.output_edit, stretch=1)
        output_row.addWidget(self.choose_output_button)
        left_layout.addLayout(output_row)
        left_layout.addWidget(QLabel("Summary"))
        left_layout.addWidget(self.summary_edit)
        hint = QLabel("默认按原视频完整长度和原始分辨率导出；勾选快速预览模式后才会缩放或限制帧数。")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #6c63ff; font-weight: 600;")
        left_layout.addWidget(hint)
        left.setLayout(left_layout)
        left.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        content.addWidget(left, stretch=3)

        right = make_card()
        right_layout = QVBoxLayout()
        right_layout.setContentsMargins(22, 22, 22, 22)
        right_layout.addWidget(self.effects)
        settings = QGroupBox("导出参数")
        settings_layout = QGridLayout()
        settings_layout.addWidget(QLabel("fast mode"), 0, 0)
        settings_layout.addWidget(self.fast_mode, 0, 1)
        settings_layout.addWidget(QLabel("process width"), 1, 0)
        settings_layout.addWidget(self.process_width, 1, 1)
        settings_layout.addWidget(QLabel("max frames"), 2, 0)
        settings_layout.addWidget(self.max_frames, 2, 1)
        settings.setLayout(settings_layout)
        right_layout.addWidget(settings)
        buttons = QHBoxLayout()
        buttons.addWidget(self.start_button)
        buttons.addWidget(self.stop_button)
        buttons.addWidget(self.open_output_button)
        right_layout.addLayout(buttons)
        right_layout.addStretch(1)
        right.setLayout(right_layout)
        right.setMinimumWidth(340)
        right.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        control_scroll = make_scroll_area(right)
        control_scroll.setMinimumWidth(360)
        control_scroll.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        content.addWidget(control_scroll, stretch=1)
        root.addLayout(content, stretch=3)

        bottom = QHBoxLayout()
        command_card = make_card()
        command_layout = QVBoxLayout()
        command_layout.setContentsMargins(16, 16, 16, 16)
        command_layout.addWidget(QLabel("命令预览"))
        command_layout.addWidget(self.command_preview)
        command_card.setLayout(command_layout)
        log_card = make_card()
        log_layout = QVBoxLayout()
        log_layout.setContentsMargins(16, 16, 16, 16)
        log_layout.addWidget(QLabel("日志"))
        log_layout.addWidget(self.log_view)
        log_card.setLayout(log_layout)
        bottom.addWidget(command_card)
        bottom.addWidget(log_card)
        root.addLayout(bottom, stretch=1)
        self.setLayout(root)

    def connect_ui(self) -> None:
        self.input_edit.textChanged.connect(self.update_outputs_and_command)
        self.fast_mode.stateChanged.connect(self.update_outputs_and_command)
        self.fast_mode.stateChanged.connect(self.update_fast_mode_controls)
        self.process_width.valueChanged.connect(self.update_outputs_and_command)
        self.max_frames.textChanged.connect(self.update_outputs_and_command)
        self.choose_output_button.clicked.connect(self.choose_output_path)
        self.start_button.clicked.connect(self.start_local_process)
        self.stop_button.clicked.connect(self.terminate_process)
        self.open_output_button.clicked.connect(self.open_output_dir)

    def input_path(self) -> Path:
        return Path(self.input_edit.text().strip()).expanduser()

    def is_image(self) -> bool:
        return self.input_path().suffix.lower() in IMAGE_EXTS

    def is_video(self) -> bool:
        return self.input_path().suffix.lower() in VIDEO_EXTS

    def current_output(self) -> Path:
        if self.output_path is None:
            self.output_path = self.default_output_path()
        return self.output_path

    def current_summary(self) -> Path:
        return LOCAL_IMAGE_SUMMARY if self.is_image() else LOCAL_VIDEO_SUMMARY

    def current_kind(self) -> str:
        return "image" if self.is_image() else "video"

    def default_output_path(self) -> Path:
        if self.is_image():
            return LOCAL_IMAGE_DIR / "stage4_image_export_{}.jpg".format(timestamp())
        return LOCAL_VIDEO_DIR / "stage4_local_export_{}.mp4".format(timestamp())

    def choose_input(self, mode: str) -> None:
        if mode == "image":
            file_filter = "Images (*.jpg *.jpeg *.png)"
            base = str(Path.home() if FROZEN_MODE else SOURCE_REPO_ROOT)
        else:
            file_filter = "Videos (*.mp4 *.mov *.avi *.mkv)"
            base = str(DEFAULT_INPUT_VIDEO.parent if DEFAULT_INPUT_VIDEO.exists() else Path.home())
        path, _ = QFileDialog.getOpenFileName(self, "选择输入文件", base, file_filter)
        if path:
            self.input_edit.setText(path)

    def choose_output_path(self) -> None:
        if self.is_image():
            default_path = LOCAL_IMAGE_DIR / "stage4_image_export_{}.jpg".format(timestamp())
            file_filter = "JPEG Image (*.jpg);;PNG Image (*.png)"
        else:
            default_path = LOCAL_VIDEO_DIR / "stage4_local_export_{}.mp4".format(timestamp())
            file_filter = "MP4 Video (*.mp4)"
        path, _ = QFileDialog.getSaveFileName(self, "选择保存位置", str(default_path), file_filter)
        if path:
            self.user_output_path = Path(path).expanduser()
            self.output_path = self.user_output_path
            self.output_kind = self.current_kind()
            self.update_outputs_and_command()

    def command(self) -> List[str]:
        effects = self.effects.selected()
        strengths = self.effects.strengths()
        strength_args = [
            "--smooth-strength",
            "{:.2f}".format(strengths["smooth_strength"]),
            "--whiten-strength",
            "{:.2f}".format(strengths["whiten_strength"]),
            "--lipstick-alpha",
            "{:.2f}".format(strengths["lipstick_alpha"]),
        ]
        if self.is_image():
            command = [
                *local_cli_base(is_image=True),
                "--image",
                self.input_edit.text().strip(),
                "--effects",
                *effects,
                *strength_args,
                "--output-image",
                str(self.current_output()),
                "--summary",
                str(LOCAL_IMAGE_SUMMARY),
            ]
            if self.fast_mode.isChecked():
                command.append("--fast-mode")
                command.extend(["--process-width", str(self.process_width.value())])
            return command
        command = [
            *local_cli_base(is_image=False),
            "--video",
            self.input_edit.text().strip(),
            "--effects",
            *effects,
            *strength_args,
        ]
        if self.fast_mode.isChecked():
            command.append("--fast-mode")
            command.extend(["--process-width", str(self.process_width.value())])
            max_frames_value = self.max_frames.text().strip()
            if max_frames_value:
                command.extend(["--max-frames", max_frames_value])
        command.extend(["--output-video", str(self.current_output()), "--summary", str(LOCAL_VIDEO_SUMMARY)])
        command.extend(["--output-path-source", "user" if self.user_output_path else "default"])
        return command

    def update_outputs_and_command(self, *_args) -> None:
        kind = self.current_kind()
        if self.user_output_path is None and (self.output_path is None or self.output_kind != kind):
            self.output_path = self.default_output_path()
            self.output_kind = kind
        self.output_edit.setText(str(self.current_output()))
        self.summary_edit.setText(str(self.current_summary()))
        input_path = self.input_path()
        kind = "图片" if self.is_image() else "视频" if self.is_video() else "未知类型"
        exists = "存在" if input_path.exists() else "未找到"
        runner_note = (
            "打包模式通过 Stage4FaceEffects --run-cli 分发本地处理。"
            if FROZEN_MODE
            else "源码模式下视频调用 stage4_run_cli.py，图片调用 stage4_process_image_cli.py。"
        )
        self.file_info.setText(
            "类型：{}\n状态：{}\n路径：{}\n\n{}".format(
                kind, exists, self.input_edit.text().strip() or "未选择", runner_note
            )
        )
        self.command_preview.setPlainText(quote_command(self.command()))

    def update_fast_mode_controls(self, *_args) -> None:
        enabled = self.fast_mode.isChecked()
        self.process_width.setEnabled(enabled)
        self.max_frames.setEnabled(enabled)
        self.update_outputs_and_command()

    def prepare_output_path_for_start(self) -> None:
        if self.user_output_path is None:
            self.output_path = self.default_output_path()
            self.output_kind = self.current_kind()
        else:
            safe_path = uniquify_path(self.user_output_path)
            self.output_path = safe_path
            self.user_output_path = safe_path
            self.output_kind = self.current_kind()
        self.update_outputs_and_command()

    def start_local_process(self) -> None:
        if not self.input_edit.text().strip():
            QMessageBox.warning(self, "Stage4", "请先选择输入文件。")
            return
        if not self.effects.selected():
            QMessageBox.warning(self, "Stage4", "请至少选择一个视觉特效。")
            return
        if not (self.is_image() or self.is_video()):
            QMessageBox.warning(self, "Stage4", "仅支持 jpg/jpeg/png/mp4/mov/avi/mkv。")
            return
        max_frames_value = self.max_frames.text().strip()
        if self.fast_mode.isChecked() and max_frames_value and not max_frames_value.isdigit():
            QMessageBox.warning(self, "Stage4", "max frames 需要是整数，或留空。")
            return
        self.prepare_output_path_for_start()
        self.start_process(self.command(), "Local import process started.")

    def open_output_dir(self) -> None:
        target = self.current_output().parent
        target.mkdir(parents=True, exist_ok=True)
        try:
            subprocess.Popen(["open", str(target)])
        except Exception as exc:
            self.append_log("Could not open output directory: {}".format(exc))


class RealtimeCameraPage(ProcessPage):
    def __init__(self, window: "MainWindow") -> None:
        super().__init__(window)
        self.last_pixmap: Optional[QPixmap] = None
        self.preview_mtime = 0.0
        self.recording_requested = False
        self.recording_output_path: Optional[Path] = None
        self.user_recording_output_path: Optional[Path] = None
        self.camera_index = QSpinBox()
        self.camera_index.setRange(0, 8)
        self.camera_index.setValue(0)
        self.process_width = QSpinBox()
        self.process_width.setRange(160, 4096)
        self.process_width.setValue(720)
        self.show_fps = QCheckBox("显示 FPS")
        self.show_fps.setChecked(True)
        self.effects = EffectsPanel(include_strengths=True, on_change=self.on_controls_changed)
        self.status_label = QLabel("等待")
        self.fps_label = QLabel("FPS: --")
        self.effects_label = QLabel("启用特效: glasses, whiten")
        self.recording_status_label = QLabel("未录像")
        self.recording_path_edit = QLineEdit()
        self.recording_path_edit.setReadOnly(True)
        self.preview_label = QLabel("正在打开摄像头...")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumSize(QSize(360, 260))
        self.preview_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.preview_label.setScaledContents(False)
        self.preview_label.setStyleSheet(
            "background: #101828; color: #d0d5dd; border-radius: 8px; font-size: 18px; font-weight: 600;"
        )
        self.restart_button = set_primary(QPushButton("重新打开摄像头"))
        self.close_button = set_danger(QPushButton("关闭摄像头"))
        self.screenshot_button = QPushButton("保存截图")
        self.record_button = set_primary(QPushButton("开始录像"))
        self.choose_recording_button = QPushButton("选择录像保存位置")
        self.open_recordings_button = QPushButton("打开录像目录")
        self.preview_timer = QTimer(self)
        self.preview_timer.setInterval(40)
        self.preview_timer.timeout.connect(self.refresh_preview)
        self.status_timer = QTimer(self)
        self.status_timer.setInterval(400)
        self.status_timer.timeout.connect(self.refresh_status)
        self.build_ui()
        self.connect_ui()
        self.update_command_preview()
        self.write_controls(stop_signal=True)

    def build_ui(self) -> None:
        root = QVBoxLayout()
        root.setContentsMargins(28, 24, 28, 24)
        root.setSpacing(16)
        top = QHBoxLayout()
        back = QPushButton("返回首页")
        back.clicked.connect(self.window.show_home_page)
        title = QLabel("实时人脸特效")
        title.setStyleSheet("font-size: 24px; font-weight: 800;")
        top.addWidget(back)
        top.addWidget(title)
        top.addStretch(1)
        top.addWidget(QLabel("摄像头状态："))
        top.addWidget(self.status_label)
        root.addLayout(top)

        content = QHBoxLayout()
        content.setSpacing(18)
        preview_card = make_card()
        preview_layout = QVBoxLayout()
        preview_layout.setContentsMargins(20, 20, 20, 20)
        preview_layout.addWidget(self.preview_label, stretch=1)
        meta = QHBoxLayout()
        meta.addWidget(self.fps_label)
        meta.addWidget(self.effects_label)
        meta.addStretch(1)
        meta.addWidget(self.screenshot_button)
        preview_layout.addLayout(meta)
        preview_card.setLayout(preview_layout)
        preview_card.setMinimumHeight(360)
        preview_card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        content.addWidget(preview_card, stretch=3)

        control_card = make_card()
        control_layout = QVBoxLayout()
        control_layout.setContentsMargins(20, 20, 20, 20)
        control_layout.addWidget(self.effects)
        settings = QGroupBox("预览设置")
        settings_layout = QGridLayout()
        settings_layout.addWidget(QLabel("camera index"), 0, 0)
        settings_layout.addWidget(self.camera_index, 0, 1)
        settings_layout.addWidget(QLabel("process width"), 1, 0)
        settings_layout.addWidget(self.process_width, 1, 1)
        settings_layout.addWidget(QLabel("show fps"), 2, 0)
        settings_layout.addWidget(self.show_fps, 2, 1)
        settings.setLayout(settings_layout)
        control_layout.addWidget(settings)
        recording_group = QGroupBox("实时录像")
        recording_layout = QGridLayout()
        recording_layout.addWidget(QLabel("状态"), 0, 0)
        recording_layout.addWidget(self.recording_status_label, 0, 1)
        recording_layout.addWidget(QLabel("输出路径"), 1, 0)
        recording_layout.addWidget(self.recording_path_edit, 1, 1)
        recording_layout.addWidget(self.record_button, 2, 0)
        recording_layout.addWidget(self.choose_recording_button, 2, 1)
        recording_layout.addWidget(self.open_recordings_button, 3, 0, 1, 2)
        recording_group.setLayout(recording_layout)
        control_layout.addWidget(recording_group)
        button_row = QHBoxLayout()
        button_row.addWidget(self.restart_button)
        button_row.addWidget(self.close_button)
        control_layout.addLayout(button_row)
        control_layout.addWidget(QLabel("命令预览"))
        control_layout.addWidget(self.command_preview)
        control_layout.addWidget(QLabel("Worker 日志"))
        control_layout.addWidget(self.log_view)
        control_layout.addStretch(1)
        control_card.setLayout(control_layout)
        control_card.setMinimumWidth(340)
        control_card.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        control_scroll = make_scroll_area(control_card)
        control_scroll.setMinimumWidth(360)
        control_scroll.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        content.addWidget(control_scroll, stretch=1)
        root.addLayout(content, stretch=1)
        self.setLayout(root)

    def connect_ui(self) -> None:
        self.camera_index.valueChanged.connect(self.update_command_preview)
        self.process_width.valueChanged.connect(self.on_controls_changed)
        self.show_fps.stateChanged.connect(self.on_controls_changed)
        self.restart_button.clicked.connect(self.restart_worker)
        self.close_button.clicked.connect(self.stop_worker)
        self.screenshot_button.clicked.connect(self.save_screenshot)
        self.record_button.clicked.connect(self.toggle_recording)
        self.choose_recording_button.clicked.connect(self.choose_recording_path)
        self.open_recordings_button.clicked.connect(self.open_recordings_dir)

    def command(self) -> List[str]:
        return [
            *live_worker_base(),
            "--camera",
            str(self.camera_index.value()),
            "--process-width",
            str(self.process_width.value()),
            "--controls",
            str(LIVE_CONTROLS),
            "--preview",
            str(LIVE_PREVIEW),
            "--status",
            str(LIVE_STATUS),
            "--screenshot-dir",
            str(SCREENSHOT_DIR),
        ]

    def controls_payload(self, stop_signal: bool = False) -> Dict:
        strengths = self.effects.strengths()
        return {
            "effects": self.effects.selected(),
            "smooth_strength": strengths["smooth_strength"],
            "whiten_strength": strengths["whiten_strength"],
            "lipstick_alpha": strengths["lipstick_alpha"],
            "show_fps": self.show_fps.isChecked(),
            "process_width": self.process_width.value(),
            "camera_index": self.camera_index.value(),
            "recording": self.recording_requested,
            "recording_output_path": str(self.recording_output_path) if self.recording_output_path else "",
            "user_selected_recording_path": bool(self.user_recording_output_path),
            "recording_fps": 30.0,
            "recording_fourcc": "mp4v",
            "stop_signal": stop_signal,
            "updated_at": timestamp(),
        }

    def write_controls(self, stop_signal: bool = False) -> None:
        ensure_dirs()
        atomic_write_json(LIVE_CONTROLS, self.controls_payload(stop_signal=stop_signal))

    def on_controls_changed(self, *_args) -> None:
        self.write_controls(stop_signal=False)
        self.update_command_preview()
        self.effects_label.setText("启用特效: {}".format(", ".join(self.effects.selected()) or "none"))

    def update_command_preview(self, *_args) -> None:
        self.command_preview.setPlainText(quote_command(self.command()))

    def start_worker_if_needed(self) -> None:
        if self.window.safe_mode:
            self.preview_label.setText("Safe mode: GUI only")
            self.status_label.setText("safe")
            return
        if self.process.state() != QProcess.ProcessState.NotRunning:
            return
        self.start_worker()

    def start_worker(self) -> None:
        ensure_dirs()
        self.write_controls(stop_signal=False)
        for path in [LIVE_PREVIEW, LIVE_STATUS]:
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        self.preview_label.setText("正在打开摄像头...")
        self.status_label.setText("等待")
        self.fps_label.setText("FPS: --")
        self.recording_status_label.setText("未录像")
        if self.user_recording_output_path:
            self.recording_path_edit.setText(str(self.user_recording_output_path))
        self.preview_mtime = 0.0
        started = self.start_process(self.command(), "Realtime embedded worker started.")
        if started:
            self.preview_timer.start()
            self.status_timer.start()

    def restart_worker(self) -> None:
        self.stop_worker(wait_ms=900)
        self.start_worker()

    def stop_worker(self, wait_ms: int = 1500) -> None:
        self.recording_requested = False
        self.record_button.setText("开始录像")
        self.write_controls(stop_signal=True)
        self.preview_timer.stop()
        self.status_timer.stop()
        if self.process.state() != QProcess.ProcessState.NotRunning:
            if not self.process.waitForFinished(wait_ms):
                self.process.terminate()
                if not self.process.waitForFinished(900):
                    self.process.kill()
        self.status_label.setText("已关闭")
        self.window.statusBar().showMessage("realtime stopped")

    def toggle_recording(self) -> None:
        if self.recording_requested:
            self.recording_requested = False
            self.record_button.setText("开始录像")
            self.recording_status_label.setText("正在保存...")
            self.write_controls(stop_signal=False)
            return
        if self.process.state() == QProcess.ProcessState.NotRunning:
            QMessageBox.warning(self, "Stage4", "请先进入实时视频并等待摄像头 worker 启动。")
            return
        RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
        if self.user_recording_output_path:
            self.recording_output_path = uniquify_path(self.user_recording_output_path)
            self.user_recording_output_path = self.recording_output_path
        else:
            self.recording_output_path = RECORDINGS_DIR / "live_recording_{}.mp4".format(timestamp())
        self.recording_requested = True
        self.record_button.setText("停止录像")
        self.recording_status_label.setText("录像中")
        self.recording_path_edit.setText(str(self.recording_output_path))
        self.write_controls(stop_signal=False)
        self.window.statusBar().showMessage("开始录像: {}".format(self.recording_output_path))

    def refresh_preview(self) -> None:
        if not LIVE_PREVIEW.exists():
            return
        mtime = LIVE_PREVIEW.stat().st_mtime
        if mtime <= self.preview_mtime:
            return
        pixmap = QPixmap(str(LIVE_PREVIEW))
        if pixmap.isNull():
            return
        self.preview_mtime = mtime
        self.last_pixmap = pixmap
        self._apply_pixmap()

    def _apply_pixmap(self) -> None:
        if not self.last_pixmap:
            return
        scaled = self.last_pixmap.scaled(
            self.preview_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.preview_label.setPixmap(scaled)

    def refresh_status(self) -> None:
        status = read_json(LIVE_STATUS)
        if not status:
            return
        if status.get("camera_opened"):
            self.status_label.setText("处理中" if status.get("running") else "已连接")
        elif status.get("last_error"):
            self.status_label.setText("失败")
            self.preview_label.setText("无法打开摄像头，请检查 macOS 系统设置 > 隐私与安全性 > 摄像头 权限。")
        else:
            self.status_label.setText("等待")
        fps = status.get("fps")
        self.fps_label.setText("FPS: {}".format("{:.2f}".format(float(fps)) if fps is not None else "--"))
        enabled = status.get("enabled_effects") or self.effects.selected()
        self.effects_label.setText("启用特效: {}".format(", ".join(enabled) or "none"))
        self.refresh_recording_status(status)
        if status.get("last_error"):
            self.append_log(str(status["last_error"]))

    def refresh_recording_status(self, status: Dict) -> None:
        recording_error = status.get("recording_error")
        if recording_error:
            self.recording_status_label.setText("录像失败：{}".format(recording_error))
            self.record_button.setText("开始录像")
            self.recording_requested = False
            return
        if status.get("recording"):
            count = status.get("recording_frame_count") or 0
            self.recording_status_label.setText("录像中：{} 帧".format(count))
            path = status.get("recording_output_path")
            if path:
                self.recording_path_edit.setText(str(path))
            return
        last_path = status.get("last_recording_path")
        if status.get("last_recording_saved") and last_path:
            count = status.get("last_recording_frame_count")
            self.recording_status_label.setText("已保存：{} 帧".format(count if count is not None else "--"))
            self.recording_path_edit.setText(str(last_path))
            self.window.statusBar().showMessage("录像已保存: {}".format(last_path))
            self.record_button.setText("开始录像")
            self.recording_requested = False
        elif not self.recording_requested:
            self.recording_status_label.setText("未录像")

    def save_screenshot(self) -> None:
        if not self.last_pixmap:
            self.window.statusBar().showMessage("没有可保存的实时画面")
            return
        SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
        path = SCREENSHOT_DIR / "live_snapshot_{}.jpg".format(timestamp())
        if self.last_pixmap.save(str(path), "JPG"):
            self.window.statusBar().showMessage("截图已保存: {}".format(path))
            self.append_log("snapshot={}".format(path))
        else:
            self.window.statusBar().showMessage("截图保存失败")

    def choose_recording_path(self) -> None:
        default_path = RECORDINGS_DIR / "live_recording_{}.mp4".format(timestamp())
        path, _ = QFileDialog.getSaveFileName(self, "选择录像保存位置", str(default_path), "MP4 Video (*.mp4)")
        if path:
            selected = Path(path).expanduser()
            if selected.suffix.lower() != ".mp4":
                selected = selected.with_suffix(".mp4")
            self.user_recording_output_path = selected
            self.recording_output_path = selected
            self.recording_path_edit.setText(str(selected))
            self.write_controls(stop_signal=False)

    def open_recordings_dir(self) -> None:
        RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
        try:
            subprocess.Popen(["open", str(RECORDINGS_DIR)])
        except Exception as exc:
            self.append_log("Could not open recordings directory: {}".format(exc))

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._apply_pixmap()

    def on_started(self) -> None:
        super().on_started()
        self.status_label.setText("正在打开")

    def on_finished(self, exit_code: int, exit_status) -> None:
        super().on_finished(exit_code, exit_status)
        self.preview_timer.stop()
        self.status_timer.stop()
        if exit_code == 0 and self.status_label.text() != "已关闭":
            self.status_label.setText("已停止")
        elif exit_code != 0:
            self.status_label.setText("失败")

    def close_process(self) -> None:
        self.stop_worker(wait_ms=900)


class MainWindow(QMainWindow):
    def __init__(self, safe_mode: bool = False) -> None:
        super().__init__()
        ensure_dirs()
        self.safe_mode = safe_mode
        self.setWindowTitle("Stage4 人脸视觉特效应用")
        self.resize(1280, 820)
        self.setMinimumSize(960, 640)
        self.setStyleSheet(APP_STYLE)
        self.stack = QStackedWidget()
        self.home = HomePage(self)
        self.local = LocalImportPage(self)
        self.realtime = RealtimeCameraPage(self)
        self.stack.addWidget(self.home)
        self.stack.addWidget(self.local)
        self.stack.addWidget(self.realtime)
        self.setCentralWidget(self.stack)
        self.show_home_page()
        if self.safe_mode:
            self.statusBar().showMessage("Safe mode: GUI only")
        else:
            self.statusBar().showMessage("首次进入实时视频时，macOS 可能请求摄像头权限。")
        write_ui_check_summary([], self.safe_mode, "Stage4 UI V3 opened.")
        write_ui_v3_summary([], "Stage4 UI V3 opened.")

    def show_home_page(self) -> None:
        self.realtime.stop_worker(wait_ms=700)
        self.home.refresh_status()
        self.stack.setCurrentWidget(self.home)

    def show_local_page(self) -> None:
        self.realtime.stop_worker(wait_ms=700)
        self.stack.setCurrentWidget(self.local)

    def show_realtime_page(self) -> None:
        self.stack.setCurrentWidget(self.realtime)
        self.realtime.start_worker_if_needed()

    def closeEvent(self, event) -> None:
        self.local.close_process()
        self.realtime.close_process()
        super().closeEvent(event)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--safe", action="store_true", help="Open GUI only; do not start CV worker subprocesses.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = QApplication(sys.argv)
    window = MainWindow(safe_mode=args.safe)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
