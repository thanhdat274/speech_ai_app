"""
ui_main.py
=============================================================================
PySide6 UI Implementation - Production Ready MainWindow
=============================================================================
Integrates flawlessly with AppController for robust async AI processing.
"""

import sys
import psutil
from typing import Dict, Any

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QStackedWidget, QTextEdit, QComboBox,
    QProgressBar, QFrame, QFileDialog, QSizePolicy, QSlider, QCheckBox
)
from PySide6.QtCore import Qt, Slot, QTimer, QThread
from PySide6.QtGui import QFont, QColor, QTextCursor, QTextCharFormat, QCloseEvent

# Import the robust controller and config mapping
from config import setup_logger, SUPPORTED_LANGUAGES, TRANSLATION_MAPPING
from app_controller import AppController
from core.config_manager import ConfigManager
from core.audio_engine import AudioEngine # Needed for device listing in UI

logger = setup_logger("UI_Layer")


# =============================================================================
# THEME CONFIGURATION
# =============================================================================
STYLE_SHEET = """
QMainWindow {
    background-color: #0b0b0d;
}
QWidget {
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    color: #e1e1e6;
}
/* Sidebar Styles */
QFrame#Sidebar {
    background-color: #111113;
    border-right: 1px solid #1f1f22;
}
QPushButton.NavButton {
    background-color: transparent;
    border: none;
    text-align: left;
    padding: 14px 20px;
    font-size: 14px;
    font-weight: 600;
    color: #88888e;
    border-radius: 8px;
    margin: 4px 10px;
}
QPushButton.NavButton:hover {
    background-color: #1f1f22;
    color: #ffffff;
}
QPushButton.NavButton:checked {
    background-color: #10a37f22;
    color: #10a37f;
}
/* Panels */
QFrame.Panel {
    background-color: #161618;
    border-radius: 12px;
    border: 1px solid #2a2a2d;
}
/* Transcript Box & Inputs */
QTextEdit {
    background-color: #0b0b0d;
    border: 1px solid #2a2a2d;
    border-radius: 8px;
    padding: 15px;
    color: #e1e1e6;
    font-size: 14px;
    selection-background-color: #10a37f;
}
QTextEdit#TranscriptBox {
    font-family: 'Inter', 'Segoe UI Semibold';
    font-size: 15px;
    line-height: 1.7;
    border-radius: 12px;
}
/* Fix for AI Context Box */
QTextEdit#ContextPrompt {
    background-color: #111113;
    border: 1px solid #333336;
}

/* Combo Boxes (Critical Fix) */
QComboBox {
    background-color: #1f1f22;
    border: 1px solid #333336;
    border-radius: 6px;
    padding: 8px 12px;
    color: #ffffff;
    min-height: 25px;
}
QComboBox:hover { border-color: #10a37f; }
QComboBox::drop-down { border: none; width: 30px; }
QComboBox QAbstractItemView {
    background-color: #161618;
    border: 1px solid #333336;
    color: #ffffff;
    selection-background-color: #10a37f;
    outline: none;
    padding: 5px;
}

/* Headers */
QLabel.Header {
    font-size: 24px;
    font-weight: 800;
    color: #ffffff;
    letter-spacing: -0.5px;
}
QLabel.SubHeader {
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    color: #88888e;
    margin-top: 15px;
}

/* Action Buttons */
QPushButton.PrimaryAction {
    background-color: #10a37f;
    color: #ffffff;
    font-weight: 700;
    padding: 12px 20px;
    border-radius: 8px;
}
QPushButton.PrimaryAction:hover { background-color: #16b38c; }
QPushButton.PrimaryAction:disabled { background-color: #2a2a2d; color: #55555a; }

/* Dedicated Small Buttons (Clear, Stop, Secondary) */
QPushButton.SecondaryAction {
    background-color: #1f1f22;
    color: #e1e1e6;
    border: 1px solid #333336;
    border-radius: 8px;
    padding: 8px 15px;
    font-weight: 600;
}
QPushButton.SecondaryAction:hover { background-color: #2a2a2d; }

QPushButton#BtnLive {
    border: 1px solid #333336;
    border-radius: 8px;
    padding: 8px 15px;
    font-weight: 600;
    color: #e1e1e6;
    background-color: #1f1f22;
}
QPushButton#BtnLive[active="true"] {
    background-color: #ef4444; 
    border-color: #ef4444;
    color: white;
}
QPushButton#BtnClear {
    background-color: #1f1f22;
    border: 1px solid #333336;
    color: #e1e1e6;
    border-radius: 8px;
    padding: 8px 15px;
    font-weight: 600;
}
QPushButton#BtnClear:hover { background-color: #2a2a2d; color: #ef4444; border-color: #ef4444; }

/* Custom Scrollbar */
QScrollBar:vertical {
    border: none; background: transparent; width: 8px;
}
QScrollBar::handle:vertical {
    background: #2a2a2d; border-radius: 4px; min-height: 30px;
}
QScrollBar::handle:vertical:hover { background: #10a37f; }
"""

# =============================================================================
# UI PANELS
# =============================================================================
class HardwareInfoPanel(QFrame):
    def __init__(self):
        super().__init__()
        self.setProperty("class", "Panel")
        layout = QVBoxLayout(self)
        
        lbl_title = QLabel("Hardware Status")
        lbl_title.setProperty("class", "Header")
        layout.addWidget(lbl_title)
        
        self.lbl_cpu = QLabel("CPU: Calculating...")
        self.lbl_ram = QLabel("RAM: Calculating...")
        self.lbl_gpu = QLabel("GPU: Waiting for init...")
        
        layout.addWidget(self.lbl_cpu)
        layout.addWidget(self.lbl_ram)
        layout.addWidget(self.lbl_gpu)
        layout.addStretch()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._update_hardware_stats)
        self.timer.start(2000)
        self._update_hardware_stats()

    @Slot()
    def _update_hardware_stats(self):
        cpu_usage = psutil.cpu_percent()
        ram = psutil.virtual_memory()
        
        self.lbl_cpu.setText(f"CPU Usage: {cpu_usage}% ({psutil.cpu_count(logical=True)} logical cores)")
        self.lbl_ram.setText(f"RAM Usage: {ram.percent}% ({ram.used / (1024**3):.1f}GB / {ram.total / (1024**3):.1f}GB)")
        
        try:
            import torch
            if torch.cuda.is_available():
                name = torch.cuda.get_device_name(0)
                mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                self.lbl_gpu.setText(f"GPU: {name} (VRAM: {mem:.1f}GB)")
            else:
                self.lbl_gpu.setText("GPU: CUDA Not Available (CPU Fallback active)")
        except Exception:
            self.lbl_gpu.setText("GPU: Hardware undetected.")


class SettingsPanel(QFrame):
    def __init__(self):
        super().__init__()
        self.setProperty("class", "Panel")
        layout = QVBoxLayout(self)
        
        lbl_title = QLabel("Settings")
        lbl_title.setProperty("class", "Header")
        layout.addWidget(lbl_title)
        
        # Audio Device
        layout.addWidget(self._build_label("Audio Input Device (for Microphone)"))
        self.cm_device = QComboBox()
        self._populate_devices()
        layout.addWidget(self.cm_device)

        # Dual Mode Toggles
        layout.addWidget(self._build_label("Dual Mode Capture Settings"))
        self.chk_sys = QCheckBox("Enable System Audio (WASAPI)")
        self.chk_mic = QCheckBox("Enable Microphone")
        self.chk_sys.setChecked(True)
        self.chk_mic.setChecked(True)
        layout.addWidget(self.chk_sys)
        layout.addWidget(self.chk_mic)

        # Noise Suppression
        layout.addWidget(self._build_label("Noise Suppression Level"))
        self.slider_noise = QSlider(Qt.Horizontal)
        self.slider_noise.setRange(0, 100)
        self.slider_noise.setValue(5) # Default 0.005 -> 5
        self.lbl_noise_val = QLabel("Gate: 0.005")
        self.slider_noise.valueChanged.connect(lambda v: self.lbl_noise_val.setText(f"Gate: {v/1000:.3f}"))
        
        noise_layout = QHBoxLayout()
        noise_layout.addWidget(self.slider_noise)
        noise_layout.addWidget(self.lbl_noise_val)
        layout.addLayout(noise_layout)

        # AI Context / Keywords
        layout.addWidget(self._build_label("AI Context / Keywords (Improve Accuracy)"))
        self.txt_context = QTextEdit()
        self.txt_context.setPlaceholderText("Enter keywords or context here (e.g. AI, Machine Learning, Vietnam Today, specific names...)")
        self.txt_context.setFixedHeight(80)
        self.txt_context.setObjectName("ContextPrompt")
        layout.addWidget(self.txt_context)
        
        layout.addStretch()

    def _populate_devices(self):
        """Fetch all available WASAPI/Microphone inputs without duplicates."""
        try:
            import sounddevice as sd
            raw_devices = AudioEngine.list_devices()
            hostapis = sd.query_hostapis()
            
            # Dictionary to store the 'best' version of each physical device
            # Key: (Clean Name, IsSystem)
            unique_map = {}
            
            for d in raw_devices:
                if d.get('max_input_channels', 0) > 0:
                    name = d['name']
                    # Detect if it's a loopback/system device
                    is_system = any(kw in name.lower() for kw in ["loopback", "wasapi", "stereo mix", "wave out", "what u hear"])
                    dev_type = "System" if is_system else "Mic"
                    
                    # Check Host API (WASAPI is usually better on Windows)
                    api_name = hostapis[d['hostapi']]['name']
                    is_wasapi = "WASAPI" in api_name
                    
                    key = (name, dev_type)
                    
                    if key not in unique_map:
                        unique_map[key] = d
                    else:
                        # If we already have this device, replace it ONLY if current one is WASAPI
                        # and the previous one wasn't.
                        prev_d = unique_map[key]
                        prev_is_wasapi = "WASAPI" in hostapis[prev_d['hostapi']]['name']
                        if is_wasapi and not prev_is_wasapi:
                            unique_map[key] = d

            # Sort for a clean UI: System first, then Mic
            sorted_devices = sorted(unique_map.values(), key=lambda x: (
                not any(kw in x['name'].lower() for kw in ["loopback", "wasapi", "stereo mix"]),
                x['name']
            ))

            for d in sorted_devices:
                name = d['name']
                is_system = any(kw in name.lower() for kw in ["loopback", "wasapi", "stereo mix", "wave out", "what u hear"])
                icon = "🔊 [System]" if is_system else "🎤 [Mic]"
                
                display_name = f"{d['index']}: {icon} {name}"
                self.cm_device.addItem(display_name, d['index'])

        except Exception as e:
            self.cm_device.addItem("Error listing devices", -1)
            logger.error(f"UI Device Listing Error: {e}")

    def _build_label(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setProperty("class", "SubHeader")
        return lbl


class TranscriptPanel(QFrame):
    def __init__(self):
        super().__init__()
        self.setProperty("class", "Panel")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        self.selected_file_path = None
        
        # Header
        header_layout = QHBoxLayout()
        lbl_title = QLabel("Transcription Output")
        lbl_title.setProperty("class", "Header")
        
        self.btn_clear = QPushButton("Clear")
        self.btn_clear.setObjectName("BtnClear")
        self.btn_clear.clicked.connect(self._clear_transcript)
        self.btn_clear.setFixedWidth(80)
        
        header_layout.addWidget(lbl_title)
        header_layout.addStretch()
        header_layout.addWidget(self.btn_clear)
        layout.addLayout(header_layout)

        # File Chooser Row
        file_layout = QHBoxLayout()
        self.btn_select_file = QPushButton("Select File / Video")
        self.btn_select_file.setProperty("class", "SecondaryAction")
        self.btn_select_file.clicked.connect(self._open_file_dialog)
        
        self.lbl_file = QLabel("No file selected.")
        self.lbl_file.setStyleSheet("color: #a0a0a0;")
        
        self.btn_live = QPushButton("🔊 Start System Audio")
        self.btn_live.setObjectName("BtnLive")
        self.btn_live.setFixedWidth(130)
        
        file_layout.addWidget(self.btn_select_file)
        file_layout.addWidget(self.lbl_file, stretch=1)
        file_layout.addWidget(self.btn_live)
        layout.addLayout(file_layout)

        # --- Quick Config Bar (Added for Better UX) ---
        quick_config_frame = QFrame()
        quick_config_frame.setStyleSheet("background-color: #111113; border-radius: 8px; border: 1px solid #1f1f22;")
        quick_config_layout = QHBoxLayout(quick_config_frame)
        quick_config_layout.setContentsMargins(15, 10, 15, 10)
        quick_config_layout.setSpacing(20)

        # Model Group
        model_group = QVBoxLayout()
        lbl_model = QLabel("AI Model")
        lbl_model.setStyleSheet("color: #a0a0a0; font-size: 11px; font-weight: bold; border: none;")
        self.cm_perf = QComboBox()
        self.cm_perf.addItems(["Auto (Recommended)", "tiny", "base", "small", "medium", "large-v3", "large-v3-turbo"])
        self.cm_perf.setFixedWidth(140)
        model_group.addWidget(lbl_model)
        model_group.addWidget(self.cm_perf)
        
        # Source Group
        src_group = QVBoxLayout()
        lbl_src = QLabel("Input Language")
        lbl_src.setStyleSheet("color: #a0a0a0; font-size: 11px; font-weight: bold; border: none;")
        self.cm_src_lang = QComboBox()
        self.cm_src_lang.addItems(list(SUPPORTED_LANGUAGES.keys()))
        self.cm_src_lang.setFixedWidth(140)
        src_group.addWidget(lbl_src)
        src_group.addWidget(self.cm_src_lang)

        # Arrow Decor
        lbl_arrow = QLabel("➜")
        lbl_arrow.setStyleSheet("font-size: 18px; color: #10a37f; margin-top: 10px; border: none;")

        # Target Group
        tgt_group = QVBoxLayout()
        lbl_tgt = QLabel("Translate To")
        lbl_tgt.setStyleSheet("color: #a0a0a0; font-size: 11px; font-weight: bold; border: none;")
        self.cm_tgt_lang = QComboBox()
        self.cm_tgt_lang.addItems(list(TRANSLATION_MAPPING.keys()))
        self.cm_tgt_lang.setFixedWidth(140)
        tgt_group.addWidget(lbl_tgt)
        tgt_group.addWidget(self.cm_tgt_lang)

        # Compute Mode (Force selection here)
        compute_group = QVBoxLayout()
        lbl_comp = QLabel("Compute")
        lbl_comp.setStyleSheet("color: #a0a0a0; font-size: 11px; font-weight: bold; border: none;")
        self.cm_compute = QComboBox()
        self.cm_compute.addItems(["Auto", "Force CPU", "Force GPU"])
        self.cm_compute.setFixedWidth(100)
        compute_group.addWidget(lbl_comp)
        compute_group.addWidget(self.cm_compute)

        quick_config_layout.addLayout(model_group)
        quick_config_layout.addLayout(src_group)
        quick_config_layout.addWidget(lbl_arrow)
        quick_config_layout.addLayout(tgt_group)
        quick_config_layout.addStretch()
        quick_config_layout.addLayout(compute_group)
        
        layout.addWidget(quick_config_frame)
        
        # Text Edit
        self.text_edit = QTextEdit()
        self.text_edit.setObjectName("TranscriptBox")
        self.text_edit.setReadOnly(True)
        self._clear_transcript()
        layout.addWidget(self.text_edit, stretch=1)
        
        # Actions
        action_layout = QHBoxLayout()
        self.btn_action = QPushButton("Start Processing")
        self.btn_action.setProperty("class", "PrimaryAction")
        self.btn_action.setDisabled(True) # Disabled until file picked
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%p%")
        self.progress_bar.setVisible(False)
        
        action_layout.addWidget(self.progress_bar)
        action_layout.addWidget(self.btn_action)
        layout.addLayout(action_layout)

    def _open_file_dialog(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Audio/Video File", "", 
            "Media Files (*.mp3 *.wav *.m4a *.mp4);;All Files (*.*)"
        )
        if file_path:
            self.selected_file_path = file_path
            self.lbl_file.setText(file_path.split("/")[-1])
            self.btn_action.setDisabled(False)

    def _clear_transcript(self):
        self.text_edit.clear()
        self.text_edit.setPlaceholderText("Select a file and click 'Start Processing' to begin transcription...")

    @Slot(int)
    def update_progress(self, val: int):
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(val)

    @Slot(dict)
    def append_transcript(self, result: dict):
        self.progress_bar.setVisible(False)
        self.progress_bar.setValue(0)
        
        cursor = self.text_edit.textCursor()
        cursor.movePosition(QTextCursor.End)
        
        # Handle Speaker Tagging / Coloring seamlessly
        colors = {"SYS": QColor("#10a37f"), "MIC": QColor("#5b8def"), "SPEAKER_01": QColor("#5b8def"), "SPEAKER_02": QColor("#10a37f")}
        
        for speaker, text in result.get("speakers", []):
            fmt = QTextCharFormat()
            fmt.setForeground(colors.get(speaker, QColor("#a0a0a0")))
            fmt.setFontWeight(QFont.Bold)
            cursor.insertText(f"[{speaker}] ", fmt)
            
            fmt.setForeground(QColor("#ececec"))
            fmt.setFontWeight(QFont.Normal)
            cursor.insertText(f"{text}\n\n", fmt)
            
        self.text_edit.ensureCursorVisible()

    def set_loading_state(self, state: bool, mode: str = "file"):
        self.btn_action.setDisabled(state)
        self.btn_select_file.setDisabled(state)
        self.btn_clear.setDisabled(state)
        
        if mode == "file":
            self.btn_action.setText("Processing Pipeline..." if state else "Start Processing")
            self.btn_live.setDisabled(state)
        else:
            self.btn_live.setText("⏹️ Stop Capture" if state else "🔊 Start System Audio")
            self.btn_live.setStyleSheet("color: #ef4444;" if state else "")
            self.btn_action.setDisabled(state)


# =============================================================================
# MAIN WINDOW
# =============================================================================
class MainWindow(QMainWindow):
    """Central container managing UI architecture, routing, and AppController interaction."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Speech AI - Production Edition")
        self.resize(1100, 700)
        self.setStyleSheet(STYLE_SHEET)
        
        # -- 1. Integrate AppController Backend --
        self.controller = AppController()
        self.config = ConfigManager()
        
        # Setup Layout
        self._central_widget = QWidget()
        self.setCentralWidget(self._central_widget)
        self._main_layout = QVBoxLayout(self._central_widget)
        self._main_layout.setContentsMargins(0, 0, 0, 0)
        self._main_layout.setSpacing(0)
        
        self._build_content_area()
        self._build_status_bar()
        self._apply_saved_settings() # Match UI with config
        self._connect_signals()

    def _apply_saved_settings(self):
        """Restore previous user choices from settings.json."""
        self.page_transcript.cm_compute.setCurrentText(self.config.get("compute_mode", "Auto").replace("Force GPU (NVIDIA)", "Force GPU"))
        self.page_transcript.cm_perf.setCurrentText(self.config.get("whisper_model", "Auto (Recommended)"))
        self.page_transcript.cm_src_lang.setCurrentText(self.config.get("source_lang", "Auto-Detect"))
        self.page_transcript.cm_tgt_lang.setCurrentText(self.config.get("target_lang", "None (Off)"))
        
        self.page_settings.slider_noise.setValue(int(self.config.get("noise_gate", 0.005) * 1000))
        self.page_settings.chk_sys.setChecked(self.config.get("enable_sys", True))
        self.page_settings.chk_mic.setChecked(self.config.get("enable_mic", True))
        
        # Device matching
        saved_idx = self.config.get("input_device_index")
        if saved_idx is not None:
            for i in range(self.page_settings.cm_device.count()):
                if self.page_settings.cm_device.itemData(i) == saved_idx:
                    self.page_settings.cm_device.setCurrentIndex(i)
                    break

    def _save_current_settings(self):
        """Persist current UI state before processing or closing."""
        compute_val = self.page_transcript.cm_compute.currentText()
        if compute_val == "Force GPU": compute_val = "Force GPU (NVIDIA)"
        
        self.config.set("compute_mode", compute_val)
        self.config.set("whisper_model", self.page_transcript.cm_perf.currentText())
        self.config.set("source_lang", self.page_transcript.cm_src_lang.currentText())
        self.config.set("target_lang", self.page_transcript.cm_tgt_lang.currentText())
        
        self.config.set("noise_gate", self.page_settings.slider_noise.value() / 1000)
        self.config.set("input_device_index", self.page_settings.cm_device.currentData())
        self.config.set("enable_sys", self.page_settings.chk_sys.isChecked())
        self.config.set("enable_mic", self.page_settings.chk_mic.isChecked())

    def _build_content_area(self):
        content_frame = QWidget()
        content_layout = QHBoxLayout(content_frame)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)
        
        # Sidebar Setup
        self.sidebar = QFrame()
        self.sidebar.setObjectName("Sidebar")
        self.sidebar.setFixedWidth(250)
        sidebar_layout = QVBoxLayout(self.sidebar)
        
        lbl_logo = QLabel("💬 Speech AI")
        lbl_logo.setProperty("class", "Header")
        lbl_logo.setStyleSheet("padding: 20px; color: #10a37f;")
        sidebar_layout.addWidget(lbl_logo)
        
        self.btn_nav_transcript = self._create_nav_item("Transcript", checked=True)
        self.btn_nav_settings = self._create_nav_item("Settings")
        self.btn_nav_hardware = self._create_nav_item("Hardware")
        
        sidebar_layout.addWidget(self.btn_nav_transcript)
        sidebar_layout.addWidget(self.btn_nav_settings)
        sidebar_layout.addWidget(self.btn_nav_hardware)
        sidebar_layout.addStretch()
        
        # Stacked Widget (Pages)
        self.stack = QStackedWidget()
        self.stack.setContentsMargins(20, 20, 20, 20)
        
        self.page_transcript = TranscriptPanel()
        self.page_settings = SettingsPanel()
        self.page_hardware = HardwareInfoPanel()
        
        self.stack.addWidget(self.page_transcript)
        self.stack.addWidget(self.page_settings)
        self.stack.addWidget(self.page_hardware)
        
        # Wiring Nav to Pages
        self.btn_nav_transcript.clicked.connect(lambda: self._switch_page(0, self.btn_nav_transcript))
        self.btn_nav_settings.clicked.connect(lambda: self._switch_page(1, self.btn_nav_settings))
        self.btn_nav_hardware.clicked.connect(lambda: self._switch_page(2, self.btn_nav_hardware))

        content_layout.addWidget(self.sidebar)
        content_layout.addWidget(self.stack, stretch=1)
        self._main_layout.addWidget(content_frame, stretch=1)

    def _build_status_bar(self):
        self.status_bar_frame = QFrame()
        self.status_bar_frame.setObjectName("StatusBar")
        self.status_bar_frame.setFixedHeight(30)
        sb_layout = QHBoxLayout(self.status_bar_frame)
        sb_layout.setContentsMargins(15, 0, 15, 0)
        
        self.lbl_status = QLabel("System Ready")
        self.lbl_status.setStyleSheet("color: #a0a0a0; font-size: 12px;")
        
        sb_layout.addWidget(self.lbl_status)
        self._main_layout.addWidget(self.status_bar_frame)

    def _create_nav_item(self, text: str, checked: bool = False) -> QPushButton:
        btn = QPushButton(text)
        btn.setProperty("class", "NavButton")
        btn.setCheckable(True)
        btn.setChecked(checked)
        return btn

    @Slot(int, QPushButton)
    def _switch_page(self, index: int, active_btn: QPushButton):
        self.stack.setCurrentIndex(index)
        for btn in [self.btn_nav_transcript, self.btn_nav_settings, self.btn_nav_hardware]:
            if btn != active_btn:
                btn.setChecked(False)
            else:
                btn.setChecked(True)

    # =====================================================================
    # SIGNAL/SLOT WIRING (Connecting UI to AppController)
    # =====================================================================
    def _connect_signals(self):
        # UI Action requests Controller method
        self.page_transcript.btn_action.clicked.connect(self._trigger_backend_transcription)
        self.page_transcript.btn_live.clicked.connect(self._toggle_live_streaming)
        
        # Controller updates View via Signals
        self.controller.controller_status.connect(self._update_status_ui)
        self.controller.transcript_progress.connect(self.page_transcript.update_progress)
        self.controller.transcript_result.connect(self._handle_result_ui)
        self.controller.streaming_chunk_ready.connect(self.page_transcript.append_transcript)

    @Slot()
    def _trigger_backend_transcription(self):
        """Pass all UI parameters explicitly to the isolated Controller."""
        file_path = self.page_transcript.selected_file_path
        if not file_path:
            return
            
        self._save_current_settings()
        
        model_name = self.page_transcript.cm_perf.currentText()
        compute_mode = self.page_transcript.cm_compute.currentText()
        if compute_mode == "Force GPU": compute_mode = "Force GPU (NVIDIA)"
        
        src_lang = self.page_transcript.cm_src_lang.currentText()
        tgt_lang = self.page_transcript.cm_tgt_lang.currentText()

        self.page_transcript._clear_transcript()
        self.page_transcript.set_loading_state(True)
        
        # Non-blocking Controller invocation
        self.controller.start_file_transcription(
            path=file_path,
            model_name=model_name,
            src_lang=src_lang,
            trans_target=tgt_lang,
            compute_mode=compute_mode
        )

    @Slot()
    def _toggle_live_streaming(self):
        """Toggle the System Audio capture and Real-time transcription."""
        is_active = "Stop" in self.page_transcript.btn_live.text()
        
        if not is_active:
            # Start
            self._save_current_settings()
            
            model_name = self.page_transcript.cm_perf.currentText()
            compute_mode = self.page_transcript.cm_compute.currentText()
            if compute_mode == "Force GPU": compute_mode = "Force GPU (NVIDIA)"
            
            src_lang = self.page_transcript.cm_src_lang.currentText()
            tgt_lang = self.page_transcript.cm_tgt_lang.currentText()
            
            device_idx = self.page_settings.cm_device.currentData()
            noise_gate = self.page_settings.slider_noise.value() / 1000
            enable_sys = self.page_settings.chk_sys.isChecked()
            enable_mic = self.page_settings.chk_mic.isChecked()
            prompt = self.page_settings.txt_context.toPlainText().strip()
            
            if not enable_sys and not enable_mic:
                self._update_status_ui("Please enable at least one source (Mic or Sys).", "error")
                return

            self.page_transcript.set_loading_state(True, mode="live")
            self.controller.start_live_streaming(
                model_name=model_name,
                src_lang=src_lang,
                trans_target=tgt_lang,
                compute_mode=compute_mode,
                enable_sys=enable_sys,
                enable_mic=enable_mic,
                device_index=device_idx,
                noise_gate=noise_gate,
                prompt=prompt
            )
        else:
            # Stop
            self.controller.stop_live_streaming()
            self.page_transcript.set_loading_state(False, mode="live")

    @Slot(str, str)
    def _update_status_ui(self, msg: str, level: str):
        colors = {"success": "#10a37f", "error": "#ef4444", "active": "#5b8def", "idle": "#a0a0a0"}
        color = colors.get(level, "#a0a0a0")
        self.lbl_status.setStyleSheet(f"color: {color}; font-size: 13px; font-weight: bold;")
        self.lbl_status.setText(f"Status: {msg}")
        
        if level == "error":
            self.page_transcript.set_loading_state(False)

    @Slot(dict)
    def _handle_result_ui(self, result: dict):
        self.page_transcript.append_transcript(result)
        self.page_transcript.set_loading_state(False)
        duration = result.get("duration", 0)
        meta = result.get("metadata", "")
        self._update_status_ui(f"Transcription Complete | Processed in {duration:.1f}s | {meta}", "success")

    # =====================================================================
    # GRACEFUL SHUTDOWN
    # =====================================================================
    def closeEvent(self, event: QCloseEvent):
        """Intercept application window close event to ensure Memory is freed."""
        logger.info("Handling UI Shutdown Phase...")
        try:
            self.controller.graceful_shutdown()
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
        event.accept()


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion") 
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())
