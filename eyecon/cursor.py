"""Win32 cursor control — move the mouse and generate click events."""
from __future__ import annotations

import ctypes
import threading

from config import ScreenConfig

# Win32 constants.
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010

user32 = ctypes.windll.user32


def set_dpi_aware() -> None:
    """Mark this process as DPI-aware so coordinates are in physical pixels."""
    try:
        user32.SetProcessDPIAware()
    except Exception:
        pass


class CursorController:
    def __init__(self, screen_config: ScreenConfig):
        self._scr = screen_config
        self._enabled = True
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        with self._lock:
            return self._enabled

    def set_enabled(self, value: bool) -> None:
        with self._lock:
            self._enabled = value

    def toggle_enabled(self) -> bool:
        with self._lock:
            self._enabled = not self._enabled
            return self._enabled

    def move(self, x: float, y: float) -> None:
        if not self.enabled:
            return
        ix = max(0, min(int(x), self._scr.width - 1))
        iy = max(0, min(int(y), self._scr.height - 1))
        user32.SetCursorPos(ix, iy)

    def left_click(self) -> None:
        if not self.enabled:
            return
        user32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
        user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)

    def right_click(self) -> None:
        if not self.enabled:
            return
        user32.mouse_event(MOUSEEVENTF_RIGHTDOWN, 0, 0, 0, 0)
        user32.mouse_event(MOUSEEVENTF_RIGHTUP, 0, 0, 0, 0)
