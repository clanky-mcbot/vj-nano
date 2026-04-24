"""DirectGui-based effect control menu for vj-nano.

Toggle panel with the 'M' key.  Each effect gets a checkbox + intensity slider.
Sections: FX, DEBUG, FILTERS.

Python 3.6 compatible.
"""

from __future__ import print_function

from direct.gui.DirectGui import (
    DirectFrame, DirectCheckButton, DirectSlider, DirectLabel,
)


class EffectMenu(object):
    """In-app menu for toggling effects, debug overlays, and filters."""

    def __init__(self, base, visualizer, filters, debug_nodes):
        # type: (ShowBase, object, object, list) -> None
        self._vis = visualizer
        self._filters = filters
        self._debug_nodes = debug_nodes
        self._open = False
        self._base = base

        # Semi-transparent panel (starts hidden via scale)
        self._panel = DirectFrame(
            frameColor=(0.05, 0.05, 0.07, 0.95),
            frameSize=(-0.44, 0.44, -1.15, 0.88),
            pos=(0.52, 0, 0.0),
            parent=base.aspect2d,
        )
        self._panel.setScale(0.01)

        self._sliders = {}  # type: Dict[str, DirectSlider]
        self._checks  = {}  # type: Dict[str, DirectCheckButton]

        y = 0.76
        y = self._add_header(y, "VISUAL EFFECTS")
        y = self._build_fx(y)

        y -= 0.04
        y = self._add_header(y, "DEBUG OVERLAY  (press H to toggle)")
        y = self._build_debug(y)

        y -= 0.04
        y = self._add_header(y, "POST-PROCESS FILTERS")
        y = self._build_filters(y)

        # default motion diff intensity lower so it doesn't dominate
        if hasattr(self._vis, "set_intensity"):
            self._vis.set_intensity("motion", 0.5)

        # Keyboard toggle: press 'm' to show/hide panel
        base.accept("m", self._toggle)
        base.accept("M", self._toggle)

    # ------------------------------------------------------------------ #
    def _add_header(self, y, text):
        # type: (float, str) -> float
        DirectLabel(
            text=text,
            scale=0.038,
            pos=(0, 0, y),
            text_fg=(0.0, 0.6, 1.0, 1.0),
            parent=self._panel,
        )
        return y - 0.06

    def _build_fx(self, start_y):
        # type: (float) -> float
        y = start_y
        for key, label in [
            ("ring", "Wave Ring"),
            ("stars", "Starfield"),
            ("grid", "Grid Floor"),
            ("cylinder", "Spectrum Cyl"),
            ("helix", "Wave Helix"),
            ("burst", "Beat Burst"),
            ("spec2d", "Spectrum Bars"),
            ("scope2d", "Wave Scope"),
            ("radial", "Radial Spectrum"),
            ("motion", "Motion Diff"),
            ("plasma", "Plasma BG"),
            ("superscope", "Superscope 3D"),
            ("water", "Water Plane"),
            ("vortex", "Vortex Swarm"),
            ("trail", "Motion Trail"),
            ("waterfall", "Spectrum Waterfall"),
            ("glow", "Beat Glow"),
        ]:
            y = self._add_row(y, key, label, has_slider=True)
        return y

    def _build_debug(self, start_y):
        # type: (float) -> float
        y = start_y
        for key, label in [
            ("webcam", "Webcam Preview"),
            ("bpm", "BPM / Stats Text"),
            ("waveform", "Waveform Scope"),
        ]:
            y = self._add_row(y, key, label, has_slider=False, section="debug")
        return y

    def _build_filters(self, start_y):
        # type: (float) -> float
        y = start_y
        for key, label in [
            ("dither", "Dither (ordered)"),
            ("pixelate", "Pixelate"),
            ("ascii", "ASCII Art (CPU)"),
        ]:
            y = self._add_row(y, key, label, has_slider=False, section="filter")
        return y

    def _add_row(self, y, key, label, has_slider=True, section="fx"):
        # type: (float, str, str, bool, str) -> float
        row = DirectFrame(
            frameColor=(0, 0, 0, 0),
            frameSize=(-0.42, 0.42, -0.04, 0.04),
            pos=(0, 0, y),
            parent=self._panel,
        )

        cb = DirectCheckButton(
            text=label,
            text_scale=0.55,
            scale=0.075,
            text_pos=(0.65, 0),
            text_fg=(0.90, 0.90, 0.95, 1.0),
            pos=(-0.36, 0, 0),
            parent=row,
            boxPlacement="left",
            command=self._make_toggle(key, section),
            frameColor=(0.10, 0.10, 0.12, 1.0),
        )
        cb['indicatorValue'] = 1

        if has_slider:
            s = DirectSlider(
                range=(0.0, 2.0),
                value=1.0,
                pageSize=0.1,
                scale=0.16,
                pos=(0.18, 0, 0),
                parent=row,
                frameSize=(-0.5, 0.5, -0.08, 0.08),
            )
            self._sliders[key] = s
            s['command'] = self._make_slider_cmd(key)

        self._checks[key] = cb
        return y - 0.095

    # ------------------------------------------------------------------ #
    def _toggle(self):
        self._open = not self._open
        if self._open:
            self._panel.setScale(1)
        else:
            self._panel.setScale(0.01)

    def _make_toggle(self, key, section):
        # type: (str, str) -> Callable[[int], None]
        def _cb(val):
            # type: (int) -> None
            if section == "fx":
                if hasattr(self._vis, "set_enabled"):
                    self._vis.set_enabled(key, bool(val))
            elif section == "debug":
                self._toggle_debug(key, bool(val))
            elif section == "filter":
                if self._filters is not None:
                    self._filters.set_enabled(key, bool(val))
        return _cb

    def _toggle_debug(self, key, show):
        # type: (str, bool) -> None
        if not self._debug_nodes:
            return
        if key == "webcam":
            # Last node is webcam card
            if len(self._debug_nodes) > 0:
                node = self._debug_nodes[-1]
                node.setScale(1 if show else 0.001)
        elif key == "bpm":
            # Text nodes: indices 1, 2, 3
            for idx in [1, 2, 3]:
                if idx < len(self._debug_nodes):
                    node = self._debug_nodes[idx]
                    node.setScale(1 if show else 0.001)
        elif key == "waveform":
            # Scope is index 0
            if len(self._debug_nodes) > 0:
                node = self._debug_nodes[0]
                node.setScale(1 if show else 0.001)

    def _make_slider_cmd(self, key):
        # type: (str) -> Callable[[], None]
        slider = self._sliders[key]

        def _cb():
            # type: () -> None
            val = slider["value"]
            if hasattr(self._vis, "set_intensity"):
                self._vis.set_intensity(key, float(val))
        return _cb

    def toggle_debug_all(self):
        # type: () -> None
        """Toggle all debug nodes (bound to 'H' key).

        NOTE: This method is kept for API compat but the actual H-key
        logic now lives in app.py._toggle_debug for state consistency.
        """
        pass
