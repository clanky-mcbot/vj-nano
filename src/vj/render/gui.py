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

    def __init__(self, base, visualizer, filters, debug_nodes, milkdrop=None):
        # type: (ShowBase, object, object, list, object) -> None
        self._vis = visualizer
        self._filters = filters
        self._debug_nodes = debug_nodes
        self._milkdrop = milkdrop
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

        y = 0.72  # start slightly lower
        y = self._add_header(y, "VISUAL EFFECTS")
        y = self._build_fx(y)

        # MilkDrop section (between FX and Debug)
        if milkdrop is not None:
            y -= 0.015
            y = self._add_header(y, "MILKDROP")
            y = self._build_milkdrop(y)

        y -= 0.015
        y = self._add_header(y, "DEBUG (press H)")
        y = self._build_debug(y)

        y -= 0.015
        y = self._add_header(y, "POST-PROCESS")
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
            scale=0.032,
            pos=(0, 0, y),
            text_fg=(0.0, 0.6, 1.0, 1.0),
            parent=self._panel,
        )
        return y - 0.035

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

    def _build_milkdrop(self, start_y):
        # type: (float) -> float
        y = start_y
        # Toggle checkbox
        row = DirectFrame(
            frameColor=(0, 0, 0, 0),
            frameSize=(-0.42, 0.42, -0.03, 0.03),
            pos=(0, 0, y),
            parent=self._panel,
        )
        cb = DirectCheckButton(
            text="Enable",
            text_scale=0.42,
            scale=0.06,
            text_pos=(0.50, 0),
            text_fg=(0.90, 0.90, 0.95, 1.0),
            pos=(-0.36, 0, 0),
            parent=row,
            boxPlacement="left",
            command=self._make_milkdrop_toggle(),
            frameColor=(0.10, 0.10, 0.12, 1.0),
        )
        cb['indicatorValue'] = self._get_initial_state("milkdrop", "fx")
        self._checks["milkdrop"] = cb
        y -= 0.06

        # Preset label (read-only) — p key cycles
        DirectLabel(
            text="Preset: " + self._milkdrop.preset_label + "  [P]",
            text_scale=0.35,
            scale=0.05,
            pos=(-0.30, 0, y + 0.02),
            text_fg=(0.0, 1.0, 0.7, 1.0),
            parent=self._panel,
            frameColor=(0, 0, 0, 0),
        )

        # Bind 'P' key to cycle presets
        self._base.accept("p", self._cycle_milkdrop_preset)
        self._base.accept("P", self._cycle_milkdrop_preset)

        return y - 0.03

    def _make_milkdrop_toggle(self):
        def _cb(val):
            if self._milkdrop is not None:
                self._milkdrop.set_enabled(bool(val))
        return _cb

    def _cycle_milkdrop_preset(self):
        if self._milkdrop is None:
            return
        label = self._milkdrop.next_preset()
        # Rebuild the presets section to update the label
        # For simplicity, just print — manual rebuild on next M-menu open
        print("[gui] MilkDrop preset:", label)

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
            frameSize=(-0.42, 0.42, -0.03, 0.03),
            pos=(0, 0, y),
            parent=self._panel,
        )

        cb = DirectCheckButton(
            text=label,
            text_scale=0.42,
            scale=0.06,
            text_pos=(0.50, 0),
            text_fg=(0.90, 0.90, 0.95, 1.0),
            pos=(-0.36, 0, 0),
            parent=row,
            boxPlacement="left",
            command=self._make_toggle(key, section),
            frameColor=(0.10, 0.10, 0.12, 1.0),
        )
        cb['indicatorValue'] = self._get_initial_state(key, section)

        if has_slider:
            s = DirectSlider(
                range=(0.0, 2.0),
                value=1.0,
                pageSize=0.1,
                scale=0.13,
                pos=(0.14, 0, 0),
                parent=row,
                frameSize=(-0.5, 0.5, -0.06, 0.06),
            )
            self._sliders[key] = s
            s['command'] = self._make_slider_cmd(key)

        self._checks[key] = cb
        return y - 0.06

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
        """Toggle all debug nodes (bound to 'H' key)."""
        pass

    def _get_initial_state(self, key, section):
        # type: (str, str) -> int
        """Return 1 (checked) or 0 (unchecked) based on actual enabled state."""
        if section == "fx" and self._vis is not None:
            if hasattr(self._vis, "enabled"):
                return 1 if self._vis.enabled.get(key, True) else 0
            return 1
        elif section == "filter" and self._filters is not None:
            if hasattr(self._filters, "_enabled"):
                return 1 if self._filters._enabled.get(key, True) else 0
            return 1
        elif section == "debug":
            return 1  # debug toggles always start visible
        elif section == "fx" and key == "milkdrop":
            return 1  # MilkDrop starts enabled
        return 1
