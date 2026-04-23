# Panda3D runtime configuration for vj-nano.
#
# Loaded automatically via `loadPrcFile('config/panda3d.prc')` in render/app.py.
# Values here can be overridden per-run by `loadPrcFileData(...)` from Python.

# --- window ---
window-title vj-nano
win-size 1280 720
framebuffer-srgb 0         # we do our own color management in shaders
framebuffer-multisample 0  # no MSAA — PS1 aesthetic doesn't want it
framebuffer-stencil 0
sync-video 1               # vsync on — smoother for projector use

# --- disable fluff we don't need ---
audio-library-name null    # we handle audio ourselves via sounddevice
show-frame-rate-meter 0    # toggled at runtime with --fps flag
load-file-type p3assimp    # lets us import .glb/.fbx without pre-conversion

# --- quality / perf ---
gl-version 3 2             # prefer GL 3.2 core (still fine on Tegra)
textures-power-2 none      # don't waste VRAM padding textures
texture-minfilter nearest  # PS1 look: no bilinear
texture-magfilter nearest
texture-anisotropic-degree 1

# --- logging ---
notify-level-glgsg warning
notify-level-display warning
