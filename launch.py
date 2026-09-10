from __future__ import annotations

import os
import tempfile
import ctypes


def prepare_isaac_environment() -> None:
    """Make Isaac Sim's Windows USD DLL search path deterministic.

    On this workstation the Codex/WindowsApps entries can be inherited by
    Python. USD tries to pass every existing PATH entry to
    ``os.add_dll_directory``; WindowsApps is protected and raises WinError 5.
    Keep the normal environment but remove only those protected entries before
    importing ``isaacsim``.
    """

    path_entries = [
        entry
        for entry in os.environ.get("PATH", "").split(os.pathsep)
        if entry and "windowsapps" not in entry.lower()
    ]
    safe_path = os.pathsep.join(path_entries)
    os.environ["PATH"] = safe_path
    os.environ["PXR_USD_WINDOWS_DLL_PATH"] = safe_path
    os.environ["PYTHONNOUSERSITE"] = "1"

    # Warp's default cache on this installation can be left as a file after a
    # failed Kit start, which causes WinError 183 on the next run. Use a fresh,
    # writable cache directory for each runner process instead.
    cache_base = os.path.join(os.environ.get("TEMP", os.getcwd()), "r1_isaac_cache")
    os.makedirs(cache_base, exist_ok=True)
    os.environ["WARP_CACHE_PATH"] = tempfile.mkdtemp(prefix="warp_", dir=cache_base)

    # The USD references in spacerobot.usd contain many large textures.  The
    # default OV cache on this Windows account is not writable, which makes
    # every new Kit process retry texture compilation and appear frozen.
    ov_cache = os.path.join(cache_base, "ov_cache")
    os.makedirs(ov_cache, exist_ok=True)
    os.environ["OV_CACHE_PATH"] = ov_cache


def available_system_memory_mb() -> int:
    """Return available physical RAM on Windows, or -1 if unavailable."""

    try:
        class _MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = _MemoryStatusEx()
        status.dwLength = ctypes.sizeof(_MemoryStatusEx)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.ullAvailPhys // (1024 * 1024))
    except Exception:
        pass
    return -1


def simulation_app_config(headless: bool, renderer: str = "RayTracedLighting") -> dict:
    """Return one consistent Isaac Sim launch configuration.

    ``OV_CACHE_PATH`` is not honored by every Kit extension on this Windows
    installation.  Pass the cache locations as Kit settings as well, so a
    headless dataset run does not repeatedly retry writes to the protected
    per-user cache.  Unknown settings are harmless in Kit and older installs
    simply keep their normal cache behavior.
    """

    cache_root = os.path.join("D:\\", "isaac_temp", "r1_dataset_cache")
    texture_cache = os.path.join(cache_root, "texturecache")
    material_cache = os.path.join(cache_root, "materialcache")
    for path in (cache_root, texture_cache, material_cache):
        try:
            os.makedirs(path, exist_ok=True)
        except OSError:
            pass
    return {
        "headless": bool(headless),
        # A pure-physics diagnostic must not spend scarce GPU/CPU time
        # rendering the textured spacerobot stage.  Dataset capture and GUI
        # preview keep the normal RTX renderer explicitly.
        "renderer": str(renderer),
        "extra_args": [
            f"--/app/cachePath={cache_root}",
            f"--/persistent/app/cachePath={cache_root}",
            f"--/rtx/texturecache/path={texture_cache}",
            f"--/rtx/textureCache/path={texture_cache}",
            f"--/app/materialCachePath={material_cache}",
        ],
    }
