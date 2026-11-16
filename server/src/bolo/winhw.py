"""What Windows says about the CPU and the graphics adapters, through ``ctypes`` (no extra package).

* CPU instruction sets: ``IsProcessorFeaturePresent`` (SSE4.2, AVX, AVX2, AVX-512F on x86-64; NEON, dot product
  and SVE on ARM64). A Windows release that predates a feature's constant answers "no" for it.
* The CPU's name: ``ProcessorNameString`` in the registry (``platform.processor()`` gives only the family).
* Graphics adapters: DXGI (``CreateDXGIFactory1``, ``EnumAdapters1``, ``GetDesc1``): name, PCI vendor, dedicated
  video memory, and with ``IDXGIAdapter3::QueryVideoMemoryInfo`` the local memory budget the system gives this
  process and what this process uses of it. The adapter's position in that list is DirectML's ``device_id``
  (``device = "dml:N"``), as ONNX Runtime documents it.

Every function returns None when it cannot ask (another system, an old Windows, an error) and never raises.
"""

from __future__ import annotations

import contextlib
import sys
import uuid
from dataclasses import dataclass
from typing import Any

# winnt.h: PF_* for IsProcessorFeaturePresent
_X86_FEATURES = (
    (38, "sse4.2"),  # PF_SSE4_2_INSTRUCTIONS_AVAILABLE
    (39, "avx"),  # PF_AVX_INSTRUCTIONS_AVAILABLE
    (40, "avx2"),  # PF_AVX2_INSTRUCTIONS_AVAILABLE
    (41, "avx512"),  # PF_AVX512F_INSTRUCTIONS_AVAILABLE
)
_ARM_FEATURES = (
    (29, "neon"),  # PF_ARM_V8_INSTRUCTIONS_AVAILABLE (Advanced SIMD is part of ARMv8-A)
    (43, "dotprod"),  # PF_ARM_V82_DP_INSTRUCTIONS_AVAILABLE
    (46, "sve"),  # PF_ARM_SVE_INSTRUCTIONS_AVAILABLE
    (47, "sve2"),  # PF_ARM_SVE2_INSTRUCTIONS_AVAILABLE
)

IID_IDXGIFactory1 = "770aae78-f26f-4dba-a829-253c83d1b387"
IID_IDXGIAdapter3 = "645967a4-1392-4310-a798-8053ce3e93fd"
DXGI_ADAPTER_FLAG_SOFTWARE = 2
DXGI_ERROR_NOT_FOUND = 0x887A0002
PCI_VENDORS = {0x10DE: "nvidia", 0x1002: "amd", 0x1022: "amd", 0x8086: "intel", 0x1414: "microsoft", 0x5143: "qualcomm"}


@dataclass
class Adapter:
    #: position in DXGI's list (DirectML's device_id), software adapters included
    index: int
    name: str
    vendor_id: int
    dedicated_bytes: int
    shared_bytes: int
    software: bool
    #: local video memory the system currently lets this process use (DXGI_QUERY_VIDEO_MEMORY_INFO.Budget)
    budget_bytes: int | None = None
    #: local video memory this process uses now
    usage_bytes: int | None = None

    @property
    def vendor(self) -> str:
        return PCI_VENDORS.get(self.vendor_id, "unknown")


def cpu_features(machine: str) -> list[str] | None:
    """Instruction sets Windows reports for this CPU, named as bolo.hardware names them."""
    if sys.platform != "win32":
        return None
    try:
        import ctypes

        present = ctypes.windll.kernel32.IsProcessorFeaturePresent
        table = _ARM_FEATURES if machine.lower() in ("arm64", "aarch64") else _X86_FEATURES
        return [name for number, name in table if present(number)]
    except Exception:
        return None


def cpu_name() -> str | None:
    """The CPU's marketing name ("AMD Ryzen 7 5800H with Radeon Graphics")."""
    if sys.platform != "win32":
        return None
    try:
        import winreg

        key = r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key) as handle:
            value, _ = winreg.QueryValueEx(handle, "ProcessorNameString")
        return " ".join(str(value).split()) or None
    except Exception:
        return None


def _guid(text: str) -> Any:
    import ctypes

    class GUID(ctypes.Structure):
        _fields_ = [("raw", ctypes.c_ubyte * 16)]

    return GUID.from_buffer_copy(uuid.UUID(text).bytes_le)


def _method(obj: Any, index: int, restype: Any, *argtypes: Any) -> Any:
    """Method ``index`` of a COM object's vtable, bound to ``obj``."""
    import ctypes

    vtable = ctypes.cast(obj, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
    prototype = ctypes.WINFUNCTYPE(restype, ctypes.c_void_p, *argtypes)  # type: ignore[attr-defined,unused-ignore]
    function = prototype(vtable[index])
    return lambda *args: function(obj, *args)


def _release(obj: Any) -> None:
    import ctypes

    if obj:
        with contextlib.suppress(Exception):
            _method(obj, 2, ctypes.c_ulong)()


def dxgi_adapters(include_software: bool = False) -> list[Adapter] | None:
    """The graphics adapters DXGI lists, in its order (the first is the one Windows prefers for display)."""
    if sys.platform != "win32":
        return None
    try:
        return _dxgi_adapters(include_software)
    except Exception:
        return None


def _dxgi_adapters(include_software: bool) -> list[Adapter]:
    import ctypes
    from ctypes import wintypes

    class AdapterDesc1(ctypes.Structure):
        _fields_ = [
            ("Description", ctypes.c_wchar * 128),
            ("VendorId", wintypes.UINT),
            ("DeviceId", wintypes.UINT),
            ("SubSysId", wintypes.UINT),
            ("Revision", wintypes.UINT),
            ("DedicatedVideoMemory", ctypes.c_size_t),
            ("DedicatedSystemMemory", ctypes.c_size_t),
            ("SharedSystemMemory", ctypes.c_size_t),
            ("AdapterLuidLow", wintypes.DWORD),
            ("AdapterLuidHigh", wintypes.LONG),
            ("Flags", wintypes.UINT),
        ]

    class VideoMemoryInfo(ctypes.Structure):
        _fields_ = [
            ("Budget", ctypes.c_uint64),
            ("CurrentUsage", ctypes.c_uint64),
            ("AvailableForReservation", ctypes.c_uint64),
            ("CurrentReservation", ctypes.c_uint64),
        ]

    hresult = ctypes.c_long
    dxgi = ctypes.WinDLL("dxgi")  # type: ignore[attr-defined,unused-ignore]
    create = dxgi.CreateDXGIFactory1
    create.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
    create.restype = hresult
    factory = ctypes.c_void_p()
    if create(ctypes.byref(_guid(IID_IDXGIFactory1)), ctypes.byref(factory)) < 0 or not factory:
        return []
    adapters: list[Adapter] = []
    try:
        enum_adapters1 = _method(factory, 12, hresult, wintypes.UINT, ctypes.POINTER(ctypes.c_void_p))
        for index in range(64):
            adapter = ctypes.c_void_p()
            hr = enum_adapters1(index, ctypes.byref(adapter))
            if hr < 0 or not adapter:
                break  # DXGI_ERROR_NOT_FOUND: the end of the list
            try:
                desc = AdapterDesc1()
                if _method(adapter, 10, hresult, ctypes.POINTER(AdapterDesc1))(ctypes.byref(desc)) < 0:
                    continue
                software = bool(desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)
                if software and not include_software:
                    continue
                found = Adapter(
                    index=index,
                    name=desc.Description.strip(),
                    vendor_id=int(desc.VendorId),
                    dedicated_bytes=int(desc.DedicatedVideoMemory),
                    shared_bytes=int(desc.SharedSystemMemory),
                    software=software,
                )
                adapter3 = ctypes.c_void_p()
                query_interface = _method(adapter, 0, hresult, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p))
                if query_interface(ctypes.byref(_guid(IID_IDXGIAdapter3)), ctypes.byref(adapter3)) >= 0 and adapter3:
                    try:
                        info = VideoMemoryInfo()
                        query = _method(
                            adapter3, 14, hresult, wintypes.UINT, ctypes.c_int, ctypes.POINTER(VideoMemoryInfo)
                        )
                        if query(0, 0, ctypes.byref(info)) >= 0:  # node 0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL
                            found.budget_bytes = int(info.Budget)
                            found.usage_bytes = int(info.CurrentUsage)
                    finally:
                        _release(adapter3)
                adapters.append(found)
            finally:
                _release(adapter)
    finally:
        _release(factory)
    return adapters
