"""What Windows says about the CPU and the graphics adapters, through ``ctypes`` (no extra package).

* CPU instruction sets: ``IsProcessorFeaturePresent`` (SSE4.2, AVX, AVX2, AVX-512F on x86-64; NEON, dot product
  and SVE on ARM64). A Windows release that predates a feature's constant answers "no" for it.
* The CPU's name: ``ProcessorNameString`` in the registry (``platform.processor()`` gives only the family).
* Graphics adapters: DXGI (``CreateDXGIFactory1``, ``EnumAdapters1``, ``GetDesc1``): name, PCI vendor, dedicated
  video memory, and with ``IDXGIAdapter3::QueryVideoMemoryInfo`` the local memory budget the system gives this
  process and what this process uses of it. The adapter's position in that list is DirectML's ``device_id``
  (``device = "dml:N"``), as ONNX Runtime documents it.
* NVIDIA GPUs' LUIDs: the CUDA driver (``nvcuda.dll``: ``cuDeviceGetLuid``, ``cuDeviceGetPCIBusId``), so that
  bolo.accel can find the DXGI adapter an NVML device is and read this process's use of it (NVML has no LUID).

Every function returns None when it cannot ask (another system, an old Windows, an error) and never raises.
"""

from __future__ import annotations

import contextlib
import re
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
IID_IDXGIFactory6 = "c1b6694f-ff09-44a9-b03c-77900a0a1d17"
IID_IDXGIAdapter1 = "29038f61-3839-4626-91fd-086879011a05"
DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE = 2
DXGI_ADAPTER_FLAG_SOFTWARE = 2
DXGI_ERROR_NOT_FOUND = 0x887A0002
PCI_VENDORS = {0x10DE: "nvidia", 0x1002: "amd", 0x1022: "amd", 0x8086: "intel", 0x1414: "microsoft", 0x5143: "qualcomm"}
#: dedicated video memory at or below which an adapter is taken for integrated graphics (a BIOS carve-out)
INTEGRATED_MAX_BYTES = 512 * 1024 * 1024
#: AMD APU graphics: "AMD Radeon(TM) Graphics", "AMD Radeon 780M Graphics", "AMD Radeon(TM) Vega 8 Graphics"
_AMD_APU = re.compile(r"^amd radeon(\(tm\))?( vega)?( \d{2,3}m| \d{1,2})? graphics$", re.IGNORECASE)
#: Intel's discrete cards: "Intel(R) Arc(TM) A770 Graphics", "Intel(R) Arc(TM) B580 Graphics"; every other Intel
#: adapter ("UHD", "Iris Xe", Meteor Lake's "Intel(R) Arc(TM) Graphics") is integrated
_INTEL_DISCRETE = re.compile(r"\barc\b.*\b[ab]\d{3}m?\b", re.IGNORECASE)


def is_integrated(vendor: str, name: str | None, dedicated_bytes: int | None) -> bool:
    """Integrated graphics: little dedicated memory, an Intel GPU other than a discrete Arc, or an AMD APU's
    Radeon. These never count as the machine's best GPU."""
    text = " ".join((name or "").split())
    if dedicated_bytes is not None and dedicated_bytes <= INTEGRATED_MAX_BYTES:
        return True
    if vendor == "intel":
        return not _INTEL_DISCRETE.search(text)
    return vendor == "amd" and bool(_AMD_APU.match(text))


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
    #: the adapter's locally unique id (DXGI_ADAPTER_DESC1.AdapterLuid), 0 when unknown
    luid: int = 0
    #: the adapter Windows names first for DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE (IDXGIFactory6)
    high_performance: bool = False

    @property
    def vendor(self) -> str:
        return PCI_VENDORS.get(self.vendor_id, "unknown")

    @property
    def integrated(self) -> bool:
        return is_integrated(self.vendor, self.name, self.dedicated_bytes)


def preferred_index(adapters: list[Adapter] | None) -> int | None:
    """DirectML's ``device_id`` for ``device = "dml"`` with no number: the adapter Windows calls high-performance,
    else the hardware adapter with the most dedicated memory (a discrete card before integrated graphics), else
    None (no adapter)."""
    hardware = [a for a in adapters or [] if not a.software]
    marked = next((a for a in hardware if a.high_performance), None)
    if marked is not None:
        return marked.index
    if not hardware:
        return None
    return max(hardware, key=lambda a: (not a.integrated, a.dedicated_bytes, -a.index)).index


_default_index: tuple[int | None] | None = None


def default_dml_index() -> int | None:
    """``preferred_index`` of this machine's adapters, asked once per process; None off Windows."""
    global _default_index
    if _default_index is None:
        _default_index = (preferred_index(dxgi_adapters()),)
    return _default_index[0]


#: AMD's switchable-graphics Vulkan implicit layer (amdvlk / the Radeon driver on hybrid AMD + NVIDIA laptops): with
#: it loaded, a Vulkan llama-server hung in device enumeration (``--list-devices`` never answered and Windows logged
#: RADAR_PRE_LEAK_64). Setting this variable to 1 keeps the Vulkan loader from loading that layer.
AMD_SWITCHABLE_LAYER_VAR = "DISABLE_LAYER_AMD_SWITCHABLE_GRAPHICS_1"


def _windows() -> bool:
    return sys.platform == "win32"


def has_amd_adapter(adapters: list[Adapter] | None) -> bool:
    return any(a.vendor == "amd" and not a.software for a in adapters or [])


def vulkan_child_env(
    flavour: str | None,
    *,
    windows: bool | None = None,
    adapters: Any = None,
    environ: dict[str, str] | None = None,
) -> dict[str, str]:
    """What a Vulkan build of llama-server or audio.cpp that bolo starts needs in its environment on this machine:
    on Windows with an AMD graphics adapter (DXGI's list; ``adapters`` may be the list or a function returning
    it), ``DISABLE_LAYER_AMD_SWITCHABLE_GRAPHICS_1=1``, unless the operator set that variable (to anything) in
    bolo's own environment. Empty for other builds and elsewhere."""
    import os

    if flavour != "vulkan":
        return {}
    if not (_windows() if windows is None else windows):
        return {}
    env = os.environ if environ is None else environ
    if any(k.upper() == AMD_SWITCHABLE_LAYER_VAR for k in env):
        return {}
    found = adapters() if callable(adapters) else adapters
    if adapters is None:
        found = dxgi_adapters()
    return {AMD_SWITCHABLE_LAYER_VAR: "1"} if has_amd_adapter(found) else {}


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
                    luid=(int(desc.AdapterLuidHigh) << 32) | (int(desc.AdapterLuidLow) & 0xFFFFFFFF),
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
        with contextlib.suppress(Exception):
            luid = _high_performance_luid(factory, AdapterDesc1)
            for a in adapters:
                a.high_performance = luid is not None and a.luid == luid and not a.software
    finally:
        _release(factory)
    return adapters


def _high_performance_luid(factory: Any, desc_type: Any) -> int | None:
    """The LUID of the first adapter IDXGIFactory6::EnumAdapterByGpuPreference(HIGH_PERFORMANCE) gives, or None
    (no IDXGIFactory6 before Windows 10 1803, or only a software adapter)."""
    import ctypes
    from ctypes import wintypes

    hresult = ctypes.c_long
    factory6 = ctypes.c_void_p()
    query_interface = _method(factory, 0, hresult, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p))
    if query_interface(ctypes.byref(_guid(IID_IDXGIFactory6)), ctypes.byref(factory6)) < 0 or not factory6:
        return None
    try:
        # IDXGIFactory6's vtable: IUnknown (3), IDXGIObject (4), IDXGIFactory (5), IDXGIFactory1 (2),
        # IDXGIFactory2 (11), IDXGIFactory3 (1), IDXGIFactory4 (2), IDXGIFactory5 (1): EnumAdapterByGpuPreference is 29
        by_preference = _method(
            factory6, 29, hresult, wintypes.UINT, ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)
        )
        for position in range(16):
            adapter = ctypes.c_void_p()
            hr = by_preference(
                position,
                DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE,
                ctypes.byref(_guid(IID_IDXGIAdapter1)),
                ctypes.byref(adapter),
            )
            if hr < 0 or not adapter:
                return None
            try:
                desc = desc_type()
                if _method(adapter, 10, hresult, ctypes.POINTER(desc_type))(ctypes.byref(desc)) < 0:
                    continue
                if desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE:
                    continue
                return (int(desc.AdapterLuidHigh) << 32) | (int(desc.AdapterLuidLow) & 0xFFFFFFFF)
            finally:
                _release(adapter)
        return None
    finally:
        _release(factory6)


_cuda_luids: tuple[dict[tuple[int, int, int, int], int] | None] | None = None


def cuda_luids() -> dict[tuple[int, int, int, int], int] | None:
    """The LUID of each GPU the CUDA driver lists, by PCI address (domain, bus, device, function), asked once per
    process; None off Windows or when the driver cannot be asked. ``cuInit`` creates no context (no GPU memory)."""
    global _cuda_luids
    if _cuda_luids is None:
        found: dict[tuple[int, int, int, int], int] | None = None
        if sys.platform == "win32":
            with contextlib.suppress(Exception):
                found = _ask_cuda_luids()
        _cuda_luids = (found,)
    return _cuda_luids[0]


def _ask_cuda_luids() -> dict[tuple[int, int, int, int], int] | None:
    import ctypes

    from bolo.accel import bus_key

    cuda = ctypes.WinDLL("nvcuda")  # type: ignore[attr-defined,unused-ignore]
    if cuda.cuInit(0) != 0:
        return None
    count = ctypes.c_int()
    if cuda.cuDeviceGetCount(ctypes.byref(count)) != 0:
        return None
    out: dict[tuple[int, int, int, int], int] = {}
    for ordinal in range(count.value):
        device = ctypes.c_int()
        if cuda.cuDeviceGet(ctypes.byref(device), ordinal) != 0:
            continue
        luid = (ctypes.c_ubyte * 8)()
        mask = ctypes.c_uint()
        bus = ctypes.create_string_buffer(64)
        if cuda.cuDeviceGetLuid(luid, ctypes.byref(mask), device) != 0:
            continue
        if cuda.cuDeviceGetPCIBusId(bus, 64, device) != 0:
            continue
        key = bus_key(bus.value.decode(errors="replace"))
        if key is None:
            continue
        raw = bytes(luid)
        low = int.from_bytes(raw[:4], "little")
        high = int.from_bytes(raw[4:], "little", signed=True)  # LUID.HighPart is a LONG
        out[key] = (high << 32) | low
    return out
