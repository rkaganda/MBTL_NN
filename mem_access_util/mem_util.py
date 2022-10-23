from struct import unpack
import os
import time
import ctypes
import psutil


wintypes = ctypes.wintypes
windll = ctypes.windll
byref = ctypes.byref
sizeof = ctypes.sizeof
create_string_buffer = ctypes.create_string_buffer

WriteMem = windll.kernel32.WriteProcessMemory
ReadMem = windll.kernel32.ReadProcessMemory
OpenProcess = windll.kernel32.OpenProcess
Module32Next = windll.kernel32.Module32Next
Module32First = windll.kernel32.Module32First
CloseHandle = windll.kernel32.CloseHandle
CreateToolhelp32Snapshot = windll.kernel32.CreateToolhelp32Snapshot

g_pid = 0
g_pro_h = 0
g_base_ad = 0


class MODULEENTRY32(ctypes.Structure):
    _fields_ = [
        ("dwSize",             wintypes.DWORD),
        ("th32ModuleID",       wintypes.DWORD),
        ("th32ProcessID",      wintypes.DWORD),
        ("GlblcntUsage",       wintypes.DWORD),
        ("ProccntUsage",       wintypes.DWORD),
        ("modBaseAddr",        ctypes.POINTER(wintypes.BYTE)),
        ("modBaseSize",        wintypes.DWORD),
        ("hModule",            wintypes.HMODULE),
        ("szModule",           ctypes.c_byte * 256),
        ("szExePath",          ctypes.c_byte * 260),
    ]


class Mem_Data_Class:
    def __init__(self, byte_len, address):
        self.ad = address
        self.val = 0
        self.b_dat = create_string_buffer(byte_len)

    def r_mem(self):
        ReadMem(g_pro_h, self.ad + g_base_ad, self.b_dat, len(self.b_dat), None)
        self.val = b_unpack(self.b_dat)
        return b_unpack(self.b_dat)

    def w_mem(self):
        WriteMem(g_pro_h, self.ad + g_base_ad, self.b_dat, len(self.b_dat), None)


def r_mem_abs_addres(addres, b_dat):
    ReadMem(g_pro_h, addres, b_dat, len(b_dat), None)
    return b_unpack(b_dat)


def w_mem_abs_addres(addres, b_dat):
    WriteMem(g_pro_h, addres, b_dat, len(b_dat), None)


def b_unpack(d_data):

    num = len(d_data)
    if num == 1:
        return unpack('b', d_data)[0]
    elif num == 2:
        return unpack('h', d_data)[0]
    elif num == 4:
        return unpack('l', d_data)[0]


def get_connection(process_name):
    global g_pid
    global g_pro_h
    global g_base_ad

    res = False

    while res == False:
        res = pidget(process_name)
        if res == False:
            os.system('cls')
            print("Waiting for " + process_name + " to start")
            time.sleep(0.5)
    pid = res
    pro_h = OpenProcess(0x1F0FFF, False, pid)
    base_ad = get_base_addres(pid)

    g_pid = pid
    g_pro_h = pro_h
    g_base_ad = base_ad


def pidget(process_name):

    dict_pids = {
        p.info["name"]: p.info["pid"]
        for p in psutil.process_iter(attrs=["name", "pid"])
    }

    try:
        pid = dict_pids[process_name]
    except:
        pid = False

    return pid


def get_base_addres(pid):

    # MODULEENTRY32を取得
    snapshot = CreateToolhelp32Snapshot(0x00000008, pid)

    lpme = MODULEENTRY32()
    lpme.dwSize = sizeof(lpme)

    res = Module32First(snapshot, byref(lpme))

    while pid != lpme.th32ProcessID:
        res = Module32Next(snapshot, byref(lpme))

    b_baseAddr = create_string_buffer(8)
    b_baseAddr.raw = lpme.modBaseAddr

    base_ad = unpack('q', b_baseAddr.raw)[0]

    return base_ad


def changeFontSize(size_x, size_y):  # Changes the font size to *size* pixels (kind of, but not really. You'll have to try it to chack if it works for your purpose ;) )
    from ctypes import POINTER, WinDLL, Structure, sizeof, byref
    from ctypes.wintypes import BOOL, SHORT, WCHAR, UINT, ULONG, DWORD, HANDLE

    LF_FACESIZE = 32
    STD_OUTPUT_HANDLE = -11

    class COORD(Structure):
        _fields_ = [
            ("X", SHORT),
            ("Y", SHORT),
        ]

    class CONSOLE_FONT_INFOEX(Structure):
        _fields_ = [
            ("cbSize", ULONG),
            ("nFont", DWORD),
            ("dwFontSize", COORD),
            ("FontFamily", UINT),
            ("FontWeight", UINT),
            ("FaceName", WCHAR * LF_FACESIZE)
        ]

    kernel32_dll = WinDLL("kernel32.dll")

    get_last_error_func = kernel32_dll.GetLastError
    get_last_error_func.argtypes = []
    get_last_error_func.restype = DWORD

    get_std_handle_func = kernel32_dll.GetStdHandle
    get_std_handle_func.argtypes = [DWORD]
    get_std_handle_func.restype = HANDLE

    get_current_console_font_ex_func = kernel32_dll.GetCurrentConsoleFontEx
    get_current_console_font_ex_func.argtypes = [HANDLE, BOOL, POINTER(CONSOLE_FONT_INFOEX)]
    get_current_console_font_ex_func.restype = BOOL

    set_current_console_font_ex_func = kernel32_dll.SetCurrentConsoleFontEx
    set_current_console_font_ex_func.argtypes = [HANDLE, BOOL, POINTER(CONSOLE_FONT_INFOEX)]
    set_current_console_font_ex_func.restype = BOOL

    stdout = get_std_handle_func(STD_OUTPUT_HANDLE)
    font = CONSOLE_FONT_INFOEX()
    font.cbSize = sizeof(CONSOLE_FONT_INFOEX)

    font.dwFontSize.X = size_x
    font.dwFontSize.Y = size_y

    set_current_console_font_ex_func(stdout, False, byref(font))