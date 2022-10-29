import copy
import ctypes
import random
from ctypes import wintypes
import time
import logging

# from torch.multiprocessing import Process, Manager, Event
from multiprocessing import Process, Manager, Event

import config

logging.basicConfig(filename='logs/train.log', level=logging.DEBUG, format='%(asctime)s %(levelname)-8s %(message)s')
logger = logging.getLogger(__name__)

user32 = ctypes.WinDLL('user32', use_last_error=True)

INPUT_MOUSE = 0
INPUT_KEYBOARD = 1
INPUT_HARDWARE = 2

KEYEVENTF_EXTENDEDKEY = 0x0001
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_UNICODE = 0x0004
KEYEVENTF_SCANCODE = 0x0008

MAPVK_VK_TO_VSC = 0

# msdn.microsoft.com/en-us/library/dd375731

# C struct definitions

wintypes.ULONG_PTR = wintypes.WPARAM


class MOUSEINPUT(ctypes.Structure):
    _fields_ = (("dx", wintypes.LONG),
                ("dy", wintypes.LONG),
                ("mouseData", wintypes.DWORD),
                ("dwFlags", wintypes.DWORD),
                ("time", wintypes.DWORD),
                ("dwExtraInfo", wintypes.ULONG_PTR))


class KEYBDINPUT(ctypes.Structure):
    _fields_ = (("wVk", wintypes.WORD),
                ("wScan", wintypes.WORD),
                ("dwFlags", wintypes.DWORD),
                ("time", wintypes.DWORD),
                ("dwExtraInfo", wintypes.ULONG_PTR))

    def __init__(self, *args, **kwds):
        super(KEYBDINPUT, self).__init__(*args, **kwds)
        # some programs use the scan code even if KEYEVENTF_SCANCODE
        # isn't set in dwFflags, so attempt to map the correct code.
        if not self.dwFlags & KEYEVENTF_UNICODE:
            self.wScan = user32.MapVirtualKeyExW(self.wVk,
                                                 MAPVK_VK_TO_VSC, 0)


class HARDWAREINPUT(ctypes.Structure):
    _fields_ = (("uMsg", wintypes.DWORD),
                ("wParamL", wintypes.WORD),
                ("wParamH", wintypes.WORD))


class INPUT(ctypes.Structure):
    class _INPUT(ctypes.Union):
        _fields_ = (("ki", KEYBDINPUT),
                    ("mi", MOUSEINPUT),
                    ("hi", HARDWAREINPUT))

    _anonymous_ = ("_input",)
    _fields_ = (("type", wintypes.DWORD),
                ("_input", _INPUT))


LPINPUT = ctypes.POINTER(INPUT)


def _check_count(result, func, args):
    if result == 0:
        raise ctypes.WinError(ctypes.get_last_error())
    return args


user32.SendInput.errcheck = _check_count
user32.SendInput.argtypes = (wintypes.UINT,  # nInputs
                             LPINPUT,  # pInputs
                             ctypes.c_int)  # cbSize


# Functions

def PressKey(hexKeyCode):
    x = INPUT(type=INPUT_KEYBOARD,
              ki=KEYBDINPUT(wVk=hexKeyCode))
    user32.SendInput(1, ctypes.byref(x), ctypes.sizeof(x))


def ReleaseKey(hexKeyCode):
    x = INPUT(type=INPUT_KEYBOARD,
              ki=KEYBDINPUT(wVk=hexKeyCode,
                            dwFlags=KEYEVENTF_KEYUP))
    user32.SendInput(1, ctypes.byref(x), ctypes.sizeof(x))


# _P1_1 = [0x41, 0x53]  # as
# _P1_2 = [0x53]  # s
# _P1_3 = [0x53, 0x44]  # sd
# _P1_4 = [0x41]  # a
# _P1_6 = [0x44]  # d
# _P1_7 = [0x57, 0x41]  # wa
# _P1_8 = [0x57]  # w
# _P1_9 = [0x57, 0x44]  # wd
# _P1_A = [0x49]  # i
# _P1_B = [0x55]  # u
# _P1_C = [0x4F]  # o
# _P1_D = [0x4A]  # j
#
# _P1_directions = [_P1_1, _P1_2, _P1_3, _P1_4, _P1_6, _P1_7, _P1_8, _P1_9, []]
# _P1_buttons = [_P1_A, _P1_B, _P1_C, _P1_D]

mapping_dicts = dict()

mapping_dicts[0] = dict()
mapping_dicts[0]['2'] = 0x53  # P1 2 # s
mapping_dicts[0]['4'] = 0x41  # P1 4 # a
mapping_dicts[0]['8'] = 0x44  # P1 8 # d
mapping_dicts[0]['6'] = 0x57  # P1 6 # w
# buttons
mapping_dicts[0]['a'] = 0x49  # P1 A # i
mapping_dicts[0]['b'] = 0x55  # P1 B # u
mapping_dicts[0]['c'] = 0x4F  # P1 C # o
mapping_dicts[0]['d'] = 0x4A  # P1 D # j

# P2
mapping_dicts[1] = dict()
# directions
mapping_dicts[1]['2'] = 0x58  # P2 2 # x #
mapping_dicts[1]['4'] = 0x56  # P2 4 # c
mapping_dicts[1]['8'] = 0x5A  # P2 8 # v
mapping_dicts[1]['6'] = 0x43  # P2 6 # z
# buttons
mapping_dicts[1]['a'] = 0x42  # P2 A # b
mapping_dicts[1]['b'] = 0x4E  # P2 B # n
mapping_dicts[1]['c'] = 0x4D  # P2 C # m
mapping_dicts[1]['d'] = 0xBC  # P2 D # ,

input_pressed = dict()
for k_ in config.settings['valid_inputs']:
    input_pressed[k_] = False


def create_input_dict(input_dict):
    for k in config.settings['valid_inputs']:
        input_dict[k] = 0

    return input_dict


def create_p2_input_dict():
    input_dict = Manager().dict()

    for k in config.settings['valid_inputs']:
        input_dict[k] = 0

    return input_dict


def do_inputs(input_dict, mapping_dict, die, env_status):
    while not die.is_set():
        if not env_status['round_done']:
            time.sleep(.012)
            for k in input_dict.keys():
                if input_dict[k]:
                    PressKey(mapping_dict[k])
                    input_pressed[k] = True
                elif input_pressed[k]:
                    ReleaseKey(mapping_dict[k])
                    input_pressed[k] = False
        else:
            time.sleep(.001)
    print("do_inputs die..")
    for k in input_pressed:
        ReleaseKey(mapping_dict[k])
    print("do_inputs dead..")


def random_inputs(input_dict, die):
    while not die.is_set():
        input_dict = randomize_inputs(input_dict)
        # time.sleep(.013)
        time.sleep(.013)
    print("random_inputs dead..")


def randomize_inputs(input_dict):
    for k in input_dict.keys():
        if random.randint(0, 1) == 1:
            input_dict[k] = not input_dict[k]
    return input_dict


def reset_round():
    time.sleep(.001)
    # make sure every key is released
    for _, mapping_dict in mapping_dicts.items():
        for k in mapping_dict.values():
            time.sleep(.001)
            ReleaseKey(k)

    PressKey(0x53)  # s # down
    time.sleep(.001)
    PressKey(0x52)  # r

    time.sleep(.005)
    ReleaseKey(0x52)  # r
    time.sleep(.001)
    ReleaseKey(0x53)  # s # down
    # time.sleep(.001)


def test_inputs():
    for r in range(0, 2):
        d = mapping_dicts[r]

        time.sleep(5)
        for _, k in d.items():
            print(_)
            time.sleep(2)
            PressKey(k)  # s # down
            time.sleep(.1)
            ReleaseKey(k)  # r
            print("----")


if __name__ == "__main__":
    test_inputs()
