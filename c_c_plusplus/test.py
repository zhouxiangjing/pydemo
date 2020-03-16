import ctypes

lib = ctypes.cdll.LoadLibrary("lib/libpycall.so")

print(lib.foo(1, 3))

print('***finish***')

ctypes.