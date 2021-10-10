from __future__ import annotations

from brainfuck import BrainfuckProgram, BuiltProgram
from proofer import runs_forever

bf = BrainfuckProgram("+[>+]")
p = BuiltProgram.from_program(bf)

print(runs_forever(p))