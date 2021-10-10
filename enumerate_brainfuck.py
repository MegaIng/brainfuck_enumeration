from __future__ import annotations

import json
from itertools import count, combinations_with_replacement, islice, product
from typing import Iterable

from brainfuck import BrainfuckProgram, LongProgram, BuiltProgram
from proofer import runs_forever

known_programs = json.load(open("known_programs.json", encoding="utf-8"))

symbols = "><+-.,[]"


def generate_basic() -> Iterable[BrainfuckProgram]:
    for n in count(1):
        yield from map(BrainfuckProgram, map(''.join, product(symbols, repeat=n)))


def filter_valid(stream: Iterable[BrainfuckProgram]) -> Iterable[BrainfuckProgram]:
    for p in stream:
        if p.is_valid:
            yield p


def filter_halts(stream: Iterable[BrainfuckProgram]) -> Iterable[BrainfuckProgram]:
    for p in stream:
        try:
            if p.halts():
                yield p
            else:
                print(p, "doesn't halt")
        except LongProgram:
            print(p, "starting proof")
            if not runs_forever(BuiltProgram.from_program(p)):
                raise ValueError("Couldn't proof that program {p} runs forever")

for i, p in enumerate(filter_halts(filter_valid(generate_basic()))):
    print(i+1, p.code)
