from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, replace
from functools import cached_property
from typing import Callable

from frozendict import frozendict


class MismatchedParentheses(ValueError):
    pass


class LongProgram(ValueError):
    def __init__(self, program: BrainfuckProgram, max_n: int):
        self.program = program
        self.max_n = max_n
        super(LongProgram, self).__init__(
            f"The Program {program.code!r} took more than {max_n} iterations without ending or a loop being recognized.")


@dataclass(frozen=True)
class BrainfuckProgram:
    code: str

    @cached_property
    def is_valid(self) -> bool:
        try:
            self.pairs
            return True
        except ValueError:
            return False

    @cached_property
    def pairs(self) -> frozendict[int, int]:
        stack = []
        pairs = {}
        for i, c in enumerate(self.code):
            if c == "[":
                stack.append(i)
            elif c == "]":
                try:
                    j = stack.pop()
                except IndexError:
                    raise MismatchedParentheses("Unmatched opening")
                pairs[i] = j
                pairs[j] = i
        if stack:
            raise MismatchedParentheses("Unmatched opening")
        return frozendict(pairs)

    def halts(self, max_n: int = None) -> bool:
        max_n = 2 ** len(self.code)
        state = BrainfuckState(self)
        states = {state}
        for _ in range(max_n):
            state = state.next()
            if state.done:
                return True
            elif state in states:
                return False  # Loop detected
            else:
                states.add(state)
        raise LongProgram(self, max_n)


@dataclass(frozen=True, slots=True)
class BrainfuckState:
    program: BrainfuckProgram
    band: frozendict[int, int] = frozendict()
    pointer: int = 0
    pc: int = 0

    def next(self, write: Callable[[int], None] = lambda _: None, read: Callable[[], int] = lambda: 0) -> BrainfuckState:
        new_band = self.band
        dict(self.band)
        new_pointer = self.pointer
        new_pc = self.pc + 1
        match self.program.code[self.pc]:
            case "+":
                new_band = dict(new_band)
                new_band[new_pointer] = new_band.get(new_pointer, 0) + 1
            case "-":
                new_band = dict(new_band)
                new_band[new_pointer] = new_band.get(new_pointer, 0) - 1
            case ">":
                new_pointer += 1
            case "<":
                new_pointer -= 1
            case ".":
                write(new_band.get(new_pointer, 0))
            case ",":
                new_band = dict(new_band)
                new_band[new_pointer] = read()
            case "[":
                if new_band.get(new_pointer, 0) == 0:
                    new_pc = self.program.pairs[self.pc] + 1
            case "]":
                if new_band.get(new_pointer, 0):
                    new_pc = self.program.pairs[self.pc] + 1
        return BrainfuckState(self.program, frozendict(new_band), new_pointer, new_pc)

    @property
    def done(self):
        return self.pc >= len(self.program.code)


class Instruction(ABC):
    pass


@dataclass(frozen=True)
class Add(Instruction):
    value: int


@dataclass(frozen=True)
class Move(Instruction):
    offset: int


@dataclass(frozen=True)
class Print(Instruction):
    pass


@dataclass(frozen=True)
class Read(Instruction):
    pass


@dataclass(frozen=True)
class Forward(Instruction):
    offset: int


@dataclass(frozen=True)
class Backward(Instruction):
    offset: int


countable = {
    ">": (1, Move),
    "<": (-1, Move),
    "+": (1, Add),
    "-": (-1, Add),
}


class BuiltProgram(tuple[Instruction, ...]):

    @classmethod
    def from_program(cls, program: BrainfuckProgram):
        count = None
        ins = None
        out = []
        for i, c in enumerate(program.code):
            if c in countable:
                f, c = countable[c]
                if ins is c:
                    count += f
                    continue
                if ins is not None and count != 0:
                    out.append(ins(count))
                ins = c
                count = f
                continue
            if ins is not None and count != 0:
                out.append(ins(count))
                count = 0
                ins = None
            match c:
                case ".":
                    out.append(Print())
                case ",":
                    out.append(Read())
                case "[":
                    out.append(Forward(program.pairs[i]))
                case "]":
                    out.append(Backward(program.pairs[i]))
        return cls(out)
