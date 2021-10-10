from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from functools import wraps
from itertools import permutations
from typing import Literal, Optional

from brainfuck import Instruction, Print, Read, Add, Move, BuiltProgram, Forward, Backward


def log_wrap(f_or_none=None, /, message_pre=None, message_post=None):
    def wrap(func):
        nonlocal message_pre, message_post
        if message_pre is None:
            message_pre = f"Function {func.__name__} called with {{args}} and {{kwargs}}"
        if message_post is None:
            message_post = f"Function {func.__name__} returned {{result}}"

        @wraps(func)
        def wrapper(*args, **kwargs):
            print(message_pre.format(args=args, kwargs=kwargs))
            res = func(*args, **kwargs)
            print(message_post.format(result=res))
            return res

        return wrapper

    if f_or_none is None:
        return wrap
    else:
        return wrap(f_or_none)


class Predicate(ABC):
    def implies(self, other: Predicate) -> Predicate:
        return implies(self, other)

    def incompatible(self, other: Predicate) -> Predicate:
        return incompatible(self, other)

    def optimize(self) -> Predicate:
        return optimize(self)


@dataclass(frozen=True)
class TruePredicate(Predicate):
    pass


true = TruePredicate()


@dataclass(frozen=True)
class FalsePredicate(Predicate):
    pass


false = FalsePredicate()


@dataclass(frozen=True)
class ZerosOutside(Predicate):
    start: int
    end: int


@dataclass(frozen=True)
class Equals(Predicate):
    offset: int
    value: int


@dataclass(frozen=True)
class GreaterThan(Predicate):
    offset: int
    value: int


@dataclass(frozen=True)
class SmallerThan(Predicate):
    offset: int
    value: int


def not_equal(offset: int, value: int) -> Predicate:
    return Any(frozenset({GreaterThan(offset, value), SmallerThan(offset, value)}))


@dataclass(frozen=True)
class All(Predicate):
    predicates: frozenset[Predicate, ...]
    
    def __repr__(self):
        return f"All({' & '.join(map(repr, self.predicates))})"

    def flatten(self) -> All:
        out = []
        for c in self.predicates:
            if isinstance(c, All):
                out.extend(c.predicates)
            else:
                out.append(c)

        out = frozenset(out)
        if not out:
            return true
        elif len(out) == 1:
            return next(iter(out))
        else:
            return All(frozenset(out))

    def optimize_all(self) -> All:
        out = list(self.flatten().predicates)
        for i in range(0, len(out)):
            for j in range(0, len(out)):
                if i == j:
                    continueh
                if implies(out[i], out[j]):
                    out[j] = true
                elif incompatible(out[i], out[j]):
                    return All(frozenset({false}))
        return All(frozenset(c for c in out if not isinstance(c, TruePredicate)))

    def expand_any_clause(self) -> Predicate:
        children = list(self.predicates)
        for i, c in enumerate(children):
            if isinstance(c, Any):
                others = children[:i] + children[i + 1:]
                out = []
                for a in c.predicates:
                    out.append(All(frozenset({*others, a})))
                return Any(frozenset(out))
        return self


@dataclass(frozen=True)
class Any(Predicate):
    predicates: frozenset[Predicate, ...]
    def __repr__(self):
        return f"Any({' | '.join(map(repr, self.predicates))})"

    def flatten(self) -> Any:
        out = []
        for c in self.predicates:
            if isinstance(c, Any):
                out.extend(c.predicates)
            else:
                out.append(c)

        if not out:
            return false
        elif len(out) == 1:
            return next(iter(out))
        else:
            return Any(frozenset(out))

    def optimize_any(self) -> Any:
        out = list(self.flatten().predicates)
        for i in range(0, len(out)):
            for j in range(0, len(out)):
                if i != j and implies(out[j], out[i]):
                    out[j] = false
        return Any(frozenset(c for c in out if not isinstance(c, FalsePredicate)))


# @log_wrap
def implies(a: Predicate, b: Predicate) -> bool:
    match a, b:
        case _, TruePredicate():
            return True
        case FalsePredicate(), _:
            return True
        case s, All(children):
            return all(implies(s, c) for c in children)
        case s, Any(children):
            return any(implies(s, c) for c in children)
        case All(children), o:
            return any(implies(c, o) for c in children)
        case Any(children), o:
            return all(implies(c, o) for c in children)
        case ZerosOutside(sx, ex), ZerosOutside(sy, ey):
            return sx <= sy and ex >= ey
        case ZerosOutside(start, end), Equals(oy, v):
            return v == 0 and not (start < oy < end)
        case ZerosOutside(start, end), GreaterThan(oy, v):
            return v < 0 and not (start < oy < end)
        case ZerosOutside(start, end), SmallerThan(oy, v):
            return v > 0 and not (start < oy < end)
        case Equals(ox, x), Equals(oy, y):
            return ox == oy and x == y
        case Equals(ox, x), GreaterThan(oy, y):
            return ox == oy and x > y
        case Equals(ox, x), SmallerThan(oy, y):
            return ox == oy and x < y
        case GreaterThan(ox, x), GreaterThan(oy, y):
            return ox == oy and x >= y
        case SmallerThan(), GreaterThan():
            return False
        case GreaterThan(), SmallerThan():
            return False
        case _, ZerosOutside(_, _):
            return False
        case _, Equals(_):
            return False
        case _:
            print("Couldn't find rule for", a, b)
            return False


def optimize(pred: Predicate) -> Predicate:
    if isinstance(pred, (All, Any)):
        pred = pred.flatten()
    match pred:
        case All(children):
            children = [optimize(c) for c in children]
            temp_self = All(frozenset(children))
            for c in children:
                match c:
                    case FalsePredicate():
                        return false
                    case Equals(o, v):
                        if implies(temp_self, not_equal(o, v)):
                            return false
                    case GreaterThan(o, v):
                        if implies(temp_self, SmallerThan(o, v + 1)):
                            return false
                    case SmallerThan(o, v):
                        if implies(temp_self, GreaterThan(o, v - 1)):
                            return false
            p = temp_self.expand_any_clause()
            if isinstance(p, Any):
                return optimize(p.flatten())
            return p
        case Any(children):
            return Any(frozenset(optimize(c) for c in children)).flatten()
        case _:
            return pred



def incompatible(a: Predicate, b: Predicate) -> bool:
    match a, b:
        case FalsePredicate(), _:
            return True
        case _, FalsePredicate():
            return True
        case s, Any(children):
            return all(incompatible(s, c) for c in children)
        case Any(children), s:
            return all(incompatible(c, s) for c in children)
        case s, All(children):
            return any(incompatible(s, c) for c in children)
        case All(children), s:
            return any(incompatible(c, s) for c in children)
        case ZerosOutside(start, end), Equals(oy, y):
            return y != 0 and not (start < oy < end)
        # case GreaterThan(offset)
        case Equals(ox, x), Equals(oy, y):
            return ox == oy and y != x
        case _:
            return False


def apply(pred: Predicate, ins: Instruction) -> Predicate:
    match ins, pred:
        case Print(), _:
            return pred
        case Add(0), _:
            return pred
        case Move(0), _:
            return pred
        case i, All(children):
            return All(frozenset(apply(c, i) for c in children)).optimize()
        case i, Any(children):
            return Any(frozenset(apply(c, i) for c in children)).optimize()
        case Add(v), ZerosOutside(start, end) if start < 0 < end:
            return ZerosOutside(start, end)
        case Add(v), ZerosOutside(start, end) if end <= 0:
            predicates = [Equals(i, 0) for i in range(end, 0)]
            return All(frozenset({Equals(0, v), *predicates, ZerosOutside(start or -1, 1)})).optimize()
        case Add(v), ZerosOutside(start, end) if start >= 0:
            predicates = [Equals(i, 0) for i in range(1, start + 1)]
            return All(frozenset({Equals(0, v), *predicates, ZerosOutside(-1, end)})).optimize()
        case Add(v), Equals(0, old):
            return Equals(0, old + v)
        case Add(_), Equals(o, v):
            return Equals(o, v)
        case Add(v), GreaterThan(0, old):
            return GreaterThan(0, old + v)
        case Add(_), GreaterThan(o, v):
            return GreaterThan(o, v)
        case Add(v), SmallerThan(0, old):
            return SmallerThan(0, old + v)
        case Add(_), SmallerThan(o, v):
            return SmallerThan(o, v)
        case Move(d), ZerosOutside(start, end):
            return ZerosOutside(start + d, end + d)
        case Move(d), Equals(offset, v):
            return Equals(offset + d, v)
        case _:
            raise ValueError(f"Unhandled {ins, pred}")


not_zero = Any(frozenset({GreaterThan(0, 0), SmallerThan(0, 0)}))
zero = Equals(0, 0)


def step(predicate: Predicate, program: BuiltProgram, ip: int) -> tuple[tuple[int, Predicate], ...]:
    match program[ip]:
        case Forward(target):
            if implies(predicate, not_zero):
                return (ip + 1, predicate),
            elif implies(predicate, zero):
                return (target, predicate),
            else:
                return (
                           ip + 1, All(frozenset({predicate, not_zero})).optimize()
                       ), (
                           target + 1, All(frozenset({predicate, zero})).optimize()
                       )
        case Backward(target):
            if implies(predicate, zero):
                return (ip + 1, predicate),
            elif implies(predicate, not_zero):
                return (target, predicate),
            else:
                return (
                           ip + 1, All(frozenset({predicate, zero})).optimize()
                       ), (
                           target, All(frozenset({predicate, not_zero})).optimize()
                       )
        case ins:
            return (ip + 1, apply(predicate, ins)),


def weaken(pred: Predicate, max_offset: int) -> Predicate:
    match pred:
        case Equals(offset, _) if abs(offset) > max_offset:
            return true
        case GreaterThan(offset, _) if abs(offset) > max_offset:
            return true
        case SmallerThan(offset, _) if abs(offset) > max_offset:
            return true
        case All(children):
            return All(frozenset(weaken(c, max_offset) for c in children)).optimize()
        case Any(children):
            return Any(frozenset(weaken(c, max_offset) for c in children)).optimize()
        
        case Equals(offset, n) if n > max_offset:
            return GreaterThan(offset, n - 1)
        case GreaterThan(offset, n) if n > max_offset:
            return GreaterThan(offset, n - 1)
        case Equals(offset, n) if n < -max_offset:
            return SmallerThan(offset, n - 1)
        case SmallerThan(offset, n) if n < -max_offset:
            return SmallerThan(offset, n - 1)
        case _:
            return pred


def intersect(a: Predicate, b: Predicate, max_offset: int) -> Predicate:
    if implies(a, b):
        return b
    elif implies(b, a):
        return a
    else:
        return weaken(Any(frozenset({a, b})), max_offset)


def runs_forever(program: BuiltProgram, max_offset: int = 10) -> bool:
    invariants = {0: ZerosOutside(0, 1)}
    points = {0}
    while points:
        new = []
        for ip in points:
            if ip >= len(program):
                return False  # Reached end of program
            pred = invariants[ip]
            new.extend(step(pred, program, ip))
        new_points = set()
        for ip, i2 in new:
            if ip in invariants:
                i1 = invariants[ip]
                if implies(i2, i1):
                    continue  # This invariant is correct
                invariants[ip] = intersect(i1, i2, max_offset)
            else:
                invariants[ip] = i2
            new_points.add(ip)
        points = new_points
        print(invariants)
    return True  # Found loop
