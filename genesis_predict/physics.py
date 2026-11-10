"""Physics engine (post-processing): fixed template solvers + unit checks.

The model does not write a solver (small models are poor at that). It fills a *spec*::

    시나리오: {"template": "projectile", "params": {"v0": "20 m/s", "angle": "30 deg", "g": "9.81 m/s^2"},
              "ask": "range", "unit": "m"}

The engine parses every quantity with units (``pint``), checks each parameter has the right
dimension, solves, and compares with the model's claimed answer. Outcomes:
``match`` / ``mismatch`` / ``unsupported`` (template or ask not known) / ``bad_spec``.
Templates are solved in closed form; ``projectile`` and ``cooling`` are cross-checked by an ODE
integration (scipy) so a wrong closed form is caught by tests, not by the eval set.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Callable

_UREG = None


def ureg():
    global _UREG
    if _UREG is None:
        import pint  # lazy

        _UREG = pint.UnitRegistry()
        _UREG.define("won = [currency]")
    return _UREG


def Q(text):
    """Parse '20 m/s' (or a plain number, dimensionless) into a pint Quantity."""
    u = ureg()
    if isinstance(text, (int, float)):
        return u.Quantity(float(text), "")
    return u.Quantity(str(text).replace("^", "**"))


@dataclass
class Template:
    params: dict[str, str]                       # name -> dimension (pint dimensionality string)
    asks: dict[str, Callable[..., object]]       # ask -> function(**quantities) -> Quantity
    defaults: dict[str, str] | None = None


def _proj_flight_time(v0, angle, g, h0):
    vy = v0 * math.sin(angle.to("rad").magnitude)
    return (vy + (vy ** 2 + 2 * g * h0) ** 0.5) / g


TEMPLATES: dict[str, Template] = {
    "projectile": Template(
        {"v0": "[length]/[time]", "angle": "", "g": "[length]/[time]**2", "h0": "[length]"},
        {
            "flight_time": _proj_flight_time,
            "range": lambda v0, angle, g, h0: v0 * math.cos(angle.to("rad").magnitude) * _proj_flight_time(v0, angle, g, h0),
            "max_height": lambda v0, angle, g, h0: h0 + (v0 * math.sin(angle.to("rad").magnitude)) ** 2 / (2 * g),
        },
        {"g": "9.81 m/s**2", "h0": "0 m"},
    ),
    "free_fall": Template(
        {"h": "[length]", "g": "[length]/[time]**2"},
        {"time": lambda h, g: (2 * h / g) ** 0.5, "impact_speed": lambda h, g: (2 * g * h) ** 0.5},
        {"g": "9.81 m/s**2"},
    ),
    "pendulum": Template(
        {"L": "[length]", "g": "[length]/[time]**2"},
        {"period": lambda L, g: 2 * math.pi * (L / g) ** 0.5},
        {"g": "9.81 m/s**2"},
    ),
    "spring": Template(
        {"k": "[mass]/[time]**2", "m": "[mass]", "x": "[length]"},
        {"period": lambda k, m, x: 2 * math.pi * (m / k) ** 0.5, "energy": lambda k, m, x: 0.5 * k * x ** 2,
         "max_speed": lambda k, m, x: x * (k / m) ** 0.5},
        {"x": "0 m"},
    ),
    "incline": Template(
        {"angle": "", "mu": "", "g": "[length]/[time]**2", "m": "[mass]"},
        {"acceleration": lambda angle, mu, g, m: g * (math.sin(angle.to("rad").magnitude) - mu.magnitude * math.cos(angle.to("rad").magnitude)),
         "normal_force": lambda angle, mu, g, m: m * g * math.cos(angle.to("rad").magnitude)},
        {"g": "9.81 m/s**2", "mu": "0", "m": "1 kg"},
    ),
    "heat": Template(
        {"m": "[mass]", "c": "[length]**2/[time]**2/[temperature]", "dT": "[temperature]"},
        {"energy": lambda m, c, dT: m * c * dT},
    ),
    "cooling": Template(
        {"T0": "[temperature]", "Tenv": "[temperature]", "k": "1/[time]", "t": "[time]"},
        {"temperature": lambda T0, Tenv, k, t: Tenv + (T0 - Tenv) * math.exp(-(k * t).to("").magnitude)},
    ),
    "rc_charge": Template(
        {"V": "[mass]*[length]**2/[time]**3/[current]", "R": "[mass]*[length]**2/[time]**3/[current]**2",
         "C": "[current]**2*[time]**4/[mass]/[length]**2", "t": "[time]"},
        {"voltage": lambda V, R, C, t: V * (1 - math.exp(-(t / (R * C)).to("").magnitude)),
         "time_constant": lambda V, R, C, t: R * C},
    ),
    "ohm": Template(
        {"V": "[mass]*[length]**2/[time]**3/[current]", "R": "[mass]*[length]**2/[time]**3/[current]**2"},
        {"current": lambda V, R: V / R, "power": lambda V, R: V ** 2 / R},
    ),
    "ideal_gas": Template(
        {"n": "[substance]", "T": "[temperature]", "V": "[length]**3"},
        {"pressure": lambda n, T, V: n * ureg().Quantity(8.314462618, "J/(mol*K)") * T / V},
    ),
}

SPEC_RE = re.compile(r"시나리오\s*[:：]\s*(\{.*\})", re.S)


def extract_spec(reply: str) -> dict | None:
    """Pull the JSON spec after '시나리오:' from a model reply (first balanced object)."""
    m = SPEC_RE.search(reply or "")
    if not m:
        return None
    text, depth = m.group(1), 0
    for i, ch in enumerate(text):
        depth += ch == "{"
        depth -= ch == "}"
        if depth == 0:
            try:
                return json.loads(text[: i + 1])
            except json.JSONDecodeError:
                return None
    return None


def _to_kelvin_if_temperature(q):
    # offset units (degC) break multiplication; absolute temperatures go to K, differences stay deltas
    if q.dimensionality == ureg().Quantity(1, "K").dimensionality and str(q.units) in ("degree_Celsius", "degree_Fahrenheit"):
        return q.to("K")
    return q


def solve(spec: dict):
    """Return (status, quantity_or_message)."""
    tpl = TEMPLATES.get(spec.get("template"))
    if tpl is None:
        return "unsupported", f"unknown template {spec.get('template')!r}"
    fn = tpl.asks.get(spec.get("ask"))
    if fn is None:
        return "unsupported", f"unknown ask {spec.get('ask')!r} for {spec['template']}"
    raw = dict(tpl.defaults or {})
    raw.update(spec.get("params") or {})
    vals = {}
    for name, dim in tpl.params.items():
        if name not in raw:
            return "bad_spec", f"missing parameter {name}"
        try:
            q = _to_kelvin_if_temperature(Q(raw[name]))
        except Exception as exc:  # pint raises many types
            return "bad_spec", f"cannot parse {name}={raw[name]!r}: {exc}"
        want = ureg().Quantity(1, "").dimensionality if dim == "" else ureg().get_dimensionality(dim)
        if name == "angle" and q.dimensionless and str(q.units) == "dimensionless":
            q = ureg().Quantity(q.magnitude, "deg")  # bare angle numbers are degrees
        if q.dimensionality != want:
            return "bad_spec", f"{name} has dimension {q.dimensionality}, expected {want}"
        vals[name] = q
    try:
        out = fn(**vals)
    except (ValueError, ZeroDivisionError) as exc:
        return "bad_spec", str(exc)
    unit = spec.get("unit")
    if unit:
        try:
            out = out.to(unit)
        except Exception as exc:
            return "bad_spec", f"answer unit {unit!r} does not fit: {exc}"
    return "ok", out


def verify(spec: dict | None, claimed: float | None, rel_tol: float = 0.01, abs_tol: float = 1e-9) -> dict:
    """Recompute and compare. Never edits the answer; the caller decides what to do with a mismatch."""
    if spec is None:
        return {"status": "bad_spec", "detail": "no spec"}
    status, out = solve(spec)
    if status != "ok":
        return {"status": status, "detail": out}
    value = float(out.magnitude)
    result = {"status": "match", "engine_value": value, "unit": str(out.units)}
    if claimed is None:
        result["status"] = "no_claim"
    elif not math.isclose(value, float(claimed), rel_tol=rel_tol, abs_tol=abs_tol):
        result["status"] = "mismatch"
    return result
