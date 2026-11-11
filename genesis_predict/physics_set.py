"""Physics dev set: Korean word problems, independent gold answers, planted errors, scorer.

Gold answers do NOT come from ``genesis_predict.physics`` (the engine under test). They come from
plain-float SI code written separately here, and wherever there is a motion law, from numerical
integration (``scipy.integrate.solve_ivp``). Units in the questions are mixed on purpose
(cm, km/h, g, kΩ, μF, ms, L, °C, min) and the gold path converts them with its own table, not pint.

Each in-scope item carries:
* ``spec``   - the reference scenario spec a perfect model would write,
* ``gold``   - the answer in ``spec['unit']``,
* ``claims`` - one correct claim and one wrong claim (a planted, typical slip).
Out-of-scope items (drag, relativity, orbits, ...) have ``status: unsupported`` and no gold.

    python -m genesis_predict.physics_set make --seed 0 --per-template 22 --unsupported 20 --out phys_dev.jsonl
    python -m genesis_predict.physics_set p0 --set phys_dev.jsonl      # engine only, gold spec
"""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict

SET_KIND = "genesis_predict.physics_set/v1"
G_EARTH = 9.81
PLANETS = [("지구", 9.81), ("달", 1.62), ("화성", 3.71)]

# ---- unit table for the gold path (independent of pint) ----
TO_SI = {"m": 1.0, "cm": 0.01, "km": 1000.0, "m/s": 1.0, "km/h": 1 / 3.6, "s": 1.0, "ms": 1e-3, "min": 60.0,
         "kg": 1.0, "g": 1e-3, "N/m": 1.0, "J": 1.0, "kJ": 1e3, "W": 1.0, "V": 1.0, "A": 1.0, "mA": 1e-3,
         "ohm": 1.0, "kohm": 1e3, "F": 1.0, "uF": 1e-6, "Pa": 1.0, "kPa": 1e3, "L": 1e-3, "m^3": 1.0,
         "1/min": 1 / 60.0, "1/s": 1.0, "N": 1.0, "m/s^2": 1.0, "mol": 1.0, "J/(kg*K)": 1.0}
KO_UNIT = {"m": "m", "cm": "cm", "km/h": "km/h", "m/s": "m/s", "s": "초", "ms": "ms", "min": "분", "kg": "kg",
           "g": "g", "N/m": "N/m", "J": "J", "kJ": "kJ", "W": "W", "V": "V", "A": "A", "mA": "mA", "ohm": "Ω",
           "kohm": "kΩ", "uF": "μF", "Pa": "Pa", "kPa": "kPa", "L": "L", "m^3": "m³", "1/min": "1/분",
           "N": "N", "m/s^2": "m/s²", "mol": "mol", "J/(kg*K)": "J/(kg·K)"}
PINT = {"kohm": "kohm", "uF": "uF", "m^3": "m**3", "1/min": "1/min", "J/(kg*K)": "J/(kg*K)", "m/s^2": "m/s**2"}


def q(v: float, unit: str) -> str:
    """Spec quantity string (pint-readable)."""
    return f"{v:g} {PINT.get(unit, unit)}"


def si(v: float, unit: str) -> float:
    return v * TO_SI[unit]


def josa(word: str, pair: str = "은는") -> str:
    """Attach 은/는 (or 이/가) by whether the last Hangul syllable has a final consonant."""
    ch = word.rstrip(")")[-1] if not word.endswith(")") else word[: word.index("(")][-1]
    has = "가" <= ch <= "힣" and (ord(ch) - 0xAC00) % 28 != 0
    return word + (pair[0] if has else pair[1])


def _r(x: float, nd: int = 3) -> float:
    return float(f"{x:.{nd}g}")


# ---- gold solvers (scipy ODE where there is a motion law) ----

def _ode(fun, y0, t_end, event=None):
    from scipy.integrate import solve_ivp

    return solve_ivp(fun, (0, t_end), y0, events=event, rtol=1e-10, atol=1e-12, max_step=t_end / 2000)


def gold_projectile(v0, ang_deg, h0, g, ask):
    a = math.radians(ang_deg)

    def hit(t, s):
        return s[1]
    hit.terminal, hit.direction = True, -1

    def apex(t, s):  # vertical velocity crosses zero: exact top of the arc (sampling would miss it)
        return s[3]
    apex.direction = -1
    sol = _ode(lambda t, s: [s[2], s[3], 0.0, -g], [0.0, h0, v0 * math.cos(a), v0 * math.sin(a)], 1e4, [hit, apex])
    t_f, x_f = sol.t_events[0][0], sol.y_events[0][0][0]
    top = sol.y_events[1][0][1] if len(sol.t_events[1]) else h0
    return {"range": x_f, "flight_time": t_f, "max_height": float(top)}[ask]


def gold_free_fall(h, g, ask):
    def hit(t, s):
        return s[0]
    hit.terminal, hit.direction = True, -1
    sol = _ode(lambda t, s: [s[1], -g], [h, 0.0], 1e4, hit)
    return {"time": sol.t_events[0][0], "impact_speed": abs(sol.y_events[0][0][1])}[ask]


def gold_pendulum(L, g):
    th0 = math.radians(0.5)  # small amplitude; the period differs from 2*pi*sqrt(L/g) by ~5e-6

    def cross(t, s):
        return s[0]
    cross.direction = 1
    sol = _ode(lambda t, s: [s[1], -g / L * math.sin(s[0])], [th0, 0.0], 50 * math.sqrt(L / g), cross)
    ups = sol.t_events[0]
    return ups[1] - ups[0]


def gold_spring(k, m, x, ask):
    if ask == "period":
        return 2 * math.pi * math.sqrt(m / k)
    if ask == "energy":
        return 0.5 * k * x * x
    sol = _ode(lambda t, s: [s[1], -k / m * s[0]], [x, 0.0], 2 * math.pi * math.sqrt(m / k))
    return float(max(abs(sol.y[1])))


def gold_incline(ang_deg, mu, g, m, ask):
    a = math.radians(ang_deg)
    return {"acceleration": g * (math.sin(a) - mu * math.cos(a)), "normal_force": m * g * math.cos(a)}[ask]


def gold_cooling(T0, Tenv, k, t):
    sol = _ode(lambda s, T: [-k * (T[0] - Tenv)], [T0], t)
    return float(sol.y[0][-1])


def gold_rc(V, R, C, t, ask):
    if ask == "time_constant":
        return R * C
    sol = _ode(lambda s, v: [(V - v[0]) / (R * C)], [0.0], t)
    return float(sol.y[0][-1])


# ---- item builders: each returns (question, spec, gold_in_answer_unit) ----

def _projectile(rng):
    planet, g = rng.choice(PLANETS)
    vu = rng.choice(["m/s", "km/h"])
    v = rng.choice([8, 12, 15, 20, 25, 30]) if vu == "m/s" else rng.choice([36, 54, 72, 90])
    ang = rng.choice([15, 25, 30, 37, 45, 53, 60, 70])
    h0 = rng.choice([0, 0, 1.5, 2, 5, 10])
    ask, au = rng.choice([("range", "m"), ("max_height", "m"), ("flight_time", "s")])
    word = {"range": "수평 도달 거리", "max_height": "최고 높이(지면 기준)", "flight_time": "공중에 있는 시간"}[ask]
    where = "" if planet == "지구" else f"{planet}(중력가속도 {g} m/s²)에서 "
    start = "지면에서" if h0 == 0 else f"높이 {h0:g} m에서"
    qtext = (f"{where}공을 {start} 수평면과 {ang}° 각도로 {v:g} {KO_UNIT[vu]}의 속력으로 던졌다. "
             f"공기 저항은 무시한다. {josa(word)} 몇 {KO_UNIT[au]}인가?")
    params = {"v0": q(v, vu), "angle": f"{ang} deg", "h0": q(h0, "m")}
    if planet != "지구":
        params["g"] = q(g, "m/s^2")
    gold = gold_projectile(si(v, vu), ang, h0, g, ask)
    return qtext, {"template": "projectile", "params": params, "ask": ask, "unit": au}, gold


def _free_fall(rng):
    planet, g = rng.choice(PLANETS)
    hu = rng.choice(["m", "cm"])
    h = rng.choice([5, 10, 20, 45, 80]) if hu == "m" else rng.choice([50, 80, 120, 200])
    ask, au = rng.choice([("time", "s"), ("impact_speed", "m/s")])
    word = {"time": "바닥에 닿을 때까지 걸리는 시간", "impact_speed": "바닥에 닿는 순간의 속력"}[ask]
    where = "" if planet == "지구" else f"{planet}(중력가속도 {g} m/s²)에서 "
    qtext = f"{where}물체를 {h:g} {KO_UNIT[hu]} 높이에서 가만히 놓았다. 공기 저항은 무시한다. {josa(word)} 몇 {KO_UNIT[au]}인가?"
    params = {"h": q(h, hu)}
    if planet != "지구":
        params["g"] = q(g, "m/s^2")
    return qtext, {"template": "free_fall", "params": params, "ask": ask, "unit": au}, gold_free_fall(si(h, hu), g, ask)


def _pendulum(rng):
    planet, g = rng.choice(PLANETS)
    lu = rng.choice(["m", "cm"])
    L = rng.choice([0.5, 1, 1.5, 2, 3]) if lu == "m" else rng.choice([25, 40, 60, 80])
    where = "" if planet == "지구" else f"{planet}(중력가속도 {g} m/s²)에서 "
    qtext = f"{where}길이 {L:g} {KO_UNIT[lu]}인 단진자가 작은 진폭으로 흔들린다. 주기는 몇 초인가?"
    params = {"L": q(L, lu)}
    if planet != "지구":
        params["g"] = q(g, "m/s^2")
    return qtext, {"template": "pendulum", "params": params, "ask": "period", "unit": "s"}, gold_pendulum(si(L, lu), g)


def _spring(rng):
    k = rng.choice([20, 50, 80, 120, 200, 400])
    mu_ = rng.choice(["kg", "g"])
    m = rng.choice([0.2, 0.5, 1, 2]) if mu_ == "kg" else rng.choice([100, 250, 400])
    xu = rng.choice(["m", "cm"])
    x = rng.choice([0.05, 0.1, 0.2]) if xu == "m" else rng.choice([3, 5, 8, 12])
    ask, au = rng.choice([("period", "s"), ("energy", "J"), ("max_speed", "m/s")])
    word = {"period": "진동 주기는 몇 초", "energy": "처음 용수철에 저장된 탄성 에너지는 몇 J",
            "max_speed": "물체의 최대 속력은 몇 m/s"}[ask]
    qtext = (f"용수철 상수가 {k} N/m인 용수철에 질량 {m:g} {KO_UNIT[mu_]}의 물체를 달고 평형 위치에서 "
             f"{x:g} {KO_UNIT[xu]} 당겼다가 놓았다. 마찰은 없다. {word}인가?")
    params = {"k": q(k, "N/m"), "m": q(m, mu_), "x": q(x, xu)}
    return qtext, {"template": "spring", "params": params, "ask": ask, "unit": au}, \
        gold_spring(k, si(m, mu_), si(x, xu), ask)


def _incline(rng):
    ang = rng.choice([20, 25, 30, 37, 45, 53])
    mu = rng.choice([0, 0.1, 0.2, 0.3])
    while mu >= math.tan(math.radians(ang)):
        mu = round(mu - 0.1, 1)
    m = rng.choice([1, 2, 5, 10])
    ask, au = rng.choice([("acceleration", "m/s^2"), ("normal_force", "N")])
    fr = "마찰이 없다" if mu == 0 else f"운동 마찰 계수는 {mu:g}이다"
    word = {"acceleration": "미끄러져 내려가는 가속도의 크기는 몇 m/s²", "normal_force": "면이 물체를 떠받치는 수직항력은 몇 N"}[ask]
    qtext = f"질량 {m} kg인 상자가 {ang}° 기울어진 경사면을 미끄러져 내려간다. {fr}. {word}인가?"
    params = {"angle": f"{ang} deg", "mu": f"{mu:g}", "m": q(m, "kg")}
    return qtext, {"template": "incline", "params": params, "ask": ask, "unit": au}, gold_incline(ang, mu, G_EARTH, m, ask)


def _heat(rng):
    sub, c = rng.choice([("물", 4186), ("알루미늄", 900), ("구리", 385), ("철", 450)])
    mu_ = rng.choice(["kg", "g"])
    m = rng.choice([0.5, 1, 2, 3]) if mu_ == "kg" else rng.choice([200, 250, 500])
    dT = rng.choice([5, 10, 20, 35, 60])
    au = rng.choice(["J", "kJ"])
    qtext = (f"{sub}(비열 {c} J/(kg·K)) {m:g} {KO_UNIT[mu_]}의 온도를 {dT} °C 올리는 데 필요한 열량은 몇 "
             f"{KO_UNIT[au]}인가? 상태 변화와 열 손실은 없다.")
    params = {"m": q(m, mu_), "c": q(c, "J/(kg*K)"), "dT": f"{dT} K"}
    gold = si(m, mu_) * c * dT / TO_SI[au]
    return qtext, {"template": "heat", "params": params, "ask": "energy", "unit": au}, gold


def _cooling(rng):
    T0, Tenv = rng.choice([(90, 20), (80, 25), (70, 22), (100, 18), (60, 15)])
    k = rng.choice([0.03, 0.05, 0.08, 0.1])
    tu = rng.choice(["min", "s"])
    t = rng.choice([5, 10, 15, 30]) if tu == "min" else rng.choice([300, 600, 900])
    qtext = (f"{T0} °C인 음료가 {Tenv} °C인 방에 놓여 뉴턴의 냉각 법칙(냉각 상수 {k:g} /분)을 따라 식는다. "
             f"{t:g} {KO_UNIT[tu]} 뒤 음료의 온도는 몇 °C인가?")
    params = {"T0": f"{T0} degC", "Tenv": f"{Tenv} degC", "k": q(k, "1/min"), "t": q(t, tu)}
    gold = gold_cooling(T0, Tenv, k / 60.0, si(t, tu))
    return qtext, {"template": "cooling", "params": params, "ask": "temperature", "unit": "degC"}, gold


def _rc(rng):
    V = rng.choice([5, 9, 12, 24])
    R = rng.choice([1, 2.2, 4.7, 10])
    C = rng.choice([10, 47, 100, 220])
    tau_ms = R * 1e3 * C * 1e-6 * 1e3
    t = _r(tau_ms * rng.choice([0.5, 1, 2, 3]), 3)
    ask = rng.choice(["voltage", "voltage", "time_constant"])
    if ask == "voltage":
        qtext = (f"방전된 {C} μF 축전기를 {R:g} kΩ 저항과 직렬로 {V} V 전지에 연결했다. "
                 f"연결하고 {t:g} ms 뒤 축전기 양단의 전압은 몇 V인가?")
        au, gold = "V", gold_rc(V, R * 1e3, C * 1e-6, t * 1e-3, "voltage")
    else:
        qtext = f"{R:g} kΩ 저항과 {C} μF 축전기로 이루어진 RC 회로의 시간 상수는 몇 ms인가?"
        au, gold = "ms", R * 1e3 * C * 1e-6 / 1e-3
    params = {"V": q(V, "V"), "R": q(R, "kohm"), "C": q(C, "uF"), "t": q(t, "ms")}
    return qtext, {"template": "rc_charge", "params": params, "ask": ask, "unit": au}, gold


def _ohm(rng):
    V = rng.choice([1.5, 3, 5, 9, 12, 220])
    ru = rng.choice(["ohm", "kohm"])
    R = rng.choice([2, 4, 10, 47, 100]) if ru == "ohm" else rng.choice([1, 2.2, 4.7])
    ask, au = rng.choice([("current", "A"), ("current", "mA"), ("power", "W")])
    word = {"current": "흐르는 전류", "power": "저항에서 소비되는 전력"}[ask]
    qtext = f"{R:g} {KO_UNIT[ru]} 저항 양단에 {V:g} V의 전압을 걸었다. {josa(word)} 몇 {KO_UNIT[au]}인가?"
    Rsi = si(R, ru)
    gold = (V / Rsi / TO_SI[au]) if ask == "current" else V * V / Rsi
    return qtext, {"template": "ohm", "params": {"V": q(V, "V"), "R": q(R, ru)}, "ask": ask, "unit": au}, gold


def _ideal_gas(rng):
    n = rng.choice([0.5, 1, 2, 3])
    tu = rng.choice(["degC", "K"])
    T = rng.choice([0, 20, 27, 100]) if tu == "degC" else rng.choice([250, 300, 350])
    vu = rng.choice(["L", "m^3"])
    V = rng.choice([10, 22.4, 25, 50]) if vu == "L" else rng.choice([0.01, 0.05, 0.1])
    au = rng.choice(["Pa", "kPa"])
    tt = f"{T} °C" if tu == "degC" else f"{T} K"
    qtext = f"이상 기체 {n:g} mol이 부피 {V:g} {KO_UNIT[vu]}의 용기에 들어 있고 온도는 {tt}이다. 압력은 몇 {KO_UNIT[au]}인가?"
    TK = T + 273.15 if tu == "degC" else T
    gold = n * 8.314462618 * TK / si(V, vu) / TO_SI[au]
    params = {"n": q(n, "mol"), "T": f"{T} {tu}", "V": q(V, vu)}
    return qtext, {"template": "ideal_gas", "params": params, "ask": "pressure", "unit": au}, gold


BUILDERS = {"projectile": _projectile, "free_fall": _free_fall, "pendulum": _pendulum, "spring": _spring,
            "incline": _incline, "heat": _heat, "cooling": _cooling, "rc_charge": _rc, "ohm": _ohm,
            "ideal_gas": _ideal_gas}

UNSUPPORTED = [
    "공기 저항 계수가 0.47인 지름 7 cm 공을 30 m/s로 쏘아 올렸다. 공기 밀도 1.2 kg/m³일 때 최고 높이는 몇 m인가?",
    "속력 0.8c로 움직이는 우주선의 시계로 1시간이 지날 때 지구의 시계로는 몇 시간이 지나는가?",
    "지구 표면에서 고도 400 km 원궤도를 도는 위성의 공전 주기는 몇 분인가?",
    "물이 지름 2 cm 관에서 3 m/s로 흐르다가 지름 1 cm 관으로 들어간다. 압력 차이는 몇 kPa인가?",
    "표면 온도 5800 K인 별의 단위 면적당 복사 에너지는 몇 W/m²인가?",
    "두께 2 cm 유리창(열전도율 0.8 W/(m·K)) 양쪽 온도 차가 20 K일 때 1 m²를 통과하는 열 흐름은 몇 W인가?",
    "질량 2 kg 막대(길이 1 m)를 한쪽 끝을 축으로 돌릴 때 관성 모멘트는 몇 kg·m²인가?",
    "10 Ω 저항과 0.5 H 인덕터를 12 V에 직렬 연결했을 때 시간 상수는 몇 초인가?",
    "1 kg 얼음(0 °C)을 모두 녹이는 데 필요한 열량은 몇 kJ인가?",
    "진동수 440 Hz 소리가 20 m/s로 다가오는 구급차에서 날 때 서 있는 사람이 듣는 진동수는 몇 Hz인가?",
    "두 전하 2 μC와 -3 μC가 0.5 m 떨어져 있다. 두 전하 사이의 힘은 몇 N인가?",
    "질량 3 kg 공이 2 m/s로 움직여 정지한 1 kg 공과 완전 탄성 충돌했다. 충돌 후 1 kg 공의 속력은 몇 m/s인가?",
    "단열 과정에서 이원자 기체의 부피가 절반이 되면 압력은 몇 배가 되는가?",
    "굴절률 1.5인 유리에 빛이 30°로 입사할 때 굴절각은 몇 도인가?",
    "반지름 0.1 m 원형 코일(감은 수 50)에 2 A가 흐를 때 중심의 자기장은 몇 mT인가?",
    "회전 반지름 0.5 m에서 원운동하는 1 kg 물체의 주기가 2초일 때 구심력은 몇 N인가?",
    "질량 70 kg인 사람이 5 m 높이 다이빙대에서 뛰어내릴 때 물에 닿는 순간 운동량은 몇 kg·m/s인가? (단, 수면에서 0.5 m 깊이까지 감속)",
    "온도 300 K, 1 기압 공기 중 소리의 속력은 몇 m/s인가?",
    "질량 1000 kg 자동차가 반지름 50 m 커브를 마찰 계수 0.7로 돌 때 미끄러지지 않는 최대 속력은 몇 km/h인가?",
    "전자기파의 진동수가 2.4 GHz일 때 파장은 몇 cm인가?",
]


# ---- planted errors ----

def _wrong(rng, item_spec, gold):
    """A typical slip, at least 2% away from gold."""
    au = item_spec["unit"]
    options = []
    if au in ("m", "J", "W", "A", "V", "Pa", "N"):
        options += [("unit_x1000", gold * 1000), ("unit_div1000", gold / 1000)]
    if au in ("s", "m/s"):
        options += [("unit_x60", gold * 60), ("unit_kmh", gold * 3.6)]
    if au in ("kJ", "kPa", "mA", "ms"):
        options += [("unit_x1000", gold * 1000)]
    if au == "degC":
        options += [("kelvin_slip", gold + 273.15), ("env_ignored", gold * 0.5)]
    options += [("factor2", gold * 2), ("half", gold / 2), ("near_5pct", gold * rng.choice([0.95, 1.05])),
                ("near_3pct", gold * rng.choice([0.97, 1.03]))]
    if gold != 0:
        options.append(("sign", -gold))
    name, val = rng.choice(options)
    if gold != 0 and abs(val - gold) / abs(gold) < 0.02:
        name, val = "factor2", gold * 2
    return {"value": _r(val, 4), "error": name}


def make(seed: int = 0, per_template: int = 22, unsupported: int = 20) -> list[dict]:
    rng = random.Random(seed)
    items = []
    for name, build in BUILDERS.items():
        seen = set()
        tries = 0
        while sum(1 for it in items if it.get("spec", {}).get("template") == name) < per_template:
            tries += 1
            if tries > per_template * 50:
                raise RuntimeError(f"{name}: parameter space too small for {per_template} distinct items")
            qtext, spec, gold = build(rng)
            if qtext in seen:
                continue
            seen.add(qtext)
            g = _r(gold, 6)
            claims = [{"value": _r(gold, 4), "error": None}, _wrong(rng, spec, gold)]
            rng.shuffle(claims)
            items.append({"id": f"{name}-{len(seen):02d}", "question": qtext, "status": "supported",
                          "spec": spec, "gold": g, "claims": claims})
    for i, qtext in enumerate(UNSUPPORTED[:unsupported]):
        items.append({"id": f"unsupported-{i:02d}", "question": qtext, "status": "unsupported"})
    return items


# ---- scoring ----

def _close(a, b, rel=0.01, abs_=1e-9):
    return a is not None and b is not None and math.isclose(float(a), float(b), rel_tol=rel, abs_tol=abs_)


def spec_equal(pred: dict | None, ref: dict) -> bool:
    """Same template, ask and answer unit, and every reference parameter equal after unit conversion."""
    from . import physics

    if not pred or pred.get("template") != ref["template"] or pred.get("ask") != ref["ask"]:
        return False
    if pred.get("unit") and str(physics.Q(f"1 {PINT.get(pred['unit'], pred['unit'])}").units) != \
            str(physics.Q(f"1 {PINT.get(ref['unit'], ref['unit'])}").units):
        return False
    for k, v in ref["params"].items():
        if k not in (pred.get("params") or {}):
            return False
        try:
            a = physics._to_kelvin_if_temperature(physics.Q(pred["params"][k]))
            b = physics._to_kelvin_if_temperature(physics.Q(v))
            if a.dimensionality != b.dimensionality or not _close(a.to(b.units).magnitude, b.magnitude):
                return False
        except Exception:
            return False
    return True


def p0(items: list[dict]) -> dict:
    """Engine only: reference spec + each claim -> verify. Measures the engine, not a model."""
    from . import physics

    res = Counter()
    per = defaultdict(Counter)
    misses = []
    for it in items:
        if it["status"] == "unsupported":
            continue
        status, out = physics.solve(it["spec"])
        t = it["spec"]["template"]
        if status != "ok":
            res["engine_error"] += 1
            misses.append({"id": it["id"], "why": f"{status}: {out}"})
            continue
        if not _close(out.magnitude, it["gold"], rel=1e-3):
            res["engine_vs_gold_off"] += 1
            misses.append({"id": it["id"], "why": f"engine {out.magnitude:.6g} vs gold {it['gold']:.6g}"})
        for c in it["claims"]:
            v = physics.verify(it["spec"], c["value"])["status"]
            key = "correct" if c["error"] is None else "wrong"
            res[f"{key}_n"] += 1
            per[t][f"{key}_n"] += 1
            if key == "correct" and v != "match":
                res["false_block"] += 1
                per[t]["false_block"] += 1
            if key == "wrong" and v == "mismatch":
                res["detected"] += 1
                per[t]["detected"] += 1
            if key == "wrong" and v != "mismatch":
                misses.append({"id": it["id"], "why": f"missed {c['error']} ({c['value']} vs gold {it['gold']})"})
    out = {"kind": "genesis_predict.physics_p0/v1", "supported_items": res["correct_n"],
           "engine_vs_gold_agree": 1 - res["engine_vs_gold_off"] / max(res["correct_n"], 1),
           "detection_rate": res["detected"] / max(res["wrong_n"], 1),
           "false_block_rate": res["false_block"] / max(res["correct_n"], 1),
           "engine_errors": res["engine_error"],
           "per_template": {t: {"detection": c["detected"] / max(c["wrong_n"], 1),
                                "false_block": c["false_block"] / max(c["correct_n"], 1)} for t, c in sorted(per.items())},
           "misses": misses[:30]}
    out["pass"] = out["detection_rate"] >= 0.8 and out["false_block_rate"] <= 0.02 and out["engine_errors"] == 0
    return out


def score_model(items: list[dict], outputs: dict[str, dict]) -> dict:
    """Score a model run. ``outputs[id] = {"spec": dict|None, "answer": float|None, "unsupported": bool}``.

    * accuracy: answer within 1% of gold (supported items)
    * spec_accuracy: spec equal to the reference spec
    * engine_rescue: answer wrong but the engine (on the model's own spec) gives gold - what a
      'trust the engine' policy would win
    * unsupported_recall: out-of-scope items the model declined (instead of inventing a spec)
    """
    from . import physics

    c = Counter()
    for it in items:
        o = outputs.get(it["id"], {})
        if it["status"] == "unsupported":
            c["uns_n"] += 1
            c["uns_declined"] += bool(o.get("unsupported"))
            continue
        c["n"] += 1
        ok = _close(o.get("answer"), it["gold"])
        c["acc"] += ok
        c["spec_ok"] += spec_equal(o.get("spec"), it["spec"])
        if o.get("spec"):
            st, val = physics.solve(o["spec"])
            eng_ok = st == "ok" and _close(val.magnitude, it["gold"])
            c["engine_ok"] += eng_ok
            c["rescue"] += (not ok) and eng_ok
            c["harm"] += ok and st == "ok" and not eng_ok
    n = max(c["n"], 1)
    return {"items": c["n"], "accuracy": c["acc"] / n, "spec_accuracy": c["spec_ok"] / n,
            "engine_on_model_spec": c["engine_ok"] / n, "engine_rescue": c["rescue"] / n, "engine_harm": c["harm"] / n,
            "unsupported_recall": c["uns_declined"] / max(c["uns_n"], 1)}


PROMPT = """다음 물리 문제를 풀어라. 마지막 두 줄을 반드시 아래 형식으로 써라.
시나리오: {"template": ..., "params": {...단위를 붙인 값...}, "ask": ..., "unit": ...}
정답: 숫자
쓸 수 있는 template과 ask: projectile(range, max_height, flight_time; v0, angle, h0, g), free_fall(time, impact_speed; h, g),
pendulum(period; L, g), spring(period, energy, max_speed; k, m, x), incline(acceleration, normal_force; angle, mu, g, m),
heat(energy; m, c, dT), cooling(temperature; T0, Tenv, k, t), rc_charge(voltage, time_constant; V, R, C, t),
ohm(current, power; V, R), ideal_gas(pressure; n, T, V). 이 목록으로 풀 수 없는 문제면 시나리오 대신 "시나리오: 미지원"이라고 써라.
문제: """


PROMPT_V2 = """다음 물리 문제에 맞는 시나리오를 써라. 계산은 시나리오를 받은 계산기가 한다.
반드시 아래 두 줄만 써라.
시나리오: {"template": 이름, "params": {이름: "숫자 단위"}, "ask": 구할 것, "unit": 답의 단위}
정답: 숫자

규칙
- params의 값은 문제에 나온 숫자와 단위를 그대로 쓴 문자열이다. 예: "36 km/h", "200 cm", "45 deg", "4.7 kohm", "10 uF", "90 degC", "0.1 1/min".
- 마찰 계수 mu처럼 단위가 없는 값만 숫자로 쓴다. 지구에서는 g를 쓰지 않는다.
- 아래 목록으로 풀 수 없는 문제면 "시나리오: 미지원"이라고만 써라.

template (ask / params)
projectile (range, max_height, flight_time / v0, angle, h0, g)
free_fall (time, impact_speed / h, g)
pendulum (period / L, g)
spring (period, energy, max_speed / k, m, x)
incline (acceleration, normal_force / angle, mu, m, g)
heat (energy / m, c, dT)
cooling (temperature / T0, Tenv, k, t)
rc_charge (voltage, time_constant / V, R, C, t)
ohm (current, power / V, R)
ideal_gas (pressure / n, T, V)

예시
문제: 4 Ω 저항 양단에 12 V를 걸었다. 저항에서 소비되는 전력은 몇 W인가?
시나리오: {"template": "ohm", "params": {"V": "12 V", "R": "4 ohm"}, "ask": "power", "unit": "W"}
정답: 36

문제: """


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("--seed", type=int, default=0)
    m.add_argument("--per-template", type=int, default=22)
    m.add_argument("--unsupported", type=int, default=20)
    m.add_argument("--out", required=True)
    z = sub.add_parser("p0")
    z.add_argument("--set", required=True)
    a = p.parse_args(argv)
    if a.cmd == "make":
        items = make(a.seed, a.per_template, a.unsupported)
        with open(a.out, "w", encoding="utf-8") as f:
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
        print(json.dumps({"items": len(items), "by": Counter(it.get("spec", {}).get("template", "unsupported") for it in items)},
                         ensure_ascii=False))
    else:
        items = [json.loads(line) for line in open(a.set, encoding="utf-8") if line.strip()]
        print(json.dumps(p0(items), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
