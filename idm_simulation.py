
import matplotlib.pyplot as plt
import argparse
import math
import os
import random
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import deque

import numpy as np
import matplotlib
matplotlib.use("Agg")

P = dict(v0=25.0, T=1.5, a=1.4, b=2.0, s0=2.0, delta=4.0, length=5.0)
DT = 0.2
C = ["#1f4e79", "#c0504d", "#4f8f3f", "#7f6000", "#6b5b95"]
VMAX_PLOT = 25.0

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Liberation Serif", "Times New Roman", "DejaVu Serif"],
    "font.size": 12, "axes.titlesize": 12, "axes.labelsize": 12, "legend.fontsize": 11,
    "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 150, "savefig.dpi": 200,
    "mathtext.fontset": "stix", "axes.unicode_minus": False,
})

CFG = dict(out="results", work=None, backend="sumo",
           gui=False, delay_ms=50, native=False, libsumo=False)
SUMMARY = []


def say(msg):
    print(msg, flush=True)
    SUMMARY.append(msg)


def s_eq(V, p=P):
    return (p["s0"] + V * p["T"]) / math.sqrt(1.0 - (V / p["v0"]) ** p["delta"])


def eq_curve(p=P, n=1500):
    V = np.linspace(0.0, p["v0"] * 0.9999, n)
    rho = 1.0 / (np.array([s_eq(v, p) for v in V]) + p["length"])
    return V, rho, rho * V


def veq_for_density(rho, p=P):
    lo, hi = 0.0, p["v0"] * 0.9999
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if 1.0 / (s_eq(mid, p) +
                                     p["length"]) > rho else (lo, mid)
    return 0.5 * (lo + hi)


def idm_accel(v, gap, vl, p=P):
    free = 1.0 - (v / p["v0"]) ** p["delta"]
    s_star = p["s0"] + max(0.0, v * p["T"] + v * (v - vl) /
                           (2.0 * math.sqrt(p["a"] * p["b"])))
    return p["a"] * (free - (s_star / max(gap, 0.1)) ** 2)


# ---------------------------------------------------------------- pure Python backend
class PySim:
    """Euler scheme as in SUMO: v <- max(0, v + a dt), x <- x + v dt. Exposes the same
    vehicle-control calls as traci.vehicle."""

    def __init__(self, p, vehicles, length, ring):
        self.p, self.length, self.ring, self.time = p, length, ring, 0.0
        self.pending = sorted(vehicles, key=lambda q: q["depart"])
        self.veh = {}

    def setSpeedMode(self, vid, mode):
        pass

    def setSpeed(self, vid, v):
        if vid in self.veh:
            self.veh[vid]["set"] = None if v < 0 else v

    def remove(self, vid):
        self.veh.pop(vid, None)

    def state(self):
        return {k: (v["x"], v["v"]) for k, v in self.veh.items()}

    def step(self):
        p, order = self.p, sorted(self.veh.items(), key=lambda kv: -kv[1]["x"])
        new = {}
        for j, (vid, v) in enumerate(order):
            if v["set"] is not None:
                new[vid] = v["set"]
                continue
            lead = order[j - 1][1] if (self.ring and len(order)
                                       > 1) or (not self.ring and j > 0) else None
            if lead is None:
                a = p["a"] * (1.0 - (v["v"] / p["v0"]) ** p["delta"])
            else:
                gap = lead["x"] - v["x"] + \
                    (self.length if self.ring and j ==
                     0 else 0.0) - p["length"]
                a = idm_accel(v["v"], gap, lead["v"], p)
            new[vid] = max(0.0, v["v"] + a * DT)
        for vid, nv in new.items():
            self.veh[vid]["v"] = nv
            self.veh[vid]["x"] += nv * DT
            if self.ring:
                self.veh[vid]["x"] %= self.length
        if not self.ring:
            for vid in [k for k, v in self.veh.items() if v["x"] >= self.length]:
                del self.veh[vid]
        t0 = self.time
        self.time = round(self.time + DT, 6)
        while self.pending and self.pending[0]["depart"] <= t0 + 1e-9:
            q = self.pending.pop(0)
            self.veh[q["id"]] = dict(
                x=q["x"] % self.length if self.ring else q["x"], v=q["v"], set=None)


def run_python(length, ring, p, vehicles, end, on_state, decide):
    sim = PySim(p, vehicles, length, ring)
    sim.step()
    while True:
        t = round(sim.time - DT, 6)
        st = sim.state()
        on_state(t, st)
        if t >= end - 1e-9:
            break
        if decide:
            decide(t, st, sim)
        sim.step()


# ---------------------------------------------------------------- SUMO backend
def sumo_bin(name):
    exe = name + (".exe" if os.name == "nt" else "")
    home = os.environ.get("SUMO_HOME")
    if home and os.path.isfile(os.path.join(home, "bin", exe)):
        return os.path.join(home, "bin", exe)
    found = shutil.which(name)
    if found:
        return found
    sys.exit(f"[ERROR] {exe} not found: set SUMO_HOME or add SUMO to PATH")


def get_traci():
    home = os.environ.get("SUMO_HOME")
    if home and os.path.join(home, "tools") not in sys.path:
        sys.path.append(os.path.join(home, "tools"))
    try:
        if CFG["libsumo"] and not CFG["gui"]:
            import libsumo as traci
        else:
            import traci
        return traci
    except ImportError:
        sys.exit(
            "[ERROR] module traci not found: set SUMO_HOME or run pip install traci")


def run_netconvert(nod, edg, net):
    subprocess.run([sumo_bin("netconvert"), "--node-files", nod, "--edge-files", edg, "--output-file", net,
                    "--no-internal-links", "true", "--no-turnarounds", "true", "--no-warnings", "true"],
                   check=True)


_NETS = {}


def ensure_net(length, ring):
    """Straight road seg_i (50 m edges) or ring ring_i (12 edges); edge length is set explicitly,
    so the coordinate is i * seg + position on the edge."""
    if (length, ring) in _NETS:
        return _NETS[(length, ring)]
    n = 12 if ring else max(1, round(length / 50.0))
    seg = length / n
    base = os.path.join(
        CFG["work"], f"{'ring' if ring else 'road'}_{int(length)}")
    nodes, edges = ET.Element("nodes"), ET.Element("edges")
    if ring:
        R = length / (2 * math.pi)
        def pt(a): return (R * math.cos(a), R * math.sin(a))
        for i in range(n):
            x, y = pt(2 * math.pi * i / n)
            ET.SubElement(
                nodes, "node", id=f"n{i}", x=f"{x:.3f}", y=f"{y:.3f}", type="priority")
            a0, a1 = 2 * math.pi * i / n, 2 * math.pi * (i + 1) / n
            shape = " ".join("%.3f,%.3f" % pt(a0 + (a1 - a0) * j / 6)
                             for j in range(7))
            ET.SubElement(edges, "edge", id=f"ring_{i}", **{"from": f"n{i}", "to": f"n{(i + 1) % n}"},
                          numLanes="1", speed="45", length=f"{seg:.6f}", shape=shape)
    else:
        for i in range(n + 1):
            ET.SubElement(
                nodes, "node", id=f"n{i}", x=f"{i * seg:.3f}", y="0", type="priority")
        for i in range(n):
            ET.SubElement(edges, "edge", id=f"seg_{i}", **{"from": f"n{i}", "to": f"n{i + 1}"},
                          numLanes="1", speed="45", length=f"{seg:.6f}")
    for root, f in ((nodes, base + ".nod.xml"), (edges, base + ".edg.xml")):
        ET.indent(root, space="  ")
        ET.ElementTree(root).write(f, encoding="utf-8", xml_declaration=True)
    run_netconvert(base + ".nod.xml", base + ".edg.xml", base + ".net.xml")
    _NETS[(length, ring)] = (base + ".net.xml", n, seg)
    return _NETS[(length, ring)]


def write_rou(path, p, vehicles, ring, n, seg, length):
    root = ET.Element("routes")
    ET.SubElement(root, "vType", id="car", vClass="passenger", length=f"{p['length']:g}", minGap=f"{p['s0']:g}",
                  maxSpeed=f"{p['v0']:g}", accel=f"{p['a']:g}", decel=f"{p['b']:g}", emergencyDecel="30",
                  sigma="0", tau=f"{p['T']:g}", speedFactor="1", speedDev="0", carFollowModel="IDM",
                  delta=f"{p['delta']:g}", guiShape="passenger/sedan")
    names = [f"{'ring' if ring else 'seg'}_{i}" for i in range(n)]
    ET.SubElement(root, "route", id="main", edges=" ".join(
        names), **({"repeat": "300"} if ring else {}))
    for v in sorted(vehicles, key=lambda q: q["depart"]):
        x = v["x"] % length if ring else v["x"]
        i = min(int(x // seg), n - 1)
        ET.SubElement(root, "vehicle", id=v["id"], type="car", route="main", depart=f"{v['depart']:.3f}",
                      departLane="0", departPos=f"{max(0.0, x - i * seg):.4f}", departSpeed=f"{v['v']:.4f}",
                      departEdge=str(i), insertionChecks="none")
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def run_sumo(tag, length, ring, p, vehicles, end, on_state, decide, speedmode0):
    traci = get_traci()
    net, n, seg = ensure_net(length, ring)
    rou = os.path.join(CFG["work"], f"{tag}.rou.xml")
    write_rou(rou, p, vehicles, ring, n, seg, length)
    cmd = [sumo_bin("sumo-gui" if CFG["gui"] else "sumo"), "-n", net, "-r", rou, "--step-length", f"{DT:g}",
           "--begin", "0", "--end", str(end +
                                        20), "--no-step-log", "true", "--no-warnings", "true",
           "--time-to-teleport", "-1", "--collision.action", "warn", "--step-method.ballistic", "false"]
    if CFG["gui"]:
        cmd += ["--start", "--quit-on-end", "--delay", str(CFG["delay_ms"])]
    traci.start(cmd)
    K = traci.constants
    var = [K.VAR_ROAD_ID, K.VAR_LANEPOSITION, K.VAR_SPEED]
    prefix = "ring_" if ring else "seg_"
    try:
        traci.simulationStep()
        while True:
            for vid in traci.simulation.getDepartedIDList():
                if speedmode0:
                    traci.vehicle.setSpeedMode(vid, 0)
                traci.vehicle.subscribe(vid, var)
            t = round(traci.simulation.getTime() - DT, 6)
            st = {}
            for vid, r in traci.vehicle.getAllSubscriptionResults().items():
                if r[K.VAR_ROAD_ID].startswith(prefix):
                    x = int(r[K.VAR_ROAD_ID][len(prefix):]) * \
                        seg + r[K.VAR_LANEPOSITION]
                    st[vid] = (x % length if ring else x, r[K.VAR_SPEED])
            on_state(t, st)
            if t >= end - 1e-9:
                break
            if decide:
                decide(t, st, traci.vehicle)
            traci.simulationStep()
    finally:
        traci.close()


def run(tag, length, ring, vehicles, end, on_state, decide=None, p=P, speedmode0=False):
    if CFG["backend"] == "python":
        run_python(length, ring, p, vehicles, end, on_state, decide)
    else:
        run_sumo(tag, length, ring, p, vehicles,
                 end, on_state, decide, speedmode0)


def external_speeds(st):
    """Next speeds for a straight road with the pure IDM formula (leader = car ahead, obstacle = stopped car)."""
    order = sorted(st, key=lambda k: -st[k][0])
    out = {}
    for j, vid in enumerate(order):
        v = st[vid][1]
        if j == 0:
            a = P["a"] * (1.0 - (v / P["v0"]) ** P["delta"])
        else:
            lead = st[order[j - 1]]
            a = idm_accel(v, lead[0] - st[vid][0] - P["length"], lead[1])
        out[vid] = max(0.0, v + a * DT)
    return out


def use_external():
    return CFG["backend"] == "sumo" and not CFG["native"]


# ---------------------------------------------------------------- space-time diagram
class XT:
    """Cells of 50 m x 10 s; cell speed = distance travelled / time spent (as SUMO edgeData)."""

    def __init__(self, length, tmax, seg=50.0, dtb=10.0):
        self.seg, self.dtb = seg, dtb
        self.nx, self.nt = int(math.ceil(length / seg)
                               ), int(math.ceil(tmax / dtb))
        self.S, self.T = np.zeros(
            (self.nt, self.nx)), np.zeros((self.nt, self.nx))

    def add(self, t, x, v):
        ti, xi = min(int(t // self.dtb), self.nt -
                     1), min(int(x // self.seg), self.nx - 1)
        self.S[ti, xi] += v * DT
        self.T[ti, xi] += DT

    def field(self):
        M = np.where(self.T > 0, self.S / np.maximum(self.T, 1e-12), np.nan)
        return np.arange(self.nx + 1) * self.seg, np.arange(self.nt + 1) * self.dtb, M


def draw_xt(ax, xt, title=None, xlim=None):
    x, t, M = xt.field()
    mesh = ax.pcolormesh(x, t, M * 3.6, cmap="RdYlGn",
                         vmin=0, vmax=VMAX_PLOT * 3.6, shading="flat")
    ax.set_facecolor("#dddddd")
    ax.set_ylim(t[-1], 0)
    if xlim:
        ax.set_xlim(*xlim)
    ax.set_xlabel("Координата, м")
    ax.grid(False)
    if title:
        ax.set_title(title, fontsize=11)
    return mesh


def save(fig, name, **kw):
    path = os.path.join(CFG["out"], name)
    fig.savefig(path, **kw)
    plt.close(fig)
    say(f"[OK] {path}")


# ---------------------------------------------------------------- figures 1-2
def exp_eq():
    dd = P["length"] + P["s0"]
    r = np.linspace(0, 1000 / dd, 600) / 1000
    Vt = np.minimum(P["v0"], (1 / np.maximum(r, 1e-9) - dd) / P["T"])
    for key, ylab, ylim, name in (("V", "Равновесная скорость $V$, км/ч", (0, 95), "fig01_equilibrium_speed.png"),
                                  ("Q", "Интенсивность $Q$, авт/ч", (0, 2400), "fig02_equilibrium_flow.png")):
        fig, ax = plt.subplots(figsize=(6.2, 3.8))
        for d, c in zip((1, 2, 4), C):
            V, rho, Q = eq_curve(dict(P, delta=float(d)))
            ax.plot(rho * 1000, V * 3.6 if key == "V" else Q *
                    3600, color=c, label=rf"$\delta={d}$")
        ax.plot(r * 1000, Vt * 3.6 if key == "V" else r * Vt *
                3600, "k--", lw=1.2, label=r"$\delta\to\infty$")
        ax.set_xlabel(r"Плотность $\rho$, авт/км")
        ax.set_ylabel(ylab)
        ax.set_xlim(0, 150)
        ax.set_ylim(*ylim)
        ax.legend()
        fig.tight_layout()
        save(fig, name)


# ---------------------------------------------------------------- figure 3
def exp_free():
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    for d, c in zip((1, 2, 4, 10), C):
        rec = []
        run(f"free_d{d}", 3000.0, False, [dict(id="car", depart=0, x=0.0, v=0.0)], 80.0,
            lambda t, st: rec.append((t, st["car"][1])) if "car" in st else None, p=dict(P, delta=float(d)))
        t, v = np.array([q[0] for q in rec]), np.array([q[1] for q in rec])
        ax.plot(t, v * 3.6, color=c, label=rf"$\delta={d}$")
        say(
            f"     delta={d}: 95 % of V0 reached at {t[int(np.argmax(v >= 0.95 * P['v0']))]:.1f} s")
    ax.axhline(P["v0"] * 3.6, color="k", ls=":", lw=1)
    ax.text(78, P["v0"] * 3.6 + 1.5, r"$V^0$", ha="right")
    ax.set_xlabel("Время, с")
    ax.set_ylabel("Скорость, км/ч")
    ax.set_xlim(0, 80)
    ax.set_ylim(0, 98)
    ax.legend(loc="lower right")
    fig.tight_layout()
    save(fig, "fig03_free_acceleration.png")


# ---------------------------------------------------------------- figures 4-6
def brake_decide(t, st, veh):
    nv = external_speeds(st) if use_external() else {}
    nv["blk"] = 0.0
    for vid, v in nv.items():
        veh.setSpeed(vid, v)


def exp_brake():
    xw = 400.0
    data = []
    for v0i in (10, 15, 20):
        rec = []
        run(f"brake_{v0i}", 700.0, False,
            [dict(id="blk", depart=0, x=xw + P["length"], v=0.0),
             dict(id="car", depart=0, x=xw - 300.0, v=float(v0i))],
            70.0, lambda t, st, rec=rec: rec.append(
                (t, st["car"][1], xw - st["car"][0])) if "car" in st else None,
            brake_decide, speedmode0=use_external())
        t, v, g = (np.array([q[i] for q in rec]) for i in range(3))
        acc = np.diff(v) / DT
        data.append((v0i, t, v, g, acc))
        say(
            f"     v0={v0i} m/s: max deceleration {-acc.min():.2f} m/s2, final gap {g[-1]:.2f} m, max speed {v.max() * 3.6:.1f} km/h")
    for idx, (ylab, name) in enumerate((("Скорость, км/ч", "fig04_braking_speed.png"),
                                        ("Расстояние до препятствия, м",
                                         "fig05_braking_distance.png"),
                                        ("Ускорение, м/с²", "fig06_braking_acceleration.png"))):
        fig, ax = plt.subplots(figsize=(6.2, 3.8))
        for (v0i, t, v, g, acc), c in zip(data, C):
            tt, yy = (t, v * 3.6) if idx == 0 else (t,
                                                    g) if idx == 1 else (t[1:], acc)
            ax.plot(tt, yy, color=c, label=rf"$v_0={v0i}$ м/с")
        if idx == 2:
            ax.axhline(-P["b"], color="k", ls=":", lw=1)
            ax.text(68, -P["b"] + 0.15, r"$-b$", ha="right")
        ax.set_xlabel("Время, с")
        ax.set_ylabel(ylab)
        ax.set_xlim(0, 70)
        if idx == 1:
            ax.set_ylim(0, 310)
        ax.legend(loc="upper right")
        fig.tight_layout()
        save(fig, name)


# ---------------------------------------------------------------- figure 7
def exp_fd(t_end=450.0):
    L = 1500.0
    pts = []
    _, rho_a, Q_a = eq_curve(P, 2000)
    for N in [10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 100, 120, 150, 180, 200]:
        means = []
        run(f"fd_{N}", L, True, [dict(id=f"car_{i}", depart=0, x=(L - i * L / N) % L, v=6.0) for i in range(N)],
            t_end, lambda t, st, means=means: means.append((t, float(np.mean([s[1] for s in st.values()])))) if st else None)
        vm = float(np.mean([m[1] for m in means if m[0] >= t_end - 100.0]))
        rho = N / L
        pts.append((rho * 1000, rho * vm * 3600))
        say(f"     N={N:3d}  rho={rho * 1000:6.1f} veh/km  Q={rho * vm * 3600:7.1f} veh/h  "
            f"(formula {float(np.interp(rho, rho_a[::-1], Q_a[::-1])) * 3600:7.1f})")
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.plot(rho_a * 1000, Q_a * 3600, color=C[0], label="аналитическая кривая")
    ax.plot([q[0] for q in pts], [q[1] for q in pts], "o",
            color=C[1], ms=5, label="расчёт на кольце")
    ax.set_xlabel(r"Плотность $\rho$, авт/км")
    ax.set_ylabel("Интенсивность $Q$, авт/ч")
    ax.set_xlim(0, 150)
    ax.set_ylim(0, 2000)
    ax.legend()
    fig.tight_layout()
    save(fig, "fig07_fundamental_diagram.png")


# ---------------------------------------------------------------- figure 8
def exp_shock():
    L, END = 3000.0, 300.0
    xs, x = [], 15.0
    while x <= L * 0.85 + 1e-9:
        xs.append(x)
        x += 30.0
    vehicles = [dict(id=f"car_{k}", depart=0, x=x, v=4.0)
                for k, x in enumerate(xs)]
    xt = XT(L, END)
    trig = dict(id=None, t0=None, v0=None, done=False)
    seen, events = set(), []

    def on_state(t, st):
        if t >= END - 1e-9:
            return
        for vid, (x, v) in st.items():
            xt.add(t, x, v)
            if t >= 30.0 and vid not in seen and v < 5.0:
                seen.add(vid)
                events.append((t, x))

    def decide(t, st, veh):
        if trig["id"] is None and not trig["done"] and t >= 30.0 - 1e-9 and st:
            vid = min(st, key=lambda k: abs(st[k][0] - 2250.0))
            trig.update(id=vid, t0=t, v0=st[vid][1])
            veh.setSpeedMode(vid, 0)
            say(
                f"     t={t:.1f} s: braking {vid} (x={st[vid][0]:.0f} m, was {st[vid][1]:.1f} m/s) to 1 m/s in 1 s, hold 25 s")
        if trig["id"] is not None:
            el = t - trig["t0"]
            if el < 26.0 - 1e-9:
                veh.setSpeed(trig["id"], trig["v0"] +
                             (1.0 - trig["v0"]) * min(1.0, (el + DT) / 1.0))
            else:
                veh.setSpeed(trig["id"], -1)
                veh.setSpeedMode(trig["id"], 31)
                trig["id"], trig["done"] = None, True

    run("shock", L, False, vehicles, END, on_state, decide)
    ev = np.array(events)
    if len(ev) > 2:
        say(f"     shock wave speed from trajectories: {np.polyfit(ev[:, 0], ev[:, 1], 1)[0]:.2f} m/s "
            f"({len(ev)} cars), theory -4.29 m/s")
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    mesh = draw_xt(ax, xt)
    ax.set_ylabel("Время, с")
    fig.colorbar(mesh, ax=ax).set_label("скорость, км/ч")
    fig.tight_layout()
    save(fig, "fig08_shock_wave.png")


# ---------------------------------------------------------------- figure 9
def exp_rare():
    L, END, XL, RED = 1500.0, 150.0, 750.0, 40.0
    xs, x = [], XL - 100.0
    while x >= 15.0 - 1e-9:
        xs.append(x)
        x -= 25.0
    vehicles = [dict(id=f"car_{k}", depart=0, x=x, v=12.0)
                for k, x in enumerate(reversed(xs))]
    vehicles += [dict(id=f"in_{k}", depart=3.0 * k,
                      x=0.1, v=12.0) for k in range(40)]
    vehicles.append(dict(id="blk", depart=0, x=XL + P["length"], v=0.0))
    xt = XT(L, END)
    state = dict(removed=False, queue=None, started={})

    def on_state(t, st):
        if t >= END - 1e-9:
            return
        if t >= RED - 1e-9 and state["queue"] is None:
            state["queue"] = {
                k for k, (_, v) in st.items() if k != "blk" and v < 0.5}
        for vid, (x, v) in st.items():
            if vid == "blk":
                continue
            xt.add(t, x, v)
            if state["queue"] and vid in state["queue"] and vid not in state["started"] and v > 1.0:
                state["started"][vid] = (t, x)

    def decide(t, st, veh):
        if not state["removed"] and t >= RED - 1e-9:
            veh.remove("blk")
            state["removed"] = True
        if use_external():
            for vid, v in external_speeds({k: s for k, s in st.items() if k != "blk" or not state["removed"]}).items():
                veh.setSpeed(vid, v)
        if not state["removed"]:
            veh.setSpeed("blk", 0.0)

    run("rare", L, False, vehicles, END, on_state,
        decide, speedmode0=use_external())
    pts = sorted(state["started"].values(), key=lambda q: -q[1])[:15]
    if len(pts) > 2:
        ev = np.array(pts)
        say(
            f"     rarefaction wave speed from start moments of 15 cars: {np.polyfit(ev[:, 0], ev[:, 1], 1)[0]:.2f} m/s, theory -4.35 m/s")
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    mesh = draw_xt(ax, xt, xlim=(0, L))
    ax.set_ylabel("Время, с")
    fig.colorbar(mesh, ax=ax).set_label("скорость, км/ч")
    fig.tight_layout()
    save(fig, "fig09_rarefaction_wave.png")


# ---------------------------------------------------------------- figure 10
class Delayed:
    """IDM driver seeing gap and leader speed as they were tau seconds ago; individual
    tau_i = tau * (1 + 0.3 xi), clipped to [0.3 tau, 2 tau], rounded to whole steps."""

    def __init__(self, tau, seed=1):
        self.tau, self.seed, self.n, self.hist = tau, seed, {}, {}

    def n_delay(self, idx):
        rnd = random.Random(f"human:{self.seed}:{idx}")
        rnd.gauss(0, 1)
        rnd.gauss(0, 1)
        tau = self.tau * min(max(1.0 + 0.3 * rnd.gauss(0, 1), 0.3), 2.0)
        return int(tau / DT + rnd.random())

    def speeds(self, st, L):
        order = sorted(st, key=lambda k: -st[k][0])
        out = {}
        for j, vid in enumerate(order):
            lead = order[j - 1]
            rel = st[lead][0] - st[vid][0] + (L if j == 0 else 0.0)
            if vid not in self.n:
                self.n[vid] = self.n_delay(int(vid.split("_")[1]))
                self.hist[vid] = deque(maxlen=self.n[vid] + 1)
            self.hist[vid].append((rel, st[lead][1]))
            rel_old, vl_old = self.hist[vid][0]
            v = st[vid][1]
            out[vid] = max(v + idm_accel(v, rel_old -
                           P["length"], vl_old) * DT, 0.0)
        return out


def exp_delay():
    L, N, END = 1500.0, 60, 900.0
    Veq = veq_for_density(N / L)
    say(f"     equilibrium speed at {N / L * 1000:.1f} veh/km: {Veq:.2f} m/s")
    res = []
    for tau in (0.0, 0.3):
        xt, tail, dl, flag = XT(L, END), [], (Delayed(
            tau) if tau > 0 else None), dict(on=False)

        def on_state(t, st, xt=xt, tail=tail):
            if t >= END - 1e-9:
                return
            for x, v in st.values():
                xt.add(t, x, v)
            if t >= END - 300.0:
                tail.extend(v for _, v in st.values())

        def decide(t, st, veh, dl=dl, flag=flag):
            pert = 20.0 - 1e-9 <= t < 23.0 - 1e-9
            if dl is None:
                if pert:
                    veh.setSpeedMode("car_5", 0)
                    veh.setSpeed("car_5", 0.7 * Veq)
                    flag["on"] = True
                elif flag["on"]:
                    veh.setSpeed("car_5", -1)
                    veh.setSpeedMode("car_5", 31)
                    flag["on"] = False
            else:
                nv = dl.speeds(st, L)
                if pert:
                    nv["car_5"] = 0.7 * Veq
                for vid, v in nv.items():
                    veh.setSpeed(vid, v)

        run(f"delay_{tau:g}", L, True, [dict(id=f"car_{i}", depart=0, x=(L - i * L / N) % L, v=Veq) for i in range(N)],
            END, on_state, decide, speedmode0=dl is not None)
        say(f"     tau={tau:g} s: speeds in the last 300 s from {min(tail):.1f} to {max(tail):.1f} m/s")
        res.append((tau, xt))
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.8), sharey=True)
    for ax, (tau, xt), lab in zip(axs, res, ("а", "б")):
        mesh = draw_xt(ax, xt, title=f"{lab}) τ = {tau:g} с")
    axs[0].set_ylabel("Время, с")
    fig.colorbar(mesh, ax=axs, fraction=0.03,
                 pad=0.02).set_label("скорость, км/ч")
    save(fig, "fig10_ring_delay.png", bbox_inches="tight")


EXPERIMENTS = dict(eq=exp_eq, free=exp_free, brake=exp_brake,
                   fd=exp_fd, shock=exp_shock, rare=exp_rare, delay=exp_delay)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--only", default="all", help=",".join(EXPERIMENTS))
    ap.add_argument("--backend", choices=["sumo", "python"], default="sumo")
    ap.add_argument("--gui", action="store_true")
    ap.add_argument("--delay-ms", type=int, default=50)
    ap.add_argument("--native", action="store_true")
    ap.add_argument("--libsumo", action="store_true")
    a = ap.parse_args()
    CFG.update(out=os.path.abspath(a.out), backend=a.backend, gui=a.gui,
               delay_ms=a.delay_ms, native=a.native, libsumo=a.libsumo)
    CFG["work"] = os.path.join(CFG["out"], "sumo_files")
    os.makedirs(CFG["work"] if a.backend ==
                "sumo" else CFG["out"], exist_ok=True)
    want = list(EXPERIMENTS) if a.only == "all" else [
        x.strip() for x in a.only.split(",")]
    if set(want) - set(EXPERIMENTS):
        sys.exit(
            f"[ERROR] unknown: {', '.join(set(want) - set(EXPERIMENTS))}; available: {', '.join(EXPERIMENTS)}")
    for name in EXPERIMENTS:
        if name in want:
            say(f"=== {name} ({a.backend})")
            EXPERIMENTS[name]()
    with open(os.path.join(CFG["out"], "summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(SUMMARY) + "\n")


if __name__ == "__main__":
    main()
