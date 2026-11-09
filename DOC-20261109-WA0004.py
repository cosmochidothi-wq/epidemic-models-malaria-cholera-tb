"""
Compartmental models of malaria, cholera and tuberculosis in Python.

Author: Cosmo Chidothi
Tools: NumPy, SciPy (solve_ivp), Matplotlib

Models
------
1. Malaria  - Ross-Macdonald model (humans SIS, mosquitoes SI)
2. Cholera  - SIR-B model (humans SIR + free-living Vibrio cholerae in water)
3. TB       - SLIR-type model (Susceptible, Latent, Infectious, Recovered/treated)

Each model is run twice: a baseline and an intervention scenario
(bed nets, water/sanitation, better TB treatment).

IMPORTANT: parameter values are illustrative, chosen from typical ranges in the
literature to demonstrate the method. They are NOT fitted to Malawian data.
"""
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------- Malaria
def malaria_params(m=0.5):
    return dict(
        m=m,      # mosquitoes per human
        a=0.25,   # bites per mosquito per day
        b=0.30,   # prob. infection mosquito -> human per bite
        c=0.30,   # prob. infection human -> mosquito per bite
        r=0.01,   # human recovery rate (1/day, ~100 days infectious)
        mu=0.10,  # mosquito death rate (1/day, ~10 day lifespan)
    )


def malaria_rhs(t, y, p):
    Ih, Im = y  # fraction of infected humans / mosquitoes
    dIh = p["m"] * p["a"] * p["b"] * Im * (1 - Ih) - p["r"] * Ih
    dIm = p["a"] * p["c"] * Ih * (1 - Im) - p["mu"] * Im
    return [dIh, dIm]


def malaria_R0(p):
    # Threshold quantity (squared form): > 1 means malaria persists
    return p["m"] * p["a"] ** 2 * p["b"] * p["c"] / (p["r"] * p["mu"])


# ---------------------------------------------------------------- Cholera
def cholera_params(beta_w=0.15):
    return dict(
        beta_w=beta_w,  # water-borne transmission rate (1/day)
        beta_h=0.05,    # direct person-to-person transmission (1/day)
        K=1e6,          # Vibrio concentration giving 50% infection chance
        gamma=0.2,      # recovery rate (1/day, ~5 days)
        xi=10.0,        # shedding rate per infected person into water
        delta=0.33,     # bacterial decay rate in water (1/day)
        mu=1 / (60 * 365),  # birth/death rate (1/day)
        N=100_000,      # community size
    )


def cholera_rhs(t, y, p):
    S, I, R, B = y
    N = p["N"]
    force = p["beta_w"] * B / (p["K"] + B) + p["beta_h"] * I / N
    dS = p["mu"] * N - force * S - p["mu"] * S
    dI = force * S - (p["gamma"] + p["mu"]) * I
    dR = p["gamma"] * I - p["mu"] * R
    dB = p["xi"] * I - p["delta"] * B
    return [dS, dI, dR, dB]


def cholera_R0(p):
    # Early-epidemic threshold, S ~ N and B small
    return (p["beta_w"] * p["xi"] * p["N"] / (p["K"] * p["delta"]) + p["beta_h"]) / (
        p["gamma"] + p["mu"]
    )


# ---------------------------------------------------------------------- TB
def tb_params(gamma=0.5):
    return dict(
        beta=1.5,      # transmission rate per infectious person per year
        sigma=0.05,    # progression latent -> active (1/yr)
        gamma=gamma,   # treatment/recovery rate (1/yr)
        mu=0.02,       # natural birth/death rate (1/yr)
        mu_tb=0.10,    # TB-induced death rate (1/yr)
        rho=0.9,       # fraction of recovered who stay protected (rest relapse to latent)
    )


def tb_rhs(t, y, p):
    S, L, I, R = y
    N = S + L + I + R
    lam = p["beta"] * I / N
    dS = p["mu"] * N - lam * S - p["mu"] * S
    dL = lam * S + (1 - p["rho"]) * p["gamma"] * I - (p["sigma"] + p["mu"]) * L
    dI = p["sigma"] * L - (p["gamma"] + p["mu"] + p["mu_tb"]) * I
    dR = p["rho"] * p["gamma"] * I - p["mu"] * R
    return [dS, dL, dI, dR]


def tb_R0(p):
    # Includes relapse: a fraction (1 - rho) of treated cases return to the
    # latent class, so they can become infectious again.
    prog = p["sigma"] / (p["sigma"] + p["mu"])  # chance latent -> active
    leave_I = p["gamma"] + p["mu"] + p["mu_tb"] - (1 - p["rho"]) * p["gamma"] * prog
    return p["beta"] * prog / leave_I


# ------------------------------------------------------------------ driver
def run(rhs, y0, p, t_end, n=1000):
    t_eval = np.linspace(0, t_end, n)
    sol = solve_ivp(rhs, (0, t_end), y0, args=(p,), t_eval=t_eval,
                    method="LSODA", rtol=1e-8, atol=1e-10)
    return sol.t, sol.y


def main():
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    summary = []

    # ---- Malaria (time in days -> plot in days)
    base, ctrl = malaria_params(0.5), malaria_params(0.2)  # nets/spraying cut m by 60%
    t, yb = run(malaria_rhs, [0.01, 0.0], base, 1500)
    _, yc = run(malaria_rhs, [0.01, 0.0], ctrl, 1500)
    ax = axes[0]
    ax.plot(t, yb[0] * 100, lw=2, label=f"Baseline (R0={malaria_R0(base):.2f})")
    ax.plot(t, yc[0] * 100, lw=2, ls="--",
            label=f"Bed nets + spraying (R0={malaria_R0(ctrl):.2f})")
    ax.set(title="Malaria: Ross-Macdonald model", xlabel="Days",
           ylabel="Infected humans (%)")
    ax.legend()
    summary.append(("Malaria", malaria_R0(base), malaria_R0(ctrl),
                    f"Endemic infected humans: {yb[0][-1]*100:.1f}% -> {yc[0][-1]*100:.1f}%"))

    # ---- Cholera (days)
    base, ctrl = cholera_params(0.15), cholera_params(0.04)  # safe water + sanitation
    N = base["N"]
    y0 = [N - 10, 10, 0, 0]
    t, yb = run(cholera_rhs, y0, base, 300)
    _, yc = run(cholera_rhs, y0, ctrl, 300)
    ax = axes[1]
    ax.plot(t, yb[1], lw=2, label=f"Baseline (R0={cholera_R0(base):.2f})")
    ax.plot(t, yc[1], lw=2, ls="--",
            label=f"Safe water + sanitation (R0={cholera_R0(ctrl):.2f})")
    ax.set(title="Cholera: SIR-B model", xlabel="Days", ylabel="Currently infected people")
    ax.legend()
    summary.append(("Cholera", cholera_R0(base), cholera_R0(ctrl),
                    f"Peak infected: {yb[1].max():,.0f} -> {yc[1].max():,.0f} "
                    f"of {N:,} people"))

    # ---- TB (years)
    base, ctrl = tb_params(0.5), tb_params(1.0)  # faster diagnosis & treatment
    y0 = [0.989, 0.01, 0.001, 0.0]
    t, yb = run(tb_rhs, y0, base, 100)
    _, yc = run(tb_rhs, y0, ctrl, 100)
    ax = axes[2]
    ax.plot(t, yb[2] * 1e5, lw=2, label=f"Baseline (R0={tb_R0(base):.2f})")
    ax.plot(t, yc[2] * 1e5, lw=2, ls="--",
            label=f"Doubled treatment rate (R0={tb_R0(ctrl):.2f})")
    ax.set(title="Tuberculosis: SLIR model", xlabel="Years",
           ylabel="Active TB per 100,000")
    ax.legend()
    summary.append(("TB", tb_R0(base), tb_R0(ctrl),
                    f"Active TB per 100,000 at year 100: {yb[2][-1]*1e5:,.0f} -> {yc[2][-1]*1e5:,.0f}"))

    for ax in axes:
        ax.grid(alpha=0.3)
    fig.suptitle("Illustrative disease models (parameters are not fitted to Malawian data)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig("disease_models.png", dpi=200)

    print(f"{'Disease':<10}{'R0 base':>9}{'R0 control':>12}  Result")
    for name, r0b, r0c, text in summary:
        print(f"{name:<10}{r0b:>9.2f}{r0c:>12.2f}  {text}")


if __name__ == "__main__":
    main()
