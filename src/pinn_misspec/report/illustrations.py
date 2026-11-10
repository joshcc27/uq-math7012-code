"""The illustrative figures, which need no network or saved result.

They share the style of the results figures, and the line styles mean the same
throughout: generating state solid, equation-fit target dashed, solution-fit target
dash-dotted, truth dotted.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar

from . import frame as rf

C1, C2, C3 = rf.PINN, rf.SOLUTION_END, rf.RESIDUAL_END  # blue, vermillion, bluish green
INK, MUTED, GRID = rf.INK, rf.MUTED, rf.GRID

STYLE = {**rf.style(), "lines.linewidth": 1.4}


def grid(ax):
    ax.grid(True, which="major", alpha=0.9, zorder=0)
    ax.set_axisbelow(True)


G_FALL, H0_FALL, T_FALL = 9.81, 50.0, 3.0
BETA_FALL = 0.02  # drag coefficient, 1/m; terminal speed about 22 m/s
N_OBS_FALL, N_COL_FALL = 31, 201


def fall_distance(t, beta, g=G_FALL):
    """Return the distance fallen from rest under h'' = -g + beta h'^2 while descending."""
    if beta == 0:
        return 0.5 * g * t**2
    return np.log(np.cosh(np.sqrt(g * beta) * t)) / beta


def fall_speed(t, beta, g=G_FALL):
    if beta == 0:
        return g * t
    return np.sqrt(g / beta) * np.tanh(np.sqrt(g * beta) * t)


def free_fall_targets(beta, g=G_FALL):
    """Return the solution-fit and equation-fit targets for g under omitted drag.

    The solution-fit target is the least-squares fit of h0 - g t^2 / 2 to the
    noise-free heights at the observation times. The equation-fit target makes
    the free-fall residual h'' + g as small as possible along the generating
    fall at the collocation times, which is the mean downward acceleration
    there. Both problems are linear in g, so both are closed form.
    """
    t_obs = np.linspace(0.0, T_FALL, N_OBS_FALL)
    t_col = np.linspace(0.0, T_FALL, N_COL_FALL)
    distance = fall_distance(t_obs, beta, g)
    solution_fit = 2.0 * np.sum(t_obs**2 * distance) / np.sum(t_obs**4)
    residual = np.mean(g - beta * fall_speed(t_col, beta, g) ** 2)
    return solution_fit, residual


def falling_object(out):
    """Write the falling-object figure and, separately, its two targets."""
    t = np.linspace(0.0, T_FALL, 600)
    t_obs = np.linspace(0.0, T_FALL, N_OBS_FALL)
    g_sol, g_eq = free_fall_targets(BETA_FALL)

    fig, a = plt.subplots(figsize=rf.figure_size(0.56, 2.9))
    a.plot(t, H0_FALL - fall_distance(t, BETA_FALL), color=C1, label="generating, with drag")
    # Hollow markers on top, so the observations show against the curve.
    a.plot(
        t_obs,
        H0_FALL - fall_distance(t_obs, BETA_FALL),
        "o",
        color=MUTED,
        mfc="white",
        mew=0.8,
        ms=3.5,
        zorder=3,
        label="observations",
    )
    a.plot(t, H0_FALL - 0.5 * g_sol * t**2, color=C2, ls="-.", label=r"free fall, fitted $g$")
    a.plot(t, H0_FALL - 0.5 * G_FALL * t**2, color=MUTED, ls=":", label=r"free fall, true $g$")
    a.set_xlabel(r"time $t$ (s)")
    a.set_ylabel(r"height $h(t)$ (m)")
    a.set_xlim(0, T_FALL)
    grid(a)
    fig.legend(
        *a.get_legend_handles_labels(),
        loc="outside lower center",
        ncol=2,
        handlelength=2.2,
        columnspacing=1.0,
    )
    fig.savefig(out / "falling-object.pdf")
    plt.close(fig)

    betas = np.linspace(0.0, 0.05, 101)
    targets = np.array([free_fall_targets(beta) for beta in betas])
    fig, b = plt.subplots(figsize=rf.figure_size(0.56, 2.9))
    b.axhline(G_FALL, color=MUTED, lw=0.9, ls=(0, (1, 2)), label=r"true $g$")
    b.plot(betas, targets[:, 1], color=C3, ls="--", label=r"equation-fit target $g_{\mathrm{eq}}$")
    b.plot(betas, targets[:, 0], color=C2, ls="-.", label=r"solution-fit target $g_{\mathrm{sol}}$")
    # The drag of the falling-object figure; solid, since dotted means the truth.
    b.axvline(BETA_FALL, color=MUTED, lw=0.6, alpha=0.7)
    b.set_xlabel(r"drag coefficient $\beta$ (1/m)")
    b.set_ylabel(r"value of $g$ (m/s$^2$)")
    b.set_xlim(0, betas[-1])
    grid(b)
    # The legend sits below the panel, because every corner holds a curve.
    fig.legend(*b.get_legend_handles_labels(), loc="outside lower center", ncol=1, handlelength=2.2)
    fig.savefig(out / "falling-object-targets.pdf")
    plt.close(fig)
    print(
        f"[check] falling object, beta={BETA_FALL:g}: g_sol={g_sol:.4f} "
        f"({100 * (g_sol / G_FALL - 1):+.2f}%), g_eq={g_eq:.4f} "
        f"({100 * (g_eq / G_FALL - 1):+.2f}%), fall at T={fall_distance(T_FALL, BETA_FALL):.2f} m"
    )


OMEGA, GAMMA_TRUE, T_END = 2.0 * np.pi, 0.30, 3.0
X0, V0 = 1.0, 0.0


def osc(t, gamma, omega=OMEGA, x0=X0, v0=V0):
    """Return the displacement of the damped oscillator in each damping regime."""
    if gamma < omega:  # underdamped
        wd = np.sqrt(omega**2 - gamma**2)
        return np.exp(-gamma * t) * (x0 * np.cos(wd * t) + (v0 + gamma * x0) / wd * np.sin(wd * t))
    if np.isclose(gamma, omega):  # critically damped
        return np.exp(-gamma * t) * (x0 + (v0 + gamma * x0) * t)
    r = np.sqrt(gamma**2 - omega**2)  # overdamped
    return np.exp(-gamma * t) * (x0 * np.cosh(r * t) + (v0 + gamma * x0) / r * np.sinh(r * t))


def oscillator_projection(
    t_obs, truth, gamma_imposed, *, bounds=(0.01, 40.0), grid_size=2001, u0=X0, v0=V0
):
    """Return the best frequency in ``bounds``, and its RMSE, with damping and ICs fixed.

    The same scan-then-refine search as the study's solution fit, on a plain
    NumPy closed form.
    """
    t_obs, truth = np.asarray(t_obs, dtype=float), np.asarray(truth, dtype=float)
    lower, upper = bounds

    def objective(omega):
        return float(np.mean((osc(t_obs, gamma_imposed, omega, u0, v0) - truth) ** 2))

    frequencies = np.linspace(lower, upper, grid_size)
    losses = np.array([objective(omega) for omega in frequencies])
    if not np.all(np.isfinite(losses)):
        raise ValueError("Non-finite trajectory loss on the projection search grid")
    minima = np.flatnonzero((losses[1:-1] <= losses[:-2]) & (losses[1:-1] <= losses[2:])) + 1
    # The best grid point stays a candidate, so refinement never worsens the scan.
    best_index = int(np.argmin(losses))
    candidates = [
        (frequencies[best_index], losses[best_index]),
        (lower, losses[0]),
        (upper, losses[-1]),
    ]
    for index in minima:
        result = minimize_scalar(
            objective,
            bounds=(frequencies[index - 1], frequencies[index + 1]),
            method="bounded",
            options={"xatol": 1e-10},
        )
        if not result.success or not np.isfinite(result.fun):
            raise RuntimeError("Projection candidate refinement failed")
        candidates.append((float(result.x), float(result.fun)))
    frequency, loss = min(candidates, key=lambda candidate: candidate[1])
    return float(frequency), float(np.sqrt(loss))


def oscillator(out):
    """Illustrate the frequency compensating for wrong damping."""
    t_obs = np.linspace(0, T_END, 201)
    truth_obs = osc(t_obs, GAMMA_TRUE)
    t = np.linspace(0, T_END, 1200)  # Smooth rendering, not the fitting design.
    ut = osc(t, GAMMA_TRUE)
    fig, (a, b) = plt.subplots(1, 2, figsize=rf.figure_size(0.94, 2.6))

    gw = 0.90  # the study's largest imposed damping, s = 2
    w_projected, discrepancy = oscillator_projection(t_obs, truth_obs, gw)
    uproj = osc(t, gw, w_projected)
    unaive = osc(t, gw)

    a.plot(t, ut, color=C1, label="generating")
    a.plot(t, uproj, color=C2, ls="--", label=r"imposed, fitted $\omega$")
    a.plot(t, unaive, color=C3, ls=":", label=r"imposed, $\omega^{\dagger}$")
    a.axhline(0, color=MUTED, lw=0.6)
    a.set_xlabel(r"$t$")
    a.set_ylabel(r"$u(t)$")
    a.set_title(r"(a) $\gamma_{s}=0.90$, $s=2$", loc="left")
    a.set_xlim(0, T_END)
    grid(a)

    # Frequency moves the phase but not the amplitude decay, which only damping sets.
    b.plot(t, np.exp(-GAMMA_TRUE * t), color=C1, label=r"generating, $\gamma^{\dagger}$")
    b.plot(t, np.exp(-gw * t), color=C2, ls="--", label=r"imposed, $\gamma_{s}$")
    b.set_xlabel(r"$t$")
    b.set_ylabel(r"$e^{-\gamma t}$")
    b.set_title("(b) amplitude decay", loc="left")
    b.set_xlim(0, T_END)
    b.set_ylim(0, 1.05)
    grid(b)

    # Separate legends: the imposed decay in (b) belongs to both curves of (a).
    for ax in (a, b):
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.3),
            ncol=2,
            handlelength=2.2,
            columnspacing=1.0,
        )
    fig.savefig(out / "osc-misspecification.pdf")
    plt.close(fig)
    print(
        f"[check] fixed-IC projection, gamma={gw:.2f}: omega={w_projected:.6f}, "
        f"d_N={discrepancy:.6f}"
    )


VEL, KAPPA = 1.0, 0.01


def advdiff(x, t, v=VEL, kap=KAPPA, kmax=64):
    """Return the exact periodic solution for u0(x) = exp(-((x - 0.3) / 0.08)^2 / 2)."""
    xg = np.linspace(0, 1, 2048, endpoint=False)
    u0 = np.exp(-0.5 * ((xg - 0.3) / 0.08) ** 2)
    ks = np.arange(-kmax, kmax + 1)
    ck = np.array([np.mean(u0 * np.exp(-2j * np.pi * k * xg)) for k in ks])
    result = np.zeros((len(t), len(x)), dtype=complex)
    for j, tj in enumerate(t):
        amp = ck * np.exp(-kap * (2 * np.pi * ks) ** 2 * tj) * np.exp(-2j * np.pi * ks * v * tj)
        result[j] = np.exp(2j * np.pi * np.outer(x, ks)) @ amp
    return result.real


def advection_diffusion(out):
    """Illustrate the profile and mass mismatch caused by an omitted sink."""
    x = np.linspace(0, 1, 800, endpoint=False)
    final_time, mu = 0.6, 1.2
    initial, imposed_final = advdiff(x, np.array([0.0, final_time]))
    true_final = np.exp(-mu * final_time) * imposed_final
    fig, (a, b) = plt.subplots(1, 2, figsize=rf.figure_size(0.94, 2.6))

    # Same styles as the oscillator figure: generating solid blue, imposed dashed vermilion.
    a.plot(x, true_final, color=C1, label=rf"generating, $\mu={mu:g}$")
    a.plot(x, imposed_final, color=C2, ls="--", label=r"imposed, $\mu=0$")
    a.plot(x, initial, color=MUTED, ls=":", label=r"initial profile, $t=0$")
    a.set_xlabel(r"$x$")
    a.set_ylabel(r"$u(x,t)$")
    a.set_title(rf"(a) profiles at $t={final_time:g}$", loc="left")
    a.set_xlim(0, 1)
    a.set_ylim(0, 1.05)
    grid(a)

    tg = np.linspace(0, final_time, 400)
    b.plot(tg, np.exp(-mu * tg), color=C1)
    b.plot(tg, np.ones_like(tg), color=C2, ls="--")
    b.set_xlabel(r"$t$")
    b.set_ylabel(r"$M(t)/M_0$")
    b.set_title("(b) mass mismatch", loc="left")
    b.set_xlim(0, final_time)
    b.set_ylim(0, 1.05)
    grid(b)

    # One legend below both panels, which share their line styles.
    fig.legend(*a.get_legend_handles_labels(), loc="outside lower center", ncol=3, handlelength=2.2)
    fig.savefig(out / "advdiff-misspecification.pdf")
    plt.close(fig)
    # Mass is conserved on the periodic domain, so this checks the Fourier
    # truncation. The grid excludes its right endpoint, so a mean is the right
    # quadrature.
    m0 = advdiff(x, np.array([0.0]))[0].mean()
    m1 = advdiff(x, np.array([0.6]))[0].mean()
    print(f"[check] advection-diffusion, mass ratio u(.,0.6)/u(.,0) = {m1 / m0:.8f} (should be 1)")


def illustrations(out):
    """Write the four illustrative figures to ``out``; return their file names."""
    with plt.rc_context(STYLE):
        falling_object(out)
        oscillator(out)
        advection_diffusion(out)
    return [
        "falling-object.pdf",
        "falling-object-targets.pdf",
        "osc-misspecification.pdf",
        "advdiff-misspecification.pdf",
    ]
