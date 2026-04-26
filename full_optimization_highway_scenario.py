import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from matplotlib import animation
import matplotlib.patches as patches
from matplotlib.transforms import Affine2D
import os

OUT_DIR = "results_highway"
os.makedirs(OUT_DIR, exist_ok=True)

DT = 0.1

X0     = np.array([0.0, 5.0, 0.0, 0.0])
X_GOAL = np.array([9.0, 5.0, 0.0, 0.0])

LANES_Y = [3.0, 5.0, 7.0]
LANE_WEIGHT = 2.0

OBSTACLES = [(4.0, 3.0), (5.5, 5.0), (7.0, 7.0)]
D_SAFE = 0.8

CAR_LENGTH = 0.8
CAR_WIDTH  = 0.4

N_STEPS = 100

# ==============================

# state: [x, y, v, theta]
# control: [a, omega]
# ==============================
Q = np.diag([10.0, 10.0, 8.0, 2.0])      # running state cost
R = np.diag([0.5,  1.0])                  # control effort cost
S = np.diag([100.0, 100.0, 200.0, 20.0]) # terminal state cost

# obstacle penalty weight
W_obs  = 1000.0
# lane penalty weight
W_lane = LANE_WEIGHT

# ==============================
# RK4 DYNAMICS
# ==============================
def dynamics(x, u):
    X, Y, v, th = x
    a, w = u
    return np.array([
        v * np.cos(th),
        v * np.sin(th),
        a,
        w
    ])

def rk4_step(x, u, dt):
    k1 = dynamics(x, u)
    k2 = dynamics(x + dt/2 * k1, u)
    k3 = dynamics(x + dt/2 * k2, u)
    k4 = dynamics(x + dt * k3, u)
    return x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)


def make_initial_guess():
    dist     = np.hypot(X_GOAL[0] - X0[0], X_GOAL[1] - X0[1])
    v_cruise = dist / (N_STEPS * DT)

    u_guess = np.zeros((N_STEPS, 2))

    for k in range(N_STEPS):
        ramp_steps = max(1, int(0.1 * N_STEPS))
        if k < ramp_steps:
            a = v_cruise / (ramp_steps * DT)
        elif k >= N_STEPS - ramp_steps:
            a = -v_cruise / (ramp_steps * DT)
        else:
            a = 0.0
        u_guess[k] = [a, 0.0]

    return u_guess.flatten()


def total_cost(u_flat):
    u_seq = u_flat.reshape(N_STEPS, 2)
    x = X0.copy()

    total = 0.0

    for k in range(N_STEPS):
        u  = u_seq[k]
        ex = x - X_GOAL
        ex[3] = (ex[3] + np.pi) % (2 * np.pi) - np.pi   # wrap angle

        # running cost:  eᵀQe + uᵀRu
        total += ex @ Q @ ex + u @ R @ u

        # lane keeping penalty
        d_lane = min([abs(x[1] - ly) for ly in LANES_Y])
        total += W_lane * d_lane**2

        # obstacle avoidance penalty
        for (ox, oy) in OBSTACLES:
            d = np.hypot(x[0] - ox, x[1] - oy)
            if d < D_SAFE + 0.5:
                total += W_obs * (D_SAFE + 0.5 - d)**2

        x = rk4_step(x, u, DT)

    # terminal cost:  eᵀSe
    ex = x - X_GOAL
    ex[3] = (ex[3] + np.pi) % (2 * np.pi) - np.pi
    total += ex @ S @ ex

    return total

# ==============================
# SOLVE
# ==============================
def solve():
    u_init = make_initial_guess()
    bounds = [(-3, 3), (-2, 2)] * N_STEPS

    print("Running SLSQP optimisation...")
    res = minimize(
        total_cost,
        u_init,
        method='SLSQP',
        bounds=bounds,
        options={'maxiter': 500, 'ftol': 1e-6}
    )
    print(f"Done. Cost: {res.fun:.4f}  Success: {res.success}")

    u_opt = res.x.reshape(N_STEPS, 2)

    x = X0.copy()
    traj = [x.copy()]
    controls = []

    for k in range(N_STEPS):
        u = u_opt[k]
        x = rk4_step(x, u, DT)
        traj.append(x.copy())
        controls.append(u.copy())

        if np.linalg.norm(x[:2] - X_GOAL[:2]) < 0.1:
            break

    return np.array(traj), np.array(controls)

# ==============================
# DRAW HIGHWAY
# ==============================
def draw_scene(ax):
    # road edges
    ax.plot([0, 10], [2, 2], 'k', linewidth=3)
    ax.plot([0, 10], [8, 8], 'k', linewidth=3)

    # lanes
    for ly in LANES_Y:
        ax.plot([0, 10], [ly, ly], '--', color='gray')

    # obstacles
    for (ox, oy) in OBSTACLES:
        rect = patches.Rectangle(
            (ox - CAR_LENGTH/2, oy - CAR_WIDTH/2),
            CAR_LENGTH, CAR_WIDTH,
            color='blue'
        )
        ax.add_patch(rect)

        circle = plt.Circle((ox, oy), D_SAFE, color='orange', alpha=0.3)
        ax.add_patch(circle)

# ==============================
# PLOTS
# ==============================
def plot_states(traj):
    t = np.arange(len(traj)) * DT

    fig, axs = plt.subplots(2, 2, figsize=(10, 6))

    axs[0, 0].plot(t, traj[:, 0])
    axs[0, 0].set_title("X Position")
    axs[0, 0].set_xlabel("Time [s]")
    axs[0, 0].set_ylabel("X")
    axs[0, 0].grid()

    axs[0, 1].plot(t, traj[:, 1])
    axs[0, 1].set_title("Y Position")
    axs[0, 1].set_xlabel("Time [s]")
    axs[0, 1].set_ylabel("Y")
    axs[0, 1].grid()

    axs[1, 0].plot(t, traj[:, 2])
    axs[1, 0].set_title("Velocity")
    axs[1, 0].set_xlabel("Time [s]")
    axs[1, 0].set_ylabel("v")
    axs[1, 0].grid()

    axs[1, 1].plot(t, traj[:, 3])
    axs[1, 1].set_title("Heading (theta)")
    axs[1, 1].set_xlabel("Time [s]")
    axs[1, 1].set_ylabel("θ")
    axs[1, 1].grid()

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "states.png"))
    plt.show()

def plot_controls(controls):
    t = np.arange(len(controls)) * DT
    plt.figure()
    plt.plot(t, controls[:, 0], label='a')
    plt.plot(t, controls[:, 1], '--', label='omega')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(OUT_DIR, "controls.png"))
    plt.show()

# ==============================
# ANIMATION
# ==============================
def animate(traj):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(2, 8)
    ax.set_aspect('equal')
    ax.grid()

    draw_scene(ax)

    line, = ax.plot([], [], 'g-', lw=2)

    car = patches.Rectangle(
        (-CAR_LENGTH/2, -CAR_WIDTH/2),
        CAR_LENGTH, CAR_WIDTH,
        color='red'
    )
    ax.add_patch(car)

    def update(i):
        x, y, th = traj[i, 0], traj[i, 1], traj[i, 3]
        line.set_data(traj[:i, 0], traj[:i, 1])

        trans = Affine2D().rotate(th).translate(x, y)
        car.set_transform(trans + ax.transData)

        return line, car

    ani = animation.FuncAnimation(fig, update,
                                  frames=len(traj),
                                  interval=100)

    ani.save(os.path.join(OUT_DIR, "trajectory.gif"), writer='pillow')
    plt.close()

# ==============================
# MAIN
# ==============================
def main():
    traj, controls = solve()

    fig, ax = plt.subplots(figsize=(7, 5))
    draw_scene(ax)

    ax.plot(traj[:, 0], traj[:, 1], 'g', label='SLSQP+RK4')

    ax.scatter(*X0[:2], c='red', zorder=5)
    ax.scatter(*X_GOAL[:2], c='green', zorder=5)

    ax.legend()
    ax.set_xlim(0, 10)
    ax.set_ylim(2, 8)
    ax.grid()

    plt.savefig(os.path.join(OUT_DIR, "trajectory.png"))
    plt.show()

    plot_states(traj)
    plot_controls(controls)
    animate(traj)

if __name__ == "__main__":
    main()
