import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
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

OBSTACLES = [(4.0,3.0),(5.5,5.0),(7.0,7.0)]
D_SAFE = 0.8

CAR_LENGTH = 0.8
CAR_WIDTH  = 0.4

GRID  = 80
WORLD = 10.0
SCALE = WORLD / GRID

# ==============================
# COST MAP
# ==============================
def build_cost_map():
    C = np.ones((GRID, GRID))
    alpha = 5.0

    for i in range(GRID):
        for j in range(GRID):
            x = i * SCALE
            y = j * SCALE

            d_lane = min([abs(y - ly) for ly in LANES_Y])
            C[j,i] += LANE_WEIGHT * d_lane**2

            for (ox, oy) in OBSTACLES:
                d = np.hypot(x - ox, y - oy)

                if d < D_SAFE:
                    C[j,i] = 1000.0
                elif d < D_SAFE + 0.5:
                    C[j,i] += alpha * (D_SAFE + 0.5 - d)**2

    return C

# ==============================
# DP
# ==============================
def dp_plan(start, goal, C):
    V = np.full_like(C, np.inf)
    policy = {}

    V[goal[1], goal[0]] = 0.0
    moves = [(1,0),(1,1),(1,-1),(0,1),(0,-1)]

    for _ in range(300):
        for i in range(GRID):
            for j in range(GRID):
                if (i,j)==goal: continue
                best = np.inf
                best_move = (0,0)

                for dx,dy in moves:
                    ni,nj = i+dx, j+dy
                    if 0<=ni<GRID and 0<=nj<GRID:
                        val = C[nj,ni] + V[nj,ni]
                        if val < best:
                            best = val
                            best_move = (dx,dy)

                V[j,i] = best
                policy[(i,j)] = best_move

    path = [start]
    cur = start
    for _ in range(300):
        if cur==goal: break
        dx,dy = policy[cur]
        cur = (cur[0]+dx, cur[1]+dy)
        path.append(cur)

    return np.array(path)

# ==============================
# SMOOTH PATH
# ==============================
def smooth_path(path):
    pts = path * SCALE
    t = np.arange(len(pts))
    t_new = np.linspace(0, len(pts)-1, len(pts)*5)

    fx = interp1d(t, pts[:,0], kind='cubic')
    fy = interp1d(t, pts[:,1], kind='cubic')

    return np.column_stack([fx(t_new), fy(t_new)])

# ==============================
# REFERENCE
# ==============================
def build_reference(traj):
    ref = []
    N = len(traj)

    for i in range(N-2):
        x,y = traj[i]
        xn,yn = traj[i+1]

        dx,dy = xn-x, yn-y
        theta = np.arctan2(dy,dx)

        v = np.clip(np.hypot(dx,dy)/DT, 0.5, 2.0)

        if i > N - 25:
            scale = (N - i) / 25.0
            v *= scale**2

        theta2 = np.arctan2(traj[i+2][1]-yn, traj[i+2][0]-xn)
        omega = (theta2 - theta)/DT

        ref.append([x,y,v,theta,omega])

    ref.append(ref[-1])
    ref[-1][2] = 0.0

    return np.array(ref)

# ==============================
# DYNAMICS + LINEARIZE
# ==============================
def f(x,u):
    X,Y,v,th = x
    a,w = u

    return np.array([
        X + v*np.cos(th)*DT,
        Y + v*np.sin(th)*DT,
        v + a*DT,
        th + w*DT
    ])

def linearize(x):
    _,_,v,th = x

    A = np.array([
        [1,0,np.cos(th)*DT,-v*np.sin(th)*DT],
        [0,1,np.sin(th)*DT, v*np.cos(th)*DT],
        [0,0,1,0],
        [0,0,0,1]
    ])

    B = np.array([
        [0,0],
        [0,0],
        [DT,0],
        [0,DT]
    ])

    return A,B

# ==============================
# LQR
# ==============================
def tvlqr(A_list, B_list, Q,R,S):
    P = S.copy()
    K_list = []

    for k in reversed(range(len(A_list))):
        A = A_list[k]
        B = B_list[k]

        K = np.linalg.inv(R + B.T@P@B) @ (B.T@P@A)
        K_list.insert(0, K)

        P = Q + A.T@P@A - A.T@P@B@K

    return K_list

# ==============================
# SIMULATION (ONE-SHOT)
# ==============================
def simulate(ref):
    Q = np.diag([10,10,8,2])
    R = np.diag([0.5,1])
    S = np.diag([100,100,200,20])

    A_list, B_list = [], []
    for i in range(len(ref)):
        A,B = linearize(ref[i,:4])
        A_list.append(A)
        B_list.append(B)

    K_list = tvlqr(A_list,B_list,Q,R,S)

    x = X0.copy()
    traj = [x.copy()]
    controls = []

    for k in range(len(K_list)):
        x_ref = ref[k,:4]

        e = x - x_ref
        e[3] = (e[3]+np.pi)%(2*np.pi)-np.pi

        u = -K_list[k] @ e
        u[0] = np.clip(u[0], -3, 3)
        u[1] = np.clip(u[1], -2, 2)

        x = f(x,u)

        traj.append(x.copy())
        controls.append(u.copy())

        if np.linalg.norm(x[:2]-X_GOAL[:2]) < 0.1:
            break

    return np.array(traj), np.array(controls)

# ==============================
# DRAW HIGHWAY
# ==============================
def draw_scene(ax):
    # road edges
    ax.plot([0,10],[2,2],'k',linewidth=3)
    ax.plot([0,10],[8,8],'k',linewidth=3)

    # lanes
    for ly in LANES_Y:
        ax.plot([0,10],[ly,ly],'--',color='gray')

    # obstacles
    for (ox,oy) in OBSTACLES:
        rect = patches.Rectangle(
            (ox-CAR_LENGTH/2, oy-CAR_WIDTH/2),
            CAR_LENGTH, CAR_WIDTH,
            color='blue'
        )
        ax.add_patch(rect)

        circle = plt.Circle((ox,oy), D_SAFE,
                            color='orange', alpha=0.3)
        ax.add_patch(circle)

# ==============================
# PLOTS
# ==============================
def plot_states(traj):
    t = np.arange(len(traj)) * DT

    fig, axs = plt.subplots(2, 2, figsize=(10,6))

    # X position
    axs[0,0].plot(t, traj[:,0])
    axs[0,0].set_title("X Position")
    axs[0,0].set_xlabel("Time [s]")
    axs[0,0].set_ylabel("X")
    axs[0,0].grid()

    # Y position
    axs[0,1].plot(t, traj[:,1])
    axs[0,1].set_title("Y Position")
    axs[0,1].set_xlabel("Time [s]")
    axs[0,1].set_ylabel("Y")
    axs[0,1].grid()

    # Velocity
    axs[1,0].plot(t, traj[:,2])
    axs[1,0].set_title("Velocity")
    axs[1,0].set_xlabel("Time [s]")
    axs[1,0].set_ylabel("v")
    axs[1,0].grid()

    # Heading
    axs[1,1].plot(t, traj[:,3])
    axs[1,1].set_title("Heading (theta)")
    axs[1,1].set_xlabel("Time [s]")
    axs[1,1].set_ylabel("θ")
    axs[1,1].grid()

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "states.png"))
    plt.show()

def plot_controls(controls):
    t = np.arange(len(controls))*DT
    plt.figure()
    plt.plot(t, controls[:,0], label='a')
    plt.plot(t, controls[:,1],'--', label='omega')
    plt.legend(); plt.grid()
    plt.savefig(os.path.join(OUT_DIR,"controls.png"))
    plt.show()

# ==============================
# ANIMATION
# ==============================
def animate(traj):
    fig, ax = plt.subplots(figsize=(7,5))
    ax.set_xlim(0,10)
    ax.set_ylim(2,8)
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
        x,y,th = traj[i,0], traj[i,1], traj[i,3]
        line.set_data(traj[:i,0], traj[:i,1])

        trans = Affine2D().rotate(th).translate(x,y)
        car.set_transform(trans + ax.transData)

        return line, car

    ani = animation.FuncAnimation(fig, update,
                                  frames=len(traj),
                                  interval=30)

    ani.save(os.path.join(OUT_DIR,"trajectory.gif"),
             writer='pillow')
    plt.close()

# ==============================
# MAIN
# ==============================
def main():
    C = build_cost_map()

    start = (int(X0[0]/SCALE), int(X0[1]/SCALE))
    goal  = (int(X_GOAL[0]/SCALE), int(X_GOAL[1]/SCALE))

    path = dp_plan(start, goal, C)
    traj_smooth = smooth_path(path)
    ref = build_reference(traj_smooth)

    traj, controls = simulate(ref)

    fig, ax = plt.subplots(figsize=(7,5))
    draw_scene(ax)

    ax.plot(traj_smooth[:,0], traj_smooth[:,1],'b--',label='DP')
    ax.plot(traj[:,0], traj[:,1],'g',label='LQR')

    ax.scatter(*X0[:2], c='red')
    ax.scatter(*X_GOAL[:2], c='green')

    ax.legend()
    ax.set_xlim(0,10)
    ax.set_ylim(2,8)
    ax.grid()

    plt.savefig(os.path.join(OUT_DIR,"trajectory.png"))
    plt.show()

    plot_states(traj)
    plot_controls(controls)
    animate(traj)

if __name__ == "__main__":
    main()