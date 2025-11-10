import streamlit as st
import numpy as np
import matplotlib
matplotlib.use("Agg")  # backend sans interface graphique pour l’hébergement
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import plotly.graph_objects as go
from io import BytesIO

# -------------------------------------
# Utilitaire : conversion figure -> ndarray (robuste Cloud)
# -------------------------------------
def _fig_to_ndarray(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=fig.dpi)
    buf.seek(0)
    arr = imageio.imread(buf)
    buf.close()
    return arr

# -------------------------------------
#       Équations de la chaleur
# -------------------------------------

# --- 1D ---
def calculate_1d(T, n_x, n_steps, alpha, dt, dx):
    r = alpha * dt / (dx**2)
    for k in range(n_steps - 1):
        Ti = T[k, :].copy()
        T[k + 1, 1:-1] = Ti[1:-1] + r * (Ti[2:] + Ti[:-2] - 2.0 * Ti[1:-1])
    return T

# --- 2D ---
def solve_heat_2d_stable(n, dx, alpha, dt, n_steps,
                         T_top, T_bottom, T_left, T_right, T_init,
                         snapshot_every=2):
    dt_max = 0.25 * dx * dx / alpha
    if dt > dt_max:
        dt = dt_max * 0.98
    T = np.full((n, n), T_init, dtype=float)
    T[0, :], T[-1, :], T[:, 0], T[:, -1] = T_bottom, T_top, T_left, T_right
    snaps = []
    r = alpha * dt / (dx * dx)
    for k in range(n_steps):
        Ti = T.copy()
        T[1:-1, 1:-1] = (
            Ti[1:-1, 1:-1]
            + r * (Ti[1:-1, 2:] + Ti[1:-1, :-2] + Ti[2:, 1:-1] + Ti[:-2, 1:-1] - 4.0 * Ti[1:-1, 1:-1])
        )
        T[0, :], T[-1, :], T[:, 0], T[:, -1] = T_bottom, T_top, T_left, T_right
        if (k % snapshot_every == 0) or (k == n_steps - 1):
            snaps.append(T.copy())
    return np.stack(snaps, axis=0), dt


# -------------------------------------
#              Courbes
# -------------------------------------
def courbe_1d(x, T, dt):
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=x, y=T[-1, :], mode="lines", line=dict(color="red", width=3)))
    fig1.update_layout(
        xaxis_title="Longueur L (m)",
        yaxis_title="Température (°C)",
        template="simple_white"
    )
    return fig1

def courbe_2d(x, Tstack):
    Tfinal = Tstack[-1]
    n = Tfinal.shape[0]
    mid = n // 2
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=x, y=Tfinal[mid, :], mode="lines", name="Courbe sur le plan X", line=dict(color="red", width=3)))
    fig2.add_trace(go.Scatter(x=x, y=Tfinal[:, mid], mode="lines", name="Courbe sur le plan Y ", line=dict(color="orange", width=3)))
    fig2.update_layout(
        xaxis_title="Longueur L (m)",
        yaxis_title="Température (°C)",
        template="simple_white"
    )
    return fig2

# -------------------------------------
#              Heatmaps
# -------------------------------------
def heatmap_gif_1d(x, T, dt, frame_every=2, duration=0.08, cmap="jet"):
    frames = []
    vmin, vmax = float(np.min(T)), float(np.max(T))
    for k in range(0, T.shape[0], frame_every):
        fig, ax = plt.subplots(figsize=(5.0, 5.0), dpi=120)
        band = T[k, :][np.newaxis, :]
        im = ax.imshow(
            band, origin="lower", aspect="auto",
            extent=[x[0], x[-1], 0, 1], vmin=vmin, vmax=vmax, cmap=cmap
        )
        ax.set_xlabel("Position x (m)")
        ax.set_yticks([])
        ax.set_title(f"t ≈ {k*dt:.3f} s")
        plt.colorbar(im, ax=ax, label="Température (°C)")
        fig.tight_layout()
        frame = _fig_to_ndarray(fig)
        frames.append(frame)
        plt.close(fig)
    buf = BytesIO()
    imageio.mimsave(buf, frames, format="gif", duration=duration)
    return buf.getvalue()

def heatmap_gif_2d(Tstack, dt_between_frames, duration=0.08, cmap="jet"):
    frames = []
    vmin, vmax = float(np.min(Tstack)), float(np.max(Tstack))
    for i, Tk in enumerate(Tstack):
        fig, ax = plt.subplots(figsize=(5.0, 5.0), dpi=120)
        im = ax.imshow(Tk, origin="lower", vmin=vmin, vmax=vmax, cmap=cmap, aspect="auto")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"t ≈ {i*dt_between_frames:.3f} s")
        plt.colorbar(im, ax=ax, label="Température (°C)")
        fig.tight_layout()
        frame = _fig_to_ndarray(fig)
        frames.append(frame)
        plt.close(fig)
    buf = BytesIO()
    imageio.mimsave(buf, frames, format="gif", duration=duration)
    return buf.getvalue()

# -------------------------------------
#            Page Streamlit
# -------------------------------------
st.set_page_config(page_title="TP1 ADELL - MRAD", layout="wide")
l, m, r = st.columns((1, 8, 1))
m.title("Diffusion de la chaleur en 1D et en 2D")
st.markdown("<br>" * 2, unsafe_allow_html=True)

with st.sidebar:
    st.header("🔧 Paramètres physiques")
    longueur = st.slider(
        "Longueur L (m)", 5.0, 50.0, 20.0, 0.5,
        help=("Plus L est grand, plus la chaleur mettra de temps à se diffuser d’un bord à l’autre. ")
    )
    alpha = st.slider(
        "Diffusivité α (m²/s)", 0.1, 10.0, 1.0, 0.1,
        help=(
            "Coefficient de diffusivité thermique du matériau : α = λ / (ρ·c). "
            "Il quantifie la vitesse à laquelle la chaleur se propage dans la matière."
        ))
    st.header("🔧 Paramètres numériques")
    delta_x = st.slider(
        "Pas spatial Δx (m)", 0.1, 2.0, 1.0, 0.1,
        help=(
            "Distance entre deux points du maillage spatial. "
            "Un Δx plus petit équivaut à un maillage plus fin soit une simulation plus précise mais plus coûteuse. "
            "Physiquement, on observe la température en plus de points le long de la barre."
        ))
    dt_default = 0.5 * (delta_x**2)
    delta_t = st.slider(
        "Pas de temps Δt (s)", 0.001, float(2.0 * dt_default), float(dt_default), 0.001,
        help=(
            "Durée d’un pas de calcul. "
            "Plus Δt est petit, plus l’évolution est douce entre deux images. "
            "Δt correspond à la durée entre deux mises à jour de la température."
        ))
    max_iter = st.slider(
        "Durée totale (nombre d’itérations)", 20, 300, 100, 10,
        help=("Nombre total d’étapes de calcul dans le temps (temps total simulé)."))
    st.header("🔥 Températures (°C)")
    t_initial = st.slider(
        "Température initiale (°C)", 0.0, 1000.0, 300.0, 10.0,
        help=("Température de départ uniforme dans tout le domaine."))
    t_left = st.slider(
        "Température au bord gauche (°C)", 0.0, 1000.0, 600.0, 10.0,
        help=("Température imposée à l’extrémité gauche (x=0)."))
    t_right = st.slider(
        "Température au bord droit (°C)", 0.0, 1000.0, 200.0, 10.0,
        help=(
            "Température imposée à l’extrémité droite (x=L). "
            "La différence entre les deux bords crée un gradient thermique "
            "et donc un flux de chaleur du chaud vers le froid."
        ))

# 1D
n_x = int(longueur / delta_x) + 1
x = np.linspace(0, longueur, n_x)
T1 = np.zeros((max_iter, n_x))
T1.fill(t_initial)
T1[:, 0].fill(t_left)
T1[:, -1].fill(t_right)

# 2D
n = int(longueur / delta_x) + 1

# Calculs
T1 = calculate_1d(T1, n_x, max_iter, alpha, delta_t, delta_x)
snap_every = 2
T2_stack, dt2 = solve_heat_2d_stable(
    n=n, dx=delta_x, alpha=alpha, dt=delta_t, n_steps=max_iter,
    T_top=t_left, T_bottom=t_right, T_left=t_left, T_right=t_right,
    T_init=t_initial, snapshot_every=snap_every
)

# GIFs
FRAME_EVERY_1D, DURATION = 2, 0.08
gif1d = heatmap_gif_1d(x, T1, delta_t, frame_every=FRAME_EVERY_1D, duration=DURATION)
gif2d = heatmap_gif_2d(T2_stack, dt_between_frames=dt2 * snap_every, duration=DURATION)

# Affichage
left, right = st.columns((1, 1))
left.subheader("1D :")
left.image(gif1d)
left.download_button("Télécharger GIF 1D", data=gif1d, file_name="diffusion_1D.gif", mime="image/gif")
left.markdown("<br>" * 2, unsafe_allow_html=True)
left.subheader(" Équation de la chaleur en 1D :")
left.plotly_chart(courbe_1d(x, T1, delta_t), use_container_width=True)
left.markdown("<br>" * 1, unsafe_allow_html=True)
left.write(
    "ρ · c · (∂T/∂t) = λ · (∂²T/∂x²)\n\n"
    "- ρ : masse volumique du matériau (kg/m³)\n"
    "- c : capacité thermique massique (J/kg·K)\n"
    "- λ : conductivité thermique (W/m·K)\n"
    ": (∂T/∂t) = α · (∂²T/∂x²) avec α = λ / (ρ·c)"
)
right.subheader("2D :")
right.image(gif2d)
right.download_button("Télécharger GIF 2D", data=gif2d, file_name="diffusion_2D.gif", mime="image/gif")
right.markdown("<br>" * 2, unsafe_allow_html=True)
right.subheader(" Équation de la chaleur en 2D :")
right.plotly_chart(courbe_2d(x, T2_stack), use_container_width=True)
right.markdown("<br>" * 1, unsafe_allow_html=True)
right.write(
    "ρ · c · (∂T/∂t) = λ · [(∂²T/∂x²) + (∂²T/∂y²)]\n\n"
    "- ρ : masse volumique du matériau (kg/m³)\n"
    "- c : capacité thermique massique (J/kg·K)\n"
    "- λ : conductivité thermique (W/m·K)\n"
    ": (∂T/∂t) = α · [(∂²T/∂x²) + (∂²T/∂y²)] avec α = λ / (ρ·c)"
)

