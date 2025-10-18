import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from io import BytesIO


# -------------------------------------
#       Equations de la chaleur
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
#              Heatmaps
# -------------------------------------
def heatmap_gif_1d(x, T, dt, frame_every=2, duration=0.08, cmap="jet"):
    frames = []
    vmin, vmax = float(np.min(T)), float(np.max(T))
    for k in range(0, T.shape[0], frame_every):
        fig, ax = plt.subplots(figsize=(5, 5), dpi=120)
        band = T[k, :][np.newaxis, :]
        im = ax.imshow(band, origin="lower", aspect="auto",
                       extent=[x[0], x[-1], 0, 1], vmin=vmin, vmax=vmax, cmap=cmap)
        ax.set_xlabel("Position x (m)"); ax.set_yticks([])
        ax.set_title(f"t ≈ {k*dt:.3f} s")
        plt.colorbar(im, ax=ax, label="Température (°C)")
        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(frame); plt.close(fig)
    buf = BytesIO(); imageio.mimsave(buf, frames, format="gif", duration=duration)
    return buf.getvalue()

def heatmap_gif_2d(Tstack, dt_between_frames, duration=0.08, cmap="jet"):
    frames = []
    vmin, vmax = float(np.min(Tstack)), float(np.max(Tstack))
    for i, Tk in enumerate(Tstack):
        fig, ax = plt.subplots(figsize=(5.0, 5.0), dpi=120)
        im = ax.imshow(Tk, origin="lower", vmin=vmin, vmax=vmax, cmap=cmap)
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_title(f"t ≈ {i*dt_between_frames:.3f} s")
        plt.colorbar(im, ax=ax, label="Température (°C)")
        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(frame); plt.close(fig)
    buf = BytesIO(); imageio.mimsave(buf, frames, format="gif", duration=duration)
    return buf.getvalue()


# -------------------------------------
#            Page Streamlit
# -------------------------------------
st.set_page_config(page_title="TP1 ADELL - MRAD", layout="wide")
l,m,r = st.columns((1,8,1))
m.title("Diffusion de la chaleur en 1D et en 2D")
st.markdown("<br>"*2, unsafe_allow_html=True)

with st.sidebar:
    st.header("🔧 Paramètres physiques")
    longueur = st.slider(
        "Longueur L (m)", 5.0, 50.0, 20.0, 0.5,
        help=(
            "Plus L est grand, plus la chaleur mettra de temps à se diffuser d’un bord à l’autre. "
        ))
    alpha = st.slider(
        "Diffusivité α (m²/s)", 0.1, 10.0, 1.0, 0.1,
        help=(
            "Coefficient de diffusivité thermique du matériau : α = λ / (ρ·c). "
            "C'est égal à la vitesse à laquelle la chaleur se propage dans la matière. "
        ))
    st.header("🔧 Paramètres numériques")
    delta_x = st.slider(
        "Pas spatial Δx (m)", 0.1, 2.0, 1.0, 0.1,
        help=(
            "Distance entre deux points du maillage spatial. "
            "Un Δx plus petit signifie un maillage plus fin, donc une simulation plus précise mais plus lente. "
            "Physiquement, cela revient à observer la température en plus de points le long de la barre."
        ))
    dt_default = 0.5 * (delta_x**2) 
    delta_t = st.slider(
        "Pas de temps Δt (s)", 0.001, float(2.0 * dt_default), float(dt_default), 0.001,
        help=(
            "Plus Δt est petit, plus la simulation avance lentement mais reste stable et précise. "
            "Un Δt trop grand peut rendre la simulation instable. "
            "Δt correspond à la durée entre deux observations de l’évolution thermique."
        ))
    max_iter = st.slider(
        "Durée totale (nombre d’itérations)", 20, 300, 100, 10,
        help=(
            "Plus ce nombre est grand, plus la simulation dure longtemps. "
            "Cela correspond au temps total simulé."
        ))
    st.header("🔥 Températures (°C)")
    t_initial = st.slider(
        "Température initiale (°C)", 0.0, 1000.0, 300.0, 10.0,
        help=(
            "Température de départ uniforme . "
        ))
    t_left = st.slider(
        "Température au bord gauche (°C)", 0.0, 1000.0, 600.0, 10.0,
        help=(
            "Température imposée à l’extrémité gauche (x=0). "
        ))
    t_right = st.slider(
        "Température au bord droit (°C)", 0.0, 1000.0, 200.0, 10.0,
        help=(
            "Température imposée à l’extrémité droite (x=L). "
            "Une différence entre T_left et T_right crée un gradient thermique, "
            "donc un transfert de chaleur de la zone chaude vers la zone froide."
        ))

n_x = int(longueur / delta_x) + 1
x = np.linspace(0, longueur, n_x)
T1 = np.zeros((max_iter, n_x))
T1.fill(t_initial)
T1[:, 0].fill(t_left)
T1[:, -1].fill(t_right)

n = int(longueur / delta_x) + 1

# Calculs
T1 = calculate_1d(T1, n_x, max_iter, alpha, delta_t, delta_x)
snap_every = 2
T2_stack, dt2 = solve_heat_2d_stable(
    n=n, dx=delta_x, alpha=alpha, dt=delta_t, n_steps=max_iter,
    T_top=t_left, T_bottom=t_right, T_left=t_left, T_right=t_right,
    T_init=t_initial, snapshot_every=snap_every
)

FRAME_EVERY_1D, DURATION = 2, 0.08
gif1d = heatmap_gif_1d(x, T1, delta_t, frame_every=FRAME_EVERY_1D, duration=DURATION)
gif2d = heatmap_gif_2d(T2_stack, dt_between_frames=dt2*snap_every, duration=DURATION)

left, right = st.columns((1, 1))
with left:
    st.subheader("1D :")
    st.image(gif1d)
    st.download_button("Télécharger GIF 1D", data=gif1d, file_name="diffusion_1D.gif", mime="image/gif")
    st.markdown("<br>"*2, unsafe_allow_html=True)
    st.subheader("Équation de la chaleur en 1D :")
    st.write("""
            ρ · c · (∂T/∂t) = λ · (∂²T/∂x²)

            - ρ : masse volumique du matériau (kg/m³)  
            - c : capacité thermique massique (J/kg·K)  
            - λ : conductivité thermique (W/m·K)  
            """)

with right:
    st.subheader("2D :")
    st.image(gif2d)
    st.download_button("Télécharger GIF 2D", data=gif2d, file_name="diffusion_2D.gif", mime="image/gif")
    st.markdown("<br>"*2, unsafe_allow_html=True)
    st.subheader("Équation de la chaleur en 2D :")
    st.write("""
            ρ · c · (∂T/∂t) = λ · [(∂²T/∂x²) + (∂²T/∂y²)]

            - ρ : masse volumique du matériau (kg/m³)  
            - c : capacité thermique massique (J/kg·K)  
            - λ : conductivité thermique (W/m·K)  
            """)
