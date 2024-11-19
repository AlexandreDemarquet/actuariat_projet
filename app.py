import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Fonction pour générer Phi
def Phi(u, alpha, beta):
    return alpha * np.exp(-beta * u) * (u >= 0)

# Fonction pour calculer R_t
def calculate_Rt(times, losses, u, c, T):
    time_points = np.linspace(0, T, 1000)
    R_t = u + c * time_points  # Richesse initiale + revenu constant
    for i, t_i in enumerate(times):
        R_t[time_points >= t_i] -= losses[i]
    return time_points, R_t

# Paramètres interactifs avec Streamlit
st.title("Simulation de Richesse $R_t$ dans le modèle Auto-Excitant")
T = st.sidebar.slider("Durée maximale (T)", 1, 20, 10)
M = st.sidebar.slider("Hauteur maximale (M)", 1, 10, 5)
mu = st.sidebar.slider("Intensité normale (μ)", 0.1, 5.0, 2.0, 0.1)
alpha = st.sidebar.slider("Paramètre α (auto-excitation)", 0.1, 5.0, 1.5, 0.1)
beta = st.sidebar.slider("Paramètre β (décroissance)", 0.1, 2.0, 0.5, 0.1)
u = st.sidebar.slider("Richesse initiale (u)", 10, 100, 50)
c = st.sidebar.slider("Revenu constant (c)", 1, 10, 5)
lambda_y = st.sidebar.slider("Paramètre de sinistres (λ)", 0.1, 5.0, 1.0)

# Étape 1 : Génération de N_t^(1)
n_points = np.random.poisson(M * T)
times = np.sort(np.random.uniform(0, T, n_points))
marks = np.random.uniform(0, M, n_points)

# Partie normale
N1_times = times[marks <= mu]
losses_N1 = np.random.exponential(1 / lambda_y, len(N1_times))

# Étape 2 : Génération de N_t^(2)
N2_times = []
for i1, t1 in enumerate(times):
    for i2, t2 in enumerate(times[i1 + 1:], start=i1 + 1):
        if mu < marks[i2] <= mu + Phi(t2 - t1, alpha, beta):
            N2_times.append(t2)
N2_times = np.array(N2_times)
losses_N2 = np.random.exponential(1 / lambda_y, len(N2_times))

# Calcul de R_t^(1) et R_t total
time_points, R_t_N1 = calculate_Rt(N1_times, losses_N1, u, c, T)
_, R_t_N_total = calculate_Rt(np.sort(np.concatenate([N1_times, N2_times])),
                              np.concatenate([losses_N1, losses_N2]), u, c, T)

# Affichage des résultats
st.subheader("Graphiques de $R_t$")
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(time_points, R_t_N1, label="$R_t$ avec $N_t^{(1)}$ (normale)", color="blue")
ax.plot(time_points, R_t_N_total, label="$R_t$ avec $N_t$ (total)", color="red")
ax.axhline(0, color="black", linestyle="--", label="Seuil critique ($R_t = 0$)")
ax.set_xlabel("Temps")
ax.set_ylabel("Richesse $R_t$")
ax.set_title("Évolution de la richesse $R_t$")
ax.legend()
st.pyplot(fig)
