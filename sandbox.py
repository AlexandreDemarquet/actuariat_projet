import numpy as np
import matplotlib.pyplot as plt

# Paramètres du modèle
T = 100  # Temps maximal
M = 5   # Hauteur maximale du rectangle
mu = 2  # Intensité de la partie normale
alpha = 0.5  # Intensité de la fonction de Hawkes
beta = 0.5   # Décroissance de la fonction de Hawkes

# Fonction Phi pour le processus auto-excitant
def Phi(u, alpha, beta):
    return alpha * np.exp(-beta * u) * (u >= 0)

# Étape 1 : Génération des points de base
n_points = np.random.poisson(M * T)  # Nombre total de points dans le rectangle
times = np.sort(np.random.uniform(0, T, n_points))  # Temps (ordonnés)
marks = np.random.uniform(0, M, n_points)  # Marques (θ)

# Étape 2 : Calcul de N_t^(1)
N1_times = times[marks <= mu]  # On garde les points où θ ≤ μ
N1 = len(N1_times)  # Nombre d'événements dans la partie normale

# Étape 3 : Calcul de N_t^(2)
N2 = 0
N2_times = []

for i1, t1 in enumerate(times):
    for i2, t2 in enumerate(times[i1 + 1:], start=i1 + 1):  # t2 > t1
        if mu < marks[i2] <= mu + Phi(t2 - t1, alpha, beta):
            N2 += 1
            N2_times.append(t2)

# Étape 4 : Combiner N1 et N2
N_total = N1 + N2
N_total_times = np.sort(np.concatenate([N1_times, N2_times]))

# # Visualisation des résultats
# plt.figure(figsize=(10, 6))
# plt.scatter(times, marks, label="Points simulés", alpha=0.5, color="grey")
# plt.scatter(N1_times, [mu] * len(N1_times), label="N_t^(1) (normale)", color="blue")
# plt.scatter(N2_times, [mu + 0.5] * len(N2_times), label="N_t^(2) (auto-excitant)", color="red")
# plt.axhline(mu, color="black", linestyle="--", label="Seuil μ")
# plt.axhline(mu + alpha, color="green", linestyle="--", label="Seuil μ + Φ(0)")
# plt.xlabel("Temps")
# plt.ylabel("Marques (θ)")
# plt.title("Simulation d'un processus auto-excitant (Hawkes)")
# plt.legend()
# plt.show()

# Afficher les résultats
# Paramètres pour R_t
u = 50  # Richesse initiale
c = 5   # Revenu constant par unité de temps
lambda_y = 1  # Paramètre de la loi exponentielle pour les tailles de sinistres

# Simulation des tailles des sinistres (Yi)
losses_cl = np.random.exponential(1 / lambda_y, n_points)

# Calcul de R_t dans le modèle classique
time_points_cl = np.linspace(0, T, 1000)  # Temps pour évaluer R_t
# Étape 1 : Calculer R_t pour N_t^(1) uniquement
losses_N1 = np.random.exponential(1 / lambda_y, len(N1_times))  # Tailles des sinistres pour N_t^(1)
R_t_N1 = u + c * time_points_cl  # Initialisation de R_t pour N_t^(1)

# Soustraire les sinistres pour N_t^(1)
for i, t_i in enumerate(N1_times):
    R_t_N1[time_points_cl >= t_i] -= losses_N1[i]

# Étape 2 : Calculer R_t pour N_t = N_t^(1) + N_t^(2)
all_times = np.sort(np.concatenate([N1_times, N2_times]))  # Combiner les temps des deux parties
losses_N_total = np.random.exponential(1 / lambda_y, len(all_times))  # Tailles des sinistres totaux
R_t_N_total = u + c * time_points_cl  # Initialisation de R_t pour N_t

# Soustraire les sinistres pour N_t total
for i, t_i in enumerate(all_times):
    R_t_N_total[time_points_cl >= t_i] -= losses_N_total[i]

# Visualisation de R_t pour N_t^(1) et N_t total
plt.figure(figsize=(12, 6))
plt.plot(time_points_cl, R_t_N1, label="Richesse $R_t$ avec $N_t^{(1)}$ (normale)", color="blue")
plt.plot(time_points_cl, R_t_N_total, label="Richesse $R_t$ avec $N_t$ (normale + auto-excitant)", color="red")
plt.axhline(0, color="black", linestyle="--", label="Seuil critique ($R_t = 0$)")
plt.xlabel("Temps")
plt.ylabel("Richesse $R_t$")
plt.title("Richesse $R_t$ avec $N_t^{(1)}$ et $N_t^{(1)} + N_t^{(2)}$")
plt.legend()
plt.show()


