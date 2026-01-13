import numpy as np
import matplotlib.pyplot as plt

# ==============================
# Physical constants / species
# ==============================
kB = 1.380649e-23         # Boltzmann constant (J/K)
m  = 1.79e-25             # mass of Ag-107 (kg) – change for other species
g  = 9.81                 # gravitational acceleration (m/s^2)
lambda_laser = 1064e-9    # laser wavelength (m)

# ==============================
# Beam parameters
# ==============================
# Beam 1: propagates along +x
P1    = 10.0              # W (only matters if you tie U0 to P; here U0 set directly)
w0_1  = 37e-6             # waist (m)
U0_1  = kB * 5e-6        # on-axis depth in Joules (~50 µK trap depth), adjust as needed

# Beam 2: propagates along +y
P2    = 10.0              # W
w0_2  = 37e-6             # waist (m)
U0_2  = kB * 5e-6        # same depth here, but can differ

# Rayleigh ranges
zR_1 = np.pi * w0_1**2 / lambda_laser
zR_2 = np.pi * w0_2**2 / lambda_laser

# ==============================
# Spatial grid (2D slice: y = 0)
# ==============================
L = 150e-6      # +/- 150 microns
N = 400

x = np.linspace(-L, L, N)
z = np.linspace(-L, L, N)
X, Z = np.meshgrid(x, z)

Y = np.zeros_like(X)   # slice at y = 0

# ==============================
# Beam 1 potential: propagates along x
# ==============================
def beam1_potential(x, y, z):
    r2 = y**2 + z**2
    w = w0_1 * np.sqrt(1 + (x / zR_1)**2)
    return -U0_1 * (w0_1 / w)**2 * np.exp(-2 * r2 / w**2)

# ==============================
# Beam 2 potential: propagates along y
# ==============================
def beam2_potential(x, y, z):
    r2 = x**2 + z**2
    w = w0_2 * np.sqrt(1 + (y / zR_2)**2)
    return -U0_2 * (w0_2 / w)**2 * np.exp(-2 * r2 / w**2)

# ==============================
# Gravity potential
# (+z is upward; gravity is downward)
# ==============================
def gravity_potential(z):
    return m * g * z

# ==============================
# Total potential on the grid
# ==============================
U1 = beam1_potential(X, Y, Z)
U2 = beam2_potential(X, Y, Z)
Ug = gravity_potential(Z)

Utot = U1 + U2 + Ug
Uwo = U1+U2

# ==============================
# Find sagged minimum numerically
# ==============================
min_index = np.unravel_index(np.argmin(Utot), Utot.shape)
z_min = z[min_index[0]]
x_min = x[min_index[1]]

print(f"Trap minimum at x = {x_min*1e6:.2f} µm, z = {z_min*1e6:.2f} µm")

# ==============================
# Plot 2D potential (x–z slice at y=0)
# ==============================
plt.figure(figsize=(7,6))

# Units: x,z in microns; potential in µK
U_plot = Utot / kB * 1e6  # convert J -> μK
U_wo = Uwo / kB * 1e6

im = plt.imshow(
    U_plot,
    extent=[x[0]*1e6, x[-1]*1e6, z[0]*1e6, z[-1]*1e6],
    origin='lower',
    aspect='equal'
)

plt.colorbar(im, label="Potential (µK)")
plt.scatter([x_min*1e6], [z_min*1e6], c='red', marker='x', s=80,
            label="Sagged minimum")


plt.xlabel("x (µm)   (y = 0 slice)")
plt.ylabel("z (µm)")
plt.title("Full Crossed Gaussian ODT + Gravity (no harmonic approximation)")
plt.legend(loc="upper right")
plt.tight_layout()
plt.show()

# ==============================
# Optional: 1D cut along z at x = 0
# ==============================
z_line = z
U_line = Utot[:, N//2] / kB * 1e6
U_line2 = Uwo[:, N//2] / kB * 1e6    # potential at x=0 column (in µK)

plt.figure(figsize=(7,4))
plt.plot(z_line*1e6, U_line, label="U_tot(x=0,y=0,z)")
plt.plot(z_line*1e6, U_line2, label="U_tot(x=0,y=0,z)")
plt.axvline(z_min*1e6, color='red', linestyle='--',
            label=f"Minimum at z = {z_min*1e6:.1f} µm")
plt.xlabel("z (µm)")
plt.ylabel("Potential (µK)")
plt.title("1D Vertical Cut Through Crossed Trap + Gravity")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
