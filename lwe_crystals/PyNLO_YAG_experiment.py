# Imports
##########
import numpy as np
import matplotlib.pyplot as plt
import pynlo


# FUNCTIONS AND CONSTANTS (YAG)
###############################
c = 299792458.0 # m/s

# From Zelmon et al.
B1, C1 = 2.282, 0.01185
B2, C2 = 3.27644, 282.734

# From Hemmer et al.
n2 = 25e-20 #m^2/W

# trying to mimic bulk material behavior so making the effective area
# for the gamma calculation the beams cross sectional area at what would be
# the entrance of the YAG plate.
spot_size = 5e-06 # m
area_eff = np.pi * spot_size**2

def n(wl):
    """
    Refractive index n(lambda) for YAG Sellmeier

    Args:
        wl: The wavelength to compute n at (um)
    returns:
        n: n computes at wl based on the Sellmeier equation (unitless)
    """
    wl2 = wl**2
    n = np.sqrt(1
                + B1*wl2/(wl2 - C1)
                + B2*wl2/(wl2 - C2)
                )
    return n

def wl_to_omega(wl):
    """Convert wavelength (m) to angular frequency (rad/s)"""
    return 2 * np.pi * c / wl

def omega_to_wl(omega):
    """Conver angular frequency (rad/s) to wavelength (m)"""
    return 2 * np.pi * c / omega

def beta_of_omega(omega):
    """Computes the propagation constant beta(omega)"""
    wl = omega_to_wl(omega)
    return n(wl * 1e6) * omega / c


# DEFINE GRID AND CENTER WAVELENGTH
###################################
wl0 = 1e-6 # 3.1 um
omega0 = wl_to_omega(wl0)

omega_range = omega0 * 0.1

N = 1001
omega_grid = omega0 + np.linspace(-omega_range, omega_range, N)
beta_grid = beta_of_omega(omega_grid)


# FINITE DIFFERENCE DERIVATIVES
###############################
d1 = np.gradient(beta_grid, omega_grid)
d2 = np.gradient(d1, omega_grid)
d3 = np.gradient(d2, omega_grid)
d4 = np.gradient(d3, omega_grid)

beta2 = d2[N//2]
beta3 = d3[N//2]
beta4 = d4[N//2]

beta2_ps_km = beta2 * 1e27
beta3_ps_km = beta3 * 1e39
beta4_ps_km = beta4 * 1e51

print("β₂  = " + "{:.3e}".format(beta2) + "s²/m = " + "{:.5f}".format(beta2_ps_km) + "ps^2/km")
print("β₃  = " + "{:.3e}".format(beta3) + "s³/m = " + "{:.5f}".format(beta3_ps_km) + "ps^3/km")
print("β₄  = " + "{:.3e}".format(beta4) + "s⁴/m = " + "{:.5f}".format(beta4_ps_km) + "ps^4/km")


# PULSE PARAMETERS
##################
fwhm = 0.07         # pulse duartion (ps)
pulseWL = 3100      # central wavelength (nm)
pulseEnergy = 3e-06 # energy per pulse (J)
gdd = 0.0           # group delay dispersion (ps^2)
tod = 0.0           # third order dispersion (ps^3)
P0 = (0.94 * pulseEnergy) / (fwhm * 1e-12) # Peak power (J/s)


# GRID PARAMETERS
#################
t_window = 2      # time span (ps)
Nz = 50             # simulation steps (propagation steps)
Nt = 2**13          # simulation points (temporal grid)


# DISPERSION COEFFICIENTS
#########################
beta2 = 70.3        # beta 2 (ps^2/km)
beta3 = 0.07        # beta 3 (ps^3/km)
beta4 = 0           # beta 4 (ps^4/km)


# FIBER PARAMETERS
##################
length = 3.0        # fiber length (mm)
alpha = 0.0         # attenuation coefficient (dB/cm)
gamma = ((n2 * wl_to_omega(pulseWL * 1e-09)) / (c * area_eff)) * 1e3    # Effective nonlinear coefficient (1/(W km))
fibWL = pulseWL
Raman = True
Steep = True


# set up plots for the results:
fig = plt.figure(figsize=(10,10))
ax0 = plt.subplot2grid((3,2), (0, 0), rowspan=1)
ax1 = plt.subplot2grid((3,2), (0, 1), rowspan=1)
ax2 = plt.subplot2grid((3,2), (1, 0), rowspan=2, sharex=ax0)
ax3 = plt.subplot2grid((3,2), (1, 1), rowspan=2, sharex=ax1)


# CREATE PULSE
##############
pulse = pynlo.light.DerivedPulses.GaussianPulse(P0, fwhm/1.76, pulseWL, time_window_ps = t_window,
                                                GDD=gdd, TOD=tod, NPTS=Nt, frep_MHz=100, power_is_avg=False)
pulse.set_epp(pulseEnergy)


# CREATE FIBER
##############
fiber = pynlo.media.fibers.fiber.FiberInstance()
fiber.generate_fiber(length * 1e-3, center_wl_nm=fibWL, betas=(beta2, beta3, beta4), gamma_W_m=gamma*1e-3,
                     gvd_units="ps^n/km", gain=-alpha)


# PROPAGATE
###########
evol = pynlo.interactions.FourWaveMixing.SSFM.SSFM(local_error=0.001, USE_SIMPLE_RAMAN=True,
                                                   disable_Raman=np.logical_not(Raman),
                                                   disable_self_steepening=np.logical_not(Steep))
y, AW, AT, pulse_out = evol.propagate(pulse_in=pulse, fiber=fiber, n_steps=Nz)


F = pulse.W_mks / (2 * np.pi) * 1e-12 # convert to THz

def dB(num):
    return 10 * np.log10(np.abs(num)**2)

zW = dB( np.transpose(AW)[:, (F > 0)] )
zT = dB( np.transpose(AT) )

y = y * 1e3 # convert distance to mm


ax0.plot(F[F > 0],  zW[-1], color='r')
ax1.plot(pulse.T_ps,zT[-1], color='r')

ax0.plot(F[F > 0],   zW[0], color='b')
ax1.plot(pulse.T_ps, zT[0], color='b')


extent = (np.min(F[F > 0]), np.max(F[F > 0]), 0, length)
ax2.imshow(zW, extent=extent, vmin=np.max(zW) - 60.0,
                 vmax=np.max(zW), aspect='auto', origin='lower')

extent = (np.min(pulse.T_ps), np.max(pulse.T_ps), np.min(y), length)
ax3.imshow(zT, extent=extent, vmin=np.max(zT) - 60.0,
           vmax=np.max(zT), aspect='auto', origin='lower')


ax0.set_ylabel('Intensity (dB)')

ax2.set_xlabel('Frequency (THz)')
ax3.set_xlabel('Time (ps)')

ax2.set_ylabel('Propagation distance (mm)')

ax2.set_xlim(0,400)

ax0.set_ylim(-80,0)
ax1.set_ylim(-40,40)

plt.show()