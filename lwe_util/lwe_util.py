import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from lwe import LightwaveExplorer as lwe
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert

def import_data(sim_object: lwe.lightwaveExplorerResult, data_keys: dict = None) -> dict:
    """
    Import data from a LightwaveExplorer result based on requested keys
    
    Args:
        sim_object: LightwaveExplorer result object
        data_keys: Dictionary with keys representing what data to import, e.g. {"Ex": True, "Ey": False}
                   If None, imports all available data
    
    Returns:
        Dictionary containing the requested data
    """
    # Check object type
    if not isinstance(sim_object, lwe.lightwaveExplorerResult):
        raise TypeError(f"import_data expected a LightwaveExplorer, "
                        f"got {type(sim_object).__name__!r}")
    
    # If no keys specified, import everything
    if data_keys is None:
        data_keys = {
            "Ex": True, 
            "Ey": True, 
            "freq": True, 
            "spectrum_x": True,
            "spectrum_y": True, 
            "time": True
        }
    
    data = {}
    if data_keys.get("Ex", False):
        data["Ex"] = sim_object.Ext_x
    if data_keys.get("Ey", False):
        data["Ey"] = sim_object.Ext_y
    if data_keys.get("freq", False):
        data["freq"] = sim_object.frequencyVectorSpectrum
    if data_keys.get("spectrum_x", False):
        data["spectrum_x"] = sim_object.spectrum_x
    if data_keys.get("spectrum_y", False):
        data["spectrum_y"] = sim_object.spectrum_y
    if data_keys.get("time", False):
        data["time"] = sim_object.timeVector
        
    return data

def radially_average(E_field: np.ndarray, radius: float, spatial_step: float, timeVector: list = []) -> tuple:
    """
        This function is made to radially average the E field data out to a certain radius. Lightwave explorer returns E field data in the form
    of a two dimensional array where each column represent the E field data across the entire time axis for some point on the transverse grid. 

    Args:
        E_field: This should be the 2D array containing the E field data of interest
        radius: The radius to average out to (must be smaller than transverse grid) (unit: m)
        spatial_step: the step size of the transverse grid (unit: m)
        timeVecotor: optional argument, pass if you want to graph the radially averaged field

    return:
        tuple: a tuple containing the radially averaged E field data and the enevlope of the pulse
    """

    # Determine whether grid has an even or odd amount of steps
    Nx = E_field.shape[1]

    if (Nx % 2) == 0:
        even_grid = True
    else:
        even_grid = False


    # Determine whether the radius is too large
    grid_size = Nx * spatial_step
    if radius >= grid_size:
        raise ValueError(f"radius is too large, grid size: {grid_size}, radius: {radius}")
    

    # Calculate amount of radial steps such that the averaging is symmetric and as close to the 
    # target radius as possible
    radial_steps = radius / spatial_step
    if even_grid:
        if (radial_steps % 2) == 0:
            pass

        if (radial_steps % 2) >= 1:
            mod = radial_steps % 2
            diff = 2 - mod
            radial_steps = radial_steps + diff

        if (radial_steps % 2) < 1:
            mod = radial_steps % 2
            radial_steps = radial_steps - mod
    else:
        if (radial_steps % 2) == 1:
            pass

        else:
            mod = radial_steps % 2
            radial_steps = radial_steps + 1 - mod


    # If rounding of radial steps increased the radius passed the grid size subtract 2 steps
    if (radial_steps * spatial_step) >= grid_size:
        radial_steps = radial_steps - 2

    
    # Determine which indices to average over
    if even_grid:
        i = (Nx / 2) - 1
        index1 = i - (radial_steps / 2 - 1)
        index2 = i + 1 + (radial_steps / 2 - 1)
    else:
        i = (Nx / 2) - 0.5
        index1 = i - ((radial_steps - 1) / 2)
        index2 = i + ((radial_steps - 1) / 2)

    
    # Compute radial average
    average = np.mean(E_field[:,int(index1):int(index2)+1], axis = 1) # last index exclusive
    envelope = np.abs(hilbert(average))

    if len(timeVector):
        plt.figure(figsize=(10, 5))
        plt.plot(timeVector * 1e15, average * 1e-9, color='blue', label=r'Radially Averaged $E_{x}$')
        plt.plot(timeVector * 1e15, envelope * 1e-9, color='red', linestyle='--', label=r'Radially Averaged $E_{x}$ Envelope')
        plt.xlabel('Time (fs)')
        plt.ylabel(r'$E_{x}$ (GV/m)')
        plt.title(r'$E_{x}(t)$ with Envelope')
        plt.legend()
        plt.grid()
        plt.show()
        print(f"averaged symmetrically over {radial_steps * spatial_step} meters")

    return (average, envelope)

# TODO: Make this function more accurate 
def calculate_pulse_energy(E_field: np.ndarray, dr: float, dt: float, rad: float = 0) -> float:
    """
    This function calculate the energy of a radially symmetric guassian pulse. The data that you pass to this function should be a matrix
    containing Ex(r,t) where each column is some radius and the center of the transverse grid is the middle index (LWE does this by default)
    
    Args:
        E_field: Matrix conatining E field data, each column is should be some r, the center of the transverse grid (origin) should be the
        center index.
        dr: transverse grid spatial step in meters
        dt: time step
        rad: radius to restrict integration to, does entire grid by default 
    """

    # Define relavant constants 
    epsilon0 = 8.854e-12    # F/m
    c = 3e8                 # m/s

    # Determine grid parameters
    Nt, Nx = E_field.shape
    if (Nx % 2) == 0:
        even_grid = True
    else:
        even_grid = False

    # Determine whether the radius is too large
    grid_size = Nx * dr
    if rad >= grid_size:
        raise ValueError(f"radius is too large, grid size: {grid_size}, radius: {rad}")


    # Compute pulse energies
    if not rad: 

        if even_grid:
            # Compute radial area weights
            Nr = int(Nx / 2)
            i = np.arange(1, Nr+1, 1)
            r = (i - 0.5) * dr
            dA = 2 * np.pi * r * dr

            # Compute intensity as a function of t
            E_radial = E_field[:, Nr:2*Nr]
            I_time = np.sum(np.abs(E_radial)**2 * dA[None, :], axis = 1)

        else:
            # Compute radial area weights
            Nr = int( (Nx/2) + 0.5)
            i = np.arange(1, Nr+1, 1)
            r = (i - 0.5) * dr
            dA = 2 * np.pi * r * dr
        
            # Compute intensity as a function of t
            E_radial = E_field[:, Nr-1:2*Nr-1]
            I_time = np.sum(np.abs(E_radial)**2 * dA[None, :], axis = 1)

        # sum over time
        pulse_energy = epsilon0 * c * np.trapz(I_time, dx=dt)
        
    else:
        # Calculate energy up to specified radius
        pulse_energy = 0

    
    return pulse_energy

def map_beam_radius(beam_waist: float, rayleigh_length: float, max_z: float, precision: float, verbose: bool = True) -> tuple:
    """
    Calculates the beam waist of the described guassian beam for some extent of z. returns a list of these
    values and has the option to plot the function.
    
    Args:
        beam_waist: guassian beam waist, most narrow radius of beam (m)
        rayleigh_length: desired rayleigh length of beam (m)
        max_z: extent to which one wants their array created to (m)
        precision: what spatial increment to compute the beam radius at, starts at 0 (m)
        verbose: whether to graph the results or not

    returns:
        tuple: A tuple containing the z_axis used for computation as well as the beam radius values
    
    """
    if not (max_z > precision):
        raise ValueError(f"Maximum Z value must be greater than the desired precision")
    
    z_axis = np.arange(0.0, max_z, precision)
    beam_radius = np.sqrt( beam_waist**2 * (1 + (z_axis / rayleigh_length)**2) )

    if verbose:
        plt.figure(figsize=(10,5))
        plt.plot(z_axis, beam_radius, label=r'Beam Radius')
        plt.xlabel("Z Position (m)")
        plt.ylabel("Beam Radius (m)")
        plt.title(" Beam Radius vs Z ")
        plt.grid()
        if max_z > rayleigh_length:
            plt.axvline(x=rayleigh_length, color='red', linestyle='--', label='$Z_{R}$')
        plt.legend()
        plt.show()

    return (z_axis, beam_radius)

def map_peak_intensity(beam_waist: float, rayleigh_length: float, max_z: float, precision: float,
                       pulse_duration: float, pulse_energy: float, verbose: bool = True) -> tuple:
    """
    Calculates the beam waist of the described guassian beam for some extent of z. Then uses the beam waist calculation to calculate the
    peak intensities along the propogation direction.
    
    Args:
        beam_waist: guassian beam waist, most narrow radius of beam (m)
        rayleigh_length: desired rayleigh length of beam (m)
        max_z: extent to which one wants their array created to (m)
        precision: what spatial increment to compute the beam radius at, starts at 0 (m)
        verbose: whether to graph the results or not

    returns:
        tuple: A tuple containing the z_axis used for computation as well as the peak intensities along this axis
    
    """

    beam_radius = map_beam_radius(beam_waist, rayleigh_length, max_z, precision, verbose=False)
    peak_intensities = (1.88 * pulse_energy) / (pulse_duration * np.pi * beam_radius[1]**2)

    if verbose:
        plt.figure(figsize=(10,5))
        plt.plot(beam_radius[0], peak_intensities * 1e-16, label=r'Peak Intensity')
        plt.xlabel("Z Position (m)")
        plt.ylabel("Peak Intensity (TW/cm^2)")
        plt.title("Peak Intensity vs Z")
        plt.grid()
        if max_z > rayleigh_length:
            plt.axvline(x=rayleigh_length, color='red', linestyle='--', label='$Z_{R}$')
        plt.legend()
        plt.show()

    return (beam_radius[0], peak_intensities)

def calculate_pulse_duration(E_field: np.ndarray, radius: float, spatial_step: float, timeVector: list, verbose: bool = True) -> float:
    """
    Calculates the pulse duration of some pulse using a radially averged E field.

    Args:
        E_field: Electrc field matrix each column is some transverse position, on axis field should be center index
        radius: radius to average out to (m)
        spatial_step: transverse grid step (m)
        timeVector: the time vector used in the simulation (s)
        verbose: optional argument, determines whether to graph the averaged E fields or not
    returns:
        pulse_duration: the FWHM of the averaged E fields temporal intensity distribution (s)
    
    """
    E_avg = radially_average(E_field, radius, spatial_step)

    pulse_duration = lwe.fwhm(timeVector, E_avg[1]**2)

    if verbose:
        plt.figure(figsize=(10, 5))
        plt.plot(timeVector * 1e15, (E_avg[0] * 1e-9)**2, color='blue', label=r'$E_{avg}(t)^2$')
        plt.plot(timeVector * 1e15, (E_avg[1] * 1e-9)**2, color='red', label='Envelope', linestyle='--')
        peak_y = (E_avg[1] * 1e-9).max()**2
        plt.text(20, peak_y - (0.2 * peak_y), f'FWHM: {pulse_duration * 1e15:.2f} fs',
                fontsize=12, color='red',
                bbox=dict(facecolor='white', edgecolor='black', alpha=0.8))
        plt.xlabel('Time (fs)')
        plt.ylabel(r'$E_{avg}(t)^2$ $(GV^2/m^2)$')
        plt.title(r'$E_{avg}(t)^2$ with Envelope')
        plt.legend()
        plt.grid()
        plt.show()

    return pulse_duration

#TODO: Finish this function. 
def calculate_beam_radius(E_field: np.ndarray, spatial_step: float, intensity_drop: float = 0.5) -> float:
    """
    Calculates the half intensity radius of an E_field matrix. The % of intensity drop off used for the radius calculation can be changed

    Args:
        E_field: E field matrix [time, x] (V/m)
        spatial_step: spatial step of transverse grid (m)
        intensity drop: decimal representing percent drop to be used for radius calculation
    
    returns:
        radius: beam radius at the desired intensity drop off
    """

    I = np.abs(E_field)**2
    t0, j0 = np.unravel_index(np.argmax(I), I.shape)
    I_prof = I[t0, :]
    I_half = 0.5 * I_prof.max()
