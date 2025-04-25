import numpy as np

# GLOBAL CONSTANTS
INEV_TO_PC = 0.197e-18 * 1e9/3.086e13   # eV^-1 to parsecs
EV_TO_SOLAR = (1.67e-27/2e30)*1e-9      # 1 eV in solar masses
SEC_TO_INEV = 1e-9/6.528e-25          # 1 second in eV^-1
AVG_VEL_DM = 1e-3 # Average velocity of galactic Dark Matter
YEAR_TO_SEC = 3.154e7 # number of seconds in 1 year
DAY_TO_SEC = 60*60*24 # number of seconds in 1 day
SPEED_OF_LIGHT = 3e8 # speed of light in m/s
GCM3_TO_EV4 = 4.2e18 # g/cm^3 to eV^4
PLANCK_MASS_EV = 1.2e28 # Planck mass in eV
GPC_TO_PC = 1e9 # gigaparsec to parsecs
PC_TO_METERS = 3.086e16 # parsecs to meters

# Densities of ISM and IGM converted from g/cm^3 to eV^4
RHO_ISM_GCM3 = 1.67e-24
RHO_ISM = RHO_ISM_GCM3 * GCM3_TO_EV4
RHO_IGM_GCM3 = 1.67e-30
RHO_IGM = RHO_IGM_GCM3 * GCM3_TO_EV4

PI = np.pi 

def signal_duration(Etot, mass, energies, burst_duration, distance_pc, aw, integration_time=1): 
    """ Calculate the duration of the signal 
    
    Args:
        Etot (float): total energy of emitted phi particles from the source [eV]
        mass (float): mass of phi field [eV]
        energies (array_like): array of individual particle energies [eV]
        burst_duration (float): t_star, the duration of the burst emission at the source [s]
        distance_pc (float): distance between source and detector [pc]
        aw (float): wavepacket uncertainty parameter based on uncertainty principle:
                    dw * t_star >= 1, where dw = spread in energies and t_star = burst_duration
                    for a given wavepacket, aw = dw * t_star, where aw >= 1
                    aw = 1 corresponds to a Gaussian wavepacket, since it has minimum uncertainty
        integration_time (float): integration time [days], default = 1 day

    Returns:
        rho (float) - energy density of phi at Earth [eV^4]
        coherence (list of float) - coherence time values for each energy [unitless]
    """ 
    # Convert integration time to seconds and then to inverse eV
    integration_time_s = integration_time * DAY_TO_SEC
    t_int = np.full_like(energies, integration_time_s * SEC_TO_INEV)
    
    # Set integration time for dark matter experiment
    integration_time_DM_s = 1e6
    t_int_DM = np.full_like(energies, integration_time_DM_s * SEC_TO_INEV)
    
    # Inverse-square law (1/4*pi*R^2) with a unit conversion
    rm2 = 1/(4*PI*(distance_pc/INEV_TO_PC)**2)
    
    # Spread in energies from uncertainty principle: dw * t_star = aw -> dw = aw / t_star
    dw = aw/(burst_duration*SEC_TO_INEV)
    
    # Physical size of the phi wave at the source, assuming it's traveling at c = 1.
    dx_burst = burst_duration*SEC_TO_INEV 
    
    # Spread of the phi wave during propagation
    dx_spread = (dw/energies)*(mass**2/energies**2)*(distance_pc/INEV_TO_PC)
    
    # Total spread of the wavepacket
    dx = dx_burst + dx_spread
    
    # Energy density of phi at Earth
    rho = (Etot/EV_TO_SOLAR)*rm2/dx

    # Calculate t_star_tilde, the signal duration at Earth
    burst_duration = burst_duration*SEC_TO_INEV
    signal_duration = np.sqrt(burst_duration**2 + dx_spread**2)
    
    # Calculate coherence times
    # tau_star = 2*pi/dw + (2*pi*R)/(q^3*m*t_star)
    # tau_DM = 2*pi/mv^2
    q = energies/mass
    distance_inev = distance_pc/INEV_TO_PC
    tau_star = 2*PI/dw + 2*PI*distance_inev/(q**3 * mass * burst_duration)
    tau_DM = 2*PI/(energies*AVG_VEL_DM**2)
    
    # Compute rescaling factor (fraction in Eq. 46 in arXiv:2502.08716v1)
    rescaling_factor = [
        (t_int_DM[i]**(1/4)) * min(tau_DM[i]**(1/4), t_int_DM[i]**(1/4))/
        (min(signal_duration[i]**(1/4), t_int[i]**(1/4)) * min(tau_star[i]**(1/4), t_int[i]**(1/4))) 
        for i in range(len(energies))]
    
    return rho, rescaling_factor

def d_probe(w, rho, coherence, eta, order):
    """ Calculate value of dilatonic coupling we can probe

    Args:
        w (float): energy of phi emitted by the source
        rho (float): density of phi at the Earth
        coherence (array of floats): coherence times
        eta (float): fractional sensitivity of coupling_type to dark matter signal
        order (int): coupling order

    Returns:
        float: value of dilatonic coupling we can probe
    """
    if order == 1:
        phi = np.sqrt(rho)/(2*w)
        d = eta*PLANCK_MASS_EV/(2*np.sqrt(PI)*phi) * coherence
    elif order == 2:
        phi = np.sqrt(2*rho)/w
        d = eta*PLANCK_MASS_EV**2/(4*PI*phi**2) * coherence
    else:
        raise ValueError(f"Unsupported coupling order: {order}")
    return d

def d_from_Lambda(Lambda, order): 
    """ Calculate value of dilatonic coupling from Lambda for a given coupling order
    
    Args:
        Lambda (float): energy scale for new physics
        order (int): coupling order
    
    Returns:
        float: dilatonic coupling calculated from a given Lambda value
    """
    d = ((1/np.sqrt(4*PI))*(PLANCK_MASS_EV/Lambda))**order
    return d

def d2_from_delta_t(dt, L, m, E, Dg, K):
    """ Calculate value of quadratic dilatonic coupling from a time delay 
        (calculates d_i^(2) from Eq.39 in arXiv:2502.08716v1)
    
    Args:
        dt (float): time delay [s]
        L (float): propagation distance [pc]
        m (float): mass of phi [eV]
        E (array of floats): energies [eV]
        Dg (float): distance per galaxy of signal propagation [pc]
        K (float): expectation value of the operator in normal matter?
        
    Returns:
        float: value of quadratic dilatonic coupling from a time delay
    """
    # Galaxy number density [galaxies / Gpc^3]
    ng = 0.006e9 * (L**3)/(GPC_TO_PC**3)
    
    prefactor = PLANCK_MASS_EV**2/(8*PI)
    L_meters = L * PC_TO_METERS
    dt_c = dt * SPEED_OF_LIGHT
    
    if L < 1e5:
        d = np.array([
            (((E[i]**2)*(1 - (1/(dt_c/L_meters + 1)**2)) - m**2)/(8*PI*RHO_ISM*K))*(PLANCK_MASS_EV**2) for i in range(len(E))
        ])
    else:
        k1 = np.array([
            ((2*E[i]**2*dt_c)/L_meters) - m**2 for i in range(len(E))
        ])
        k3 = 1/((ng**(1/3) + 1)*Dg*RHO_ISM*K + (1-(ng**(1/3)+1)*Dg)*RHO_IGM*K)
        d = prefactor*k3*k1
    return d

def omegaoverm_noscreen(dt, L):
    """ Returns omega/m without screening (beta(x) = 0)
        omega           L+dt
        ----- = ---------------------
          m     sqrt(2*L*dt + dt**2)
          
    Args:
        dt (float): time delay 
        L (float): distance between source and detection
    
    Returns:
        omega_over_m (float): frequency to mass ratio
    """
    L = L*PC_TO_METERS
    dt = dt*SPEED_OF_LIGHT
    return (L+dt)/(np.sqrt(dt * (2*L+dt)))

def d2_screen(E, R, rho, m, K):
    """ Calculate the critical coupling 
    
    Args:
        E (array of floats): energies [eV]
        R (float): distance [pc]
        rho (float): energy density [eV^4]
        m (float): mass of phi [eV]
        K (float): energy density fraction [unitless]
    """
    d = PLANCK_MASS_EV**2 / (8*PI*rho*K) * (1/R**2 + E**2 - m**2)
    return d

def E_from_uncert(burst_duration):
    """ Calculate the energy from the burst duration using the Heisenberg uncertainty relation 
    
    Args: 
        burst_duration (float): t_star, the duration of the burst emission at the source [s]
    
    Returns:
        float: energy [eV]
    """
    return (2*PI/burst_duration)/SEC_TO_INEV