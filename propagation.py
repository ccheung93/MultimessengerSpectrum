import numpy as np

# CONVERSION FACTORS
SPEED_OF_LIGHT = 3e8                       # speed of light in m/s
HBAR = 6.528e-16                           # reduced Planck constant in eV*s

INEV_TO_METERS = 1.97e-7                   # eV^-1 to meters
PC_TO_METERS = 3.086e16                    # parsecs to meters
INEV_TO_PC = INEV_TO_METERS/PC_TO_METERS   # eV^-1 to parsecs
MASS_SUN_KG = 2e30                         # mass of the sun in kg
EV_TO_JOULES = 1.6e-19                     # eV to joules
EV_TO_KG = EV_TO_JOULES/SPEED_OF_LIGHT**2  # eV to kg
EV_TO_SOLAR = EV_TO_KG/MASS_SUN_KG         # eV to solar mass

SEC_TO_INEV = 1/HBAR                       # 1 second to eV^-1

DAY_TO_SEC = 60*60*24                      # number of seconds in 1 day
YEAR_TO_SEC = 365*DAY_TO_SEC               # number of seconds in 1 year

GCM3_TO_EV4 = 4.2e18                       # g/cm^3 to eV^4
PLANCK_MASS_EV = 1.2e28                    # Planck mass in eV
GPC_TO_PC = 1e9                            # gigaparsec to parsecs
METERS_TO_INEV = 5.07e6                    # meters to 1/eV
AVG_VEL_DM = 1e-3                          # average velocity of galactic Dark Matter

# Densities of ISM and IGM converted from g/cm^3 to eV^4
RHO_ISM_GCM3 = 1.67e-24
RHO_ISM = RHO_ISM_GCM3 * GCM3_TO_EV4
RHO_IGM_GCM3 = 1.67e-30
RHO_IGM = RHO_IGM_GCM3 * GCM3_TO_EV4

# Density and radius of the Earth, atmosphere and experiment
RHO_E = 5.5 * GCM3_TO_EV4
R_E = 6.371e6 * METERS_TO_INEV
RHO_ATM = 1e-3 * GCM3_TO_EV4
R_ATM = 1e4 * METERS_TO_INEV
RHO_EXP = RHO_E
R_EXP = 1 * METERS_TO_INEV

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
        rescaling_factor (list of float) - rescaling factors for each energy [unitless]
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
    mass_DM = energies
    tau_DM = 2*PI/(mass_DM*AVG_VEL_DM**2)
    
    # Compute rescaling factor (fraction in Eq. 46 in arXiv:2502.08716v1)
    rescaling_factor = [
        (t_dm**(1/4)) * min(tau_dm**(1/4), t_dm**(1/4)) /
        (min(sig_dur**(1/4), t_i**(1/4)) * min(tau_s**(1/4), t_i**(1/4)))
        for t_dm, tau_dm, sig_dur, t_i, tau_s in zip(t_int_DM, tau_DM, signal_duration, t_int, tau_star)
    ]
    return rho, rescaling_factor

def d_probe(w, rho, rescaling_factor, eta, order):
    """ Calculate value of dilatonic coupling we can probe

    Args:
        w (float): energy of phi emitted by the source
        rho (float): density of phi at the Earth
        rescaling_factor (array of floats): rescaling factors
        eta (float): fractional sensitivity of coupling_type to dark matter signal
        order (int): coupling order

    Returns:
        float: value of dilatonic coupling we can probe
    """
    if order == 1:
        phi = np.sqrt(rho)/(2*w)
        d = eta*PLANCK_MASS_EV/(2*np.sqrt(PI)*phi) * rescaling_factor
    elif order == 2:
        phi = np.sqrt(2*rho)/w
        d = eta*PLANCK_MASS_EV**2/(4*PI*phi**2) * rescaling_factor
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

def d2_from_delta_t(dt, R, m, E, Dg, K):
    """ Calculate value of quadratic dilatonic coupling from a time delay 
        (calculates d_i^(2) from Eq.39 in arXiv:2502.08716v1)
    
    Args:
        dt (float): time delay [s]
        R (float): propagation distance [pc]
        m (float): mass of phi [eV]
        E (array of floats): energies [eV]
        Dg (float): distance per galaxy of signal propagation [pc]
        K (float): energy density fraction [unitless]
        
    Returns:
        float: value of quadratic dilatonic coupling from a time delay
    """
    # Galaxy number density [galaxies / Gpc^3]
    number_density = 0.006e9
    Ng = number_density * (R/GPC_TO_PC)**3
    
    prefactor = PLANCK_MASS_EV**2/(8*PI)
    R_meters = R * PC_TO_METERS
    dt_c = dt * SPEED_OF_LIGHT
    
    # ISM regime:
    #          M_pl^2                       1
    # d2 = -------------- [ E^2 ( 1 - ------------ ) - m^2 ]
    #       8*pi*rho_ISM               (1+dt/R)^2
    #
    if R < 1e5:
        d2 = prefactor * (E**2*(1 - 1/(1 + dt_c/R_meters)**2) - m**2)/(RHO_ISM*K)
    # ISM+IGM regime:
    #          M_pl^2                     2E^2*dt/R - m^2
    # d2 = -------------- [ -------------------------------------------- ]
    #           8*pi         Ng*(Dg/R)*rho_ISM + (1-Ng*(Dg/R))*rho_IGM
    #
    else:
        Ng_eff = Ng**(1/3) + 1
        numer = 2*E**2*dt_c/R_meters - m**2
        denom = Ng_eff*Dg/R*RHO_ISM*K + (1-(Ng_eff*Dg/R))*RHO_IGM*K
        d2 = prefactor * numer / denom
    return d2

def omegaoverm_noscreen(dt, R):#
    """ Returns omega/m without screening (beta(x) = 0)
        omega           R+dt
        ----- = ---------------------
          m     sqrt(2*R*dt + dt**2)
          
    Args:
        dt (float): time delay 
        R (float): distance between source and detection
    
    Returns:
        omega_over_m (float): frequency to mass ratio
    """
    R = R * PC_TO_METERS
    dt = dt * SPEED_OF_LIGHT
    return (R + dt)/(np.sqrt(dt * (2*R+dt)))

def d2_screen(E, R, rho, m, K):
    """ Calculate the critical values of the quadratic dilatonic coupling 
    
    Args:
        E (array of floats): energies [eV]
        R (float): length scale [1/eV]
        rho (float): energy density [eV^4]
        m (float): mass of phi [eV]
        K (float): energy density fraction [unitless]
        
    Returns:
        float: critical value of quadratic dilatonic coupling
    """
    d = PLANCK_MASS_EV**2 / (8*PI*rho*K) * (1/(2*R)**2 + E**2 - m**2)
    return d

def E_from_uncert(burst_duration):
    """ Calculate the energy from the burst duration using the Heisenberg uncertainty relation 
    
    Args: 
        burst_duration (float): t_star, the duration of the burst emission at the source [s]
    
    Returns:
        float: energy [eV]
    """
    return 2*PI/(burst_duration*SEC_TO_INEV)