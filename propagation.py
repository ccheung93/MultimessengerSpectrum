import numpy as np

# CONVERSION FACTORS
SPEED_OF_LIGHT = 3e8                       # speed of light in m/s
HBAR = 6.528e-16                           # reduced Planck constant in eV*s

INEV_TO_METERS = 1.97e-7                   # eV^-1 to meters
PC_TO_METERS = 3.086e16                    # parsecs to meters
PC_TO_INEV = PC_TO_METERS/INEV_TO_METERS   # parsecs to 1/eV
EV_TO_JOULES = 1.6e-19                     # eV to joules
EV_TO_KG = EV_TO_JOULES/SPEED_OF_LIGHT**2  # eV to kg

MASS_SUN_KG = 2e30                         # mass of the sun in kg
SOLAR_TO_EV = MASS_SUN_KG/EV_TO_KG         # eV to solar mass

SEC_TO_INEV = 1/HBAR                       # 1 second to 1/eV

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

def signal_duration(Etot, m_phi, energies, burst_duration, distance, aw, integration_time=1): 
    """ Calculate the energy density of phi at the Earth and rescaling factor calculated for an array of signal durations
    
    Args:
        Etot (float): total energy of emitted phi particles from the source [eV]
        m_phi (float): mass of phi field [eV]
        energies (array_like): array of individual particle energies [eV]
        burst_duration (float): t_star, the duration of the burst emission at the source [s]
        distance (float): distance between source and detector [pc]
        aw (float): wavepacket uncertainty parameter based on uncertainty principle:
                    dw * t_star >= 1, where dw = spread in energies and t_star = burst_duration
                    for a given wavepacket, aw = dw * t_star, where aw >= 1
                    aw = 1 corresponds to a Gaussian wavepacket, since it has minimum uncertainty
        integration_time (float): integration time [days], default = 1 day

    Returns:
        rho (float): energy density of phi at Earth [eV^4]
        rescaling_factor (array_like): rescaling factors for each energy [unitless]
    """ 
    # Convert to proper units
    t_star = burst_duration*SEC_TO_INEV
    R = distance*PC_TO_INEV
    
    # Compute energy density of phi at Earth
    rho = calc_rho(Etot, m_phi, energies, t_star, R, aw)
    
    # Compute rescaling factor (fraction in Eq. 46 in arXiv:2502.08716v1)
    rescaling_factor = calc_rescaling_factor(m_phi, energies, t_star, R, aw, integration_time)
    
    return rho, rescaling_factor

def calc_rescaling_factor(m_phi, w, t_star, R, aw, integration_time=1):
    """ Calculate rescaling factor (Eq. 46 in arXiv:2502.08716v1)
                    t_int_DM^(1/4) * min(tau_DM^(1/4), t_int_DM^(1/4))
    rf = -----------------------------------------------------------------------------
            min(sig_dur^(1/4), t_int^(1/4)) * min(tau_star^(1/4), t_int_star^(1/4))
    Args:
        m_phi (float): mass of phi field [eV]
        w (array_like): array of individual particle energies [eV]
        burst_duration (float): the duration of the burst emission at the source [1/eV]
        R (float): distance between source and detector [1/eV]
        aw (float): wavepacket uncertainty parameter based on uncertainty principle:
                    dw * t_star >= 1, where dw = spread in energies and t_star = burst_duration
                    for a given wavepacket, aw = dw * t_star, where aw >= 1
                    aw = 1 corresponds to a Gaussian wavepacket, since it has minimum uncertainty
        integration_time (float): integration time [days], default = 1 day

    Returns:
        array_like: array of rescaling factors calculated for each signal duration [unitless]
    """
    # Convert integration time to seconds and then to inverse eV
    integration_time_s = integration_time * DAY_TO_SEC
    t_int = np.full_like(w, integration_time_s * SEC_TO_INEV)
    
    # Set integration time in seconds for dark matter experiment
    integration_time_DM_s = 1e6
    t_int_DM = np.full_like(w, integration_time_DM_s * SEC_TO_INEV)
    
    # Spread in energies from uncertainty principle: dw * t_star = aw -> dw = aw / t_star
    dw = aw/t_star
    
    # Spread of the phi wave during propagation
    q = w/m_phi
    dx_spread = (dw/w)*(R/q**2)
    
    # Calculate t_star_tilde, the signal duration at Earth (Eq. 43)
    signal_duration = np.sqrt(t_star**2 + dx_spread**2)
    
    # Calculate effective coherence time observed by the detector (Eq. 40)
    #              2*pi          2*pi*R
    # tau_star = -------- + ----------------
    #               dw        q^3*m*t_star
    tau_star = 2*PI/dw + 2*PI*R/(q**3 * m_phi * t_star)
    
    # Calculate coherence times for non-relativistic, ambient DM (Eq. 46)
    #            2*pi
    # tau_DM = ----------, m_DM = w
    #           m_DM*v^2
    mass_DM = w
    tau_DM = 2*PI/(mass_DM*AVG_VEL_DM**2)
    
    rescaling_factor = [
        (t_dm**(1/4)) * min(tau_dm**(1/4), t_dm**(1/4)) /
        (min(sig_dur**(1/4), t_i**(1/4)) * min(tau_s**(1/4), t_i**(1/4)))
        for t_dm, tau_dm, sig_dur, t_i, tau_s in zip(t_int_DM, tau_DM, signal_duration, t_int, tau_star)
    ]
    
    return rescaling_factor

def calc_rho(Etot, m_phi, w, t_star, R, aw, axion=False):
    """ Calculate the energy density of phi at Earth in eV^4
    
    Args:
        Etot (float): total energy of emitted phi particles from the source [eV]
        m_phi (float): mass of phi field [eV]
        energies (array_like): array of individual particle energies [eV]
        t_star (float): the duration of the burst emission at the source [1/eV]
        R (float): distance between source and detector [1/eV]
        aw (float): wavepacket uncertainty parameter based on uncertainty principle:
                    dw * t_star >= 1, where dw = spread in energies and t_star = burst_duration
                    for a given wavepacket, aw = dw * t_star, where aw >= 1
                    aw = 1 corresponds to a Gaussian wavepacket, since it has minimum uncertainty

    Returns:
        float: energy density of phi at Earth [eV^4]
    """ 
    # Spread in energies from uncertainty principle: dw * t_star = aw -> dw = aw / t_star
    dw = aw/t_star
    
    # Physical size of the phi wave at the source, assuming it's traveling at c = 1.
    dx_burst = t_star
    
    # Spread of the phi wave during propagation
    q = np.sqrt((w/m_phi)**2-1) if axion else w/m_phi # TODO - figure out where expression for q for axions comes from
    dx_spread = (dw/w)*(R/q**2)
    
    # Total spread of the wavepacket
    dx = dx_burst + dx_spread
    
    # Energy density of phi at the detector location (Eq. 30)
    rho = Etot/(4*PI*R**2*dx)
    
    return rho

def d_probe(w, rho, rescaling_factor, eta, order, axion=False):
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
    if axion:
        rhoDM = 3.05e-6 #eV^4 what is this
        d_DM = 2e-13 * (1e3/1.022) # TODO - what is this?
        d = d_DM * AVG_VEL_DM * np.sqrt(rhoDM/rho) * rescaling_factor # TODO - check how this is derived, dimensions not equal (perhaps replace AVG_VEL_DM with velocity_ratio = v_DM/v_star = AVG_VEL_DM)
        return d
    
    if order == 1:
        phi = np.sqrt(rho)/(2*w)
        d = eta*PLANCK_MASS_EV/(2*np.sqrt(PI)*phi) * rescaling_factor # TODO - check how this is derived (Eq. 47?)
    elif order == 2:
        phi = np.sqrt(2*rho)/w
        d = eta*PLANCK_MASS_EV**2/(4*PI*phi**2) * rescaling_factor # TODO - check how this is derived (Eq. 48?)
    else:
        raise ValueError(f"Unsupported coupling order: {order}")
    return d

def d_from_Lambda(Lambda, order, axion=False): # TODO - update docstring and digure out constants
    """ Calculate value of dilatonic coupling from Lambda for a given coupling order
    
    Args:
        Lambda (float): energy scale for new physics
        order (int): coupling order
    
    Returns:
        float: dilatonic coupling calculated from a given Lambda value
    """
    if axion:
        return 1e-13 * (1e3/1.022) # TODO - what is this? NOTE - 1e-13 here, 2e-13 above in d_probe()
    else:
        return ((1/np.sqrt(4*PI))*(PLANCK_MASS_EV/Lambda))**order

def d2_from_delta_t(dt, R, m, E, Dg, K, axion=False):
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
    if axion: # TODO - explain this
        gamma = (dt * SPEED_OF_LIGHT) / (R * PC_TO_METERS) + 1
        moverE = np.sqrt(1 - 1 / gamma**2)
        Em = m / moverE
        return [0 if Ei < Em else 1e100 for Ei in E]
    
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