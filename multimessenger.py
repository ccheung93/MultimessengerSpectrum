import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Computer Modern sans serif']

from matplotlib.ticker import FuncFormatter
import time
usetex: True

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
PI = np.pi 
COLORLIST = ["tab:red", "tab:orange", "tab:purple"]


def signal_duration(Etot, mass, energies, burst_duration, distance_pc, aw, integration_time=1): 
    """ Calculate the duration of the signal 
    
    Args:
        Etot (float): total energy of emitted phi particles from the source [eV]
        mass (float): mass of phi field [eV]
        energies (array_like): array of individual particle energies [eV]
        burst_duration (float): t_star, the duration of the burst emission at the source [s]
        distance_pc (float): distance between source and detector [parsecs]
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
    # tau_DM = 2*pi/mv^2 ! NOTE - should m be energies or mass (m_phi)?
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

def d1_probe(w, rho, coherence, eta):
    """ Calculate value of linear dilatonic coupling we can probe

    Args:
        w (float): energy of phi emitted by the source
        rho (float): density of phi at the Earth
        coherence (array of floats): coherence times
        eta (float): fractional sensitivity of coupling_type to dark matter signal

    Returns:
        float: value of linear dilatonic coupling we can probe
    """
    phi = np.sqrt(rho)/(2*w)
    d = eta*PLANCK_MASS_EV/(2*np.sqrt(PI)*phi) * coherence
    return d

def d2_probe(w, rho, coherence, eta): 
    """ Calculate value of quadratic dilatonic coupling we can probe
    
    Args:
        w (float): energy of phi emitted by the source
        rho (float): density of phi at the Earth
        coherence (array of floats): coherence times
        eta (float): fractional sensitivity of coupling_type to dark matter signal
        
    Returns:
        float: value of quadratic dilatonic coupling we can probe
    """
    phi = np.sqrt(2*rho)/w
    d = eta*PLANCK_MASS_EV**2/(4*PI*phi**2) * coherence
    return d

def d_from_Lambda(Lambda): #For supernova constraint
    d = (1/(4*PI))*(PLANCK_MASS_EV/Lambda)**2
    return d

def d_from_delta_t(dt, L, m, E, Dg, K):
    rho_ISM = 1.67e-24 * K * GCM3_TO_EV4 #density in g/cm^3 times the expectation value of the operator in normal matter times conversion to eV^4
    rho_IGM = 1.67e-30 * K * GCM3_TO_EV4   
    ng = 0.006e9 * (L**3)/(GPC_TO_PC**3)#1e6 #galaxies/Gpc^3
    
    k1 = np.array([((2*E[i]**2*(dt*SPEED_OF_LIGHT))/(L*PC_TO_METERS)) - m**2 for i in range(len(E))])
    k2 = PLANCK_MASS_EV**2/(2*4*PI)
    if L < 1e5:
        d = np.array([(((E[i]**2)*(1 - (1/((dt*SPEED_OF_LIGHT)/(L*PC_TO_METERS) + 1)**2)) - m**2)/(8*PI*rho_ISM))*(PLANCK_MASS_EV**2) for i in range(len(E))])
    else:
        k1 = np.array([((2*E[i]**2*(dt*SPEED_OF_LIGHT))/(L*PC_TO_METERS)) - m**2 for i in range(len(E))])
        k3 = 1/((ng**(1/3) + 1)*Dg*rho_ISM + (1-(ng**(1/3)+1)*Dg)*rho_IGM)
        d = k3*k2*k1
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

def d_screenearth(E, m, K): #Probably don't need this or the next for the ALP, but left it here just in case.
    rho_op_ex = 1e-3
    rho_E = 5.5 * rho_op_ex * GCM3_TO_EV4
    R = 6.371e6 * 5.07e6
    d = [((PLANCK_MASS_EV**2)/(2*4*PI*rho_E*K))*((1/(2*R)**2) + E[i]**2-m**2) for i in range(len(E))]
    return d


def d_screen(E, R, rho, m, K):
    d = [((PLANCK_MASS_EV**2)/(2*4*PI*rho*K))*((1/R**2) + E[i]**2-m**2) for i in range(len(E))]
    return d

def E_from_uncert(t_star):
    return (2*PI/t_star)/SEC_TO_INEV

def exponentlabel(x, pos):
    return str("{:.0f}".format(np.log10(x)))

def get_distance_label(R):
    """ Returns distance scale label for a given value in parsecs 
    
    Args:
        R (float): distance in parsecs
        
    Returns:
        distance (str): a string label representing the distance scale in kpc or Mpc
    """
    if R == 1e4:
        distance_label = '10kpc'
    elif R == 1e7:
        distance_label = '10Mpc'
    else:
        raise ValueError(f"Unsupported distance value: {R}")
    
    return distance_label

def set_K_params(coupling_type, coupling_order):
    """ determine K_parameters # TODO - add details

    Args:
        coupling_type (str): type of coupling (photon, electron, gluon)
        coupling_order (str): order of coupling (linear or quadratic)

    Returns:
        K_space (float): fraction of energy density of the ISM that is from coupling_type
        K_E (float): fraction of energy density of the Earth that is from coupling_type
        K_atm (float): fraction of energy density of the atmosphere that is from coupling_type
        eta (float): fractional sensitivity of coupling_type to dark matter signal
        ylabel (str): y-axis label
    """
    if coupling_type == 'photon':
        K_space = 6.3e-4
        K_E = 1.9e-3
        K_atm = 9.5e-4
        eta = 1e-19/6000
        if coupling_order == 'linear':
            ylabel = r'$\log_{10}(d^{(1)}_e)$'
        if coupling_order == 'quad':    
            ylabel = r'$\log_{10}(d^{(2)}_e)$'
    elif coupling_type == 'electron':
        K_space = 4.4e-4
        K_E = 2.4e-4
        K_atm = 2.7e-4
        eta = 1e-17
        if coupling_order == 'linear':
            ylabel = r'$\log_{10}(d^{(1)}_{m_e})$'
        if coupling_order == 'quad':
            ylabel = r'$\log_{10}(d^{(2)}_{m_e})$'
    elif coupling_type == 'gluon':
        K_space = 1.0
        K_E = 1.0
        K_atm = 1.0
        eta = 1e-24
        if coupling_order == 'linear':
            ylabel = r'$\log_{10}(d^{(1)}_g)$'
        if coupling_order == 'quad':    
            ylabel = r'$\log_{10}(d^{(2)}_g)$'
            
    return K_space, K_E, K_atm, eta, ylabel

def setup_axes(ax, formatter, coupling_order):
    """ Set up axes for subplot (i, j) """
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)
    ax.set_xticks(np.logspace(-20,-6,7))
    ax.set_xlim(.3e-20,0.9e-6)
    if coupling_order == 'linear': 
        ax.set_ylim(1e-9,0.9e0)
    ax.tick_params(direction="in")
    
def plot_MICROSCOPE(ax, Elist, Microscope_m):
    """ Plot MICROSCOPE EP violation limits """
    ax.plot(Elist, Microscope_m, color = 'gray', linewidth = 2)
    ax.fill_between(Elist, Microscope_m, [1e50 for i in range(len(Elist))], color = 'gray', alpha = 0.1)
    ax.text(3.5e-11, Microscope_m[0]*1.3, r'${\rm MICROSCOPE}$', color = 'k')

def plot_FifthForce(ax, t, Elist, E_unc, FifthForce_m):
    """ Plot fifth-force limits """
    ax.plot(Elist, FifthForce_m)
    ax.plot([E_unc, E_unc], [1e50, 1e-50], color = 'chocolate', linestyle = '--')
    ax.fill_between([1e-50, E_unc], [1e-50, 1e-50], [1e50, 1e50], color = 'chocolate', alpha = 0.1)

def plot_coupling(ax, Elist, t, m, R, eta, Etot, m_bench, wmp_contour, coupling_order):
    """ Plot coupling limits """
    rho, coherence = signal_duration(Etot, m, Elist, t, R, 1)
    
    if coupling_order == "linear":
        coupling = d1_probe(Elist, rho, coherence, eta)
        min_y = 1e-50
    elif coupling_order == "quad":
        coupling = d2_probe(Elist, rho, coherence, eta)
        min_y = 1e-10
    
    ax.plot(m_bench*wmp_contour, coupling, c = 'k', linewidth = 2, alpha = 1)
    ax.plot([m, m], [min_y, 1e50], c = 'k', linestyle = '--')
    ax.fill_between([1e-30, m], min_y, 1e50, facecolor = 'none', hatch = "/", edgecolor = 'k', alpha = 0.3)

def plot_d_from_delta_t(ax, Elist, m, R, dt, Dg, K_space):
    d = d_from_delta_t(DAY_TO_SEC, R, m, Elist, Dg, K_space)
    ax.plot(Elist, d, color = COLORLIST[2], linewidth = 2, linestyle = '--'  )
    
    d = d_from_delta_t(dt, R, m, Elist, Dg, K_space)
    ax.plot(Elist, d, color = COLORLIST[0], linewidth = 2, linestyle = '--'  )

def plot_fill_region(ax, Elist, Microscope_m, t, m, R, eta, Etot, E_unc):    
    omega_over_m = omegaoverm_noscreen(DAY_TO_SEC, R)
    fillregion_x = np.array([Elist[l] for l in range(len(Elist)) if all([Elist[l] > E_unc, Elist[l] > m*omega_over_m])])
    fillregion_y = [Microscope_m[l] for l in range(len(fillregion_x))]
    
    rho, coherence = signal_duration(Etot, m, fillregion_x, t, R, 1)
    coupling = d1_probe(fillregion_x, rho, coherence, eta)
    ax.fill_between(fillregion_x, coupling, fillregion_y, where = coupling < fillregion_y, color = 'tab:green',alpha = 0.3)

def plot_fill_region_quad(ax, Elist, t, m, R, eta, dt, Etot, E_unc, R_exp, rho_exp, K_E, K_space):
    fillregion_x = Elist[Elist > E_unc]
    rho, coherence = signal_duration(Etot, m, fillregion_x, t, R, 1)
    coupling = d2_probe(fillregion_x, rho, coherence, eta)
    d_exp = d_screen(fillregion_x, R_exp, rho_exp, m, K_E)
    
    if R < 1e5:
        ddt_day = d_from_delta_t(DAY_TO_SEC, R, m, fillregion_x, 30e-6, K_space)
        fillregion_y = np.minimum(d_exp, ddt_day)

        ax.fill_between(fillregion_x, coupling, fillregion_y, where = coupling < fillregion_y, color = 'tab:green', alpha = 0.3)
    else:
        plot_d_from_delta_t(ax, Elist, m, R, dt, 1e-6, K_space)
        
        ddt_day_fill = d_from_delta_t(DAY_TO_SEC, R, m, fillregion_x, 1e-6, K_space)
        fillregion_y = np.minimum(d_exp, ddt_day_fill)
        
        ax.fill_between(fillregion_x, coupling, fillregion_y, where = coupling < fillregion_y, color = 'tab:green', alpha = 0.3)
        
        ddt_day1 = d_from_delta_t(DAY_TO_SEC, R, m, Elist, 1e-6, K_space)
        ddt1 = d_from_delta_t(dt, R, m, Elist, 1e-6, K_space)
        ddt_day30 = d_from_delta_t(DAY_TO_SEC, R, m, Elist, 30e-6, K_space)
        ddt30 = d_from_delta_t(dt, R, m, Elist, 30e-6, K_space)
        ax.fill_between(Elist, ddt_day1, ddt_day30, color = COLORLIST[2], alpha = 0.1)
        ax.fill_between(Elist, ddt1, ddt30, color = COLORLIST[0], alpha = 0.1)

def plot_time_labels(ax, m, dt, R, coupling_order):
    bbox_style = lambda color: dict(facecolor = 'white', alpha = 1, edgecolor = color, boxstyle = 'round,pad=.1')
    
    # Define time labels 
    time_labels = [
        {
            "delta_t": dt, 
            "color": "tab:red", 
            "txt": r'$\delta t\, \gtrsim \, 1~{\rm yr}~ \uparrow $' # dt >~ 1 year
            },
        {
            "delta_t": DAY_TO_SEC, 
            "color": "tab:purple", 
            "txt": r'$\delta t\, \gtrsim \, 1~{\rm day}~ \uparrow$' # dt >~ 1 day
            }
    ]
    
    # Define y positions of text box based on coupling_order
    pos_y_coupling = {
        "linear": 1e-7,
        "quad": 1e16
    }
    
    # Label regions in parameter space for each time label
    for label in time_labels:
        omega_over_m = omegaoverm_noscreen(label["delta_t"], R)
        pos_x = m*omega_over_m/4
        pos_y = pos_y_coupling[coupling_order]
        txt = label["txt"]
        ax.text(pos_x, pos_y, txt, rotation = 90, fontsize = 25, color = label["color"], bbox = bbox_style(label["color"]))
        
def plot_couplings_screened(ax, Elist, m, K_E, K_atm, R_atm, rho_atm, R_exp, rho_exp):
    d_screen_earth = d_screenearth(Elist, m, K_E)
    d_screen_atm = d_screen(Elist, R_atm, rho_atm, m, K_atm)
    d_screen_exp = d_screen(Elist, R_exp, rho_exp, m, K_E)
    ax.fill_between(Elist, d_screen_earth, 1e100, color = 'tab:blue', alpha = .05)
    ax.plot(Elist, d_screen_atm, color = 'tab:blue', linestyle = 'dashed')
    ax.plot(Elist, d_screen_exp, color = 'tab:blue', linestyle = 'dotted')
    ax.plot(Elist, d_screen_earth, color = 'tab:blue', linewidth = 3)

def plot_E_unc(ax, E_unc):
    ax.plot([E_unc, E_unc], [1e50, 1e-50], color = 'chocolate', linestyle = '--')
    ax.fill_between([1e-50, E_unc], [1e-50, 1e-50], [1e50, 1e50], color = 'chocolate', alpha = 0.1)

def label_omega_lt_mass(ax, m, coupling_order):
    # Label region in parameter space where omega < scalar field mass
    if m > 1e-20:
        pos_x = m/200
        pos_y = {
            "linear": 1e-7,
            "quad": 1e12
        }
        txt = r'$\omega<m_{\phi}$'
        bbox_style = dict(facecolor = 'whitesmoke',
                          alpha = 1,
                          edgecolor = 'k',
                          boxstyle = 'round,pad=.1')
        ax.text(pos_x, pos_y[coupling_order], txt, color = 'k', bbox = bbox_style)

def plot_supernova(ax, Elist, coupling_type):
    supernova_config = {
        "photon": {
            "ylim": (.5e6, 8e32),
            "txt_y": 3e29,
            "lbl_y": 3e31,
            "txt": r'${\rm Supernova}~\gamma \gamma \rightarrow \phi \phi$',
            "line": [d_from_Lambda(1e12)] * len(Elist)
        },
        "electron": {
            "ylim": (.5e9, 5e33),
            "txt_y": 3e29,
            "lbl_y": 1e32,
            "txt": r'${\rm Supernova}~e^+ e^- \rightarrow \phi \phi$',
            "line": [5e31] * len(Elist) # NOTE - should this be d_from_Lambda(5e31)?
        },
        "gluon": {
            "ylim": (.5e4, 5e30),
            "txt_y": 3e26,
            "lbl_y": 3e29,
            "txt": r'${\rm Supernova}~N N \rightarrow N N \phi \phi$',
            "line": [d_from_Lambda(15e12)] * len(Elist)
        }
    }
    
    txt_x = 1e-19
    lbl_x = 1e-19
    omega_txt = r'$\omega\,t_{*} \lesssim \, 2\pi$'
    bbox_style = dict(facecolor='white', 
                      alpha = 1, 
                      edgecolor='chocolate', 
                      boxstyle='round,pad=.1')
    
    if coupling_type in supernova_config:
        config = supernova_config[coupling_type]
        ax.set_ylim(*config["ylim"])
        ax.text(txt_x, config["txt_y"], omega_txt, color = 'tab:brown', bbox = bbox_style)
        ax.text(lbl_x, config["lbl_y"], config["txt"], fontsize = 30, color = 'black')
        ax.plot(Elist, config["line"], color = 'gray', linewidth = 3)
        ax.fill_between(Elist, config["line"], 1e100, color = 'gray', alpha = 0.1)

def plot_omega_ts(ax, E_unc, filename):
    """ Annotate plot regions """
    # Define lambda for bbox style
    bbox_style = lambda color: dict(facecolor = 'white', alpha = 1, edgecolor = color, boxstyle = 'round,pad=.1')
    
    # Define plot configurations for relevant distance scales
    distance_scale_config = {
        "10Mpc_": {
            "pos": (E_unc/200, 1e-1),
            "txt": r'$E_{\rm tot} = M_{\odot}$'+'\n' + r'$R = 10~{\rm Mpc}$'
        },
        "10kpc_": {
            "pos": (1e-20, 1e-2),
            "txt": r'$E_{\rm tot} = 10^{-2} M_{\odot}$'+'\n' + r'$R = 10~{\rm kpc}$'
        }
    }
    
    for prefix, val in distance_scale_config.items():
        if filename.startswith(prefix):
            pos_x, pos_y = val["pos"]
            txt = r'$\omega\,t_{*} \lesssim \, 2\pi$' # omega t_* <~ 2pi
            ax.text(pos_x, pos_y, txt, color = 'tab:brown', bbox = bbox_style("chocolate"))
                
def plot_parameter_list(ax, i, j, coupling_type, coupling_order, filename):
    """ Plot parameter lists in bottom right plot """
    
    # Only plot parameter list for bottom right subplot ax(1, 1)
    if (i, j) != (1, 1): return
    
    distance_scales = ["10Mpc_", "10kpc_"]
    
    bbox_style = dict(facecolor='tab:purple', 
                      alpha = 0.0, 
                      edgecolor = 'white', 
                      boxstyle='round,pad=.2')
    
    # Define positions for parameter list for different plots
    parameter_list_positions = {
        "photon": {
            "10Mpc_": (1e-11, 1e7),
            "10kpc_": (1e-11, 1e8)
        },
        "electron": {
            "10Mpc_": (1e-10, 1e10),
            "10kpc_": (1e-11, 1e10)
        },
        "gluon": {
            "10Mpc_": (1e-11, 1e5),
            "10kpc_": (1e-11, 1e7)
        }
    }
    
    # Define Etot in solar masses and R in parsecs
    parameter_list = {
        "photon": {
            "10Mpc_": r'$E_{\rm tot} = M_{\odot}$'+ '\n' + r'$R = 10~{\rm Mpc}$',
            "10kpc_": r'$E_{\rm tot} = 10^{-2} M_{\odot}$' + '\n' + r'$R = 10~{\rm kpc}$'
        },
        "electron": {
            "10Mpc_": r'$E_{\rm tot} = M_{\odot}$' + '\n' + r'$R = 10~{\rm Mpc}$',
            "10kpc_": r'$E_{\rm tot} = 10^{-2} M_{\odot}$' + '\n' + r'$R = 10~{\rm kpc}$'
        },
        "gluon": {
            "10Mpc_": r'$E_{\rm tot} = M_{\odot}$'+'\n' +r'$R = 10~{\rm Mpc}$',
            "10kpc_": r'$E_{\rm tot} = 10^{-2} M_{\odot}$' + '\n' + r'$R = 10~{\rm kpc}$'
        }
    }
    
    # Define eta_DM (sensitivity to DM signal) values for different fields
    eta_DM = {
        "photon": r'$\,\eta_{\rm DM} = 10^{-19}/5900$',
        "electron": r'$\,\eta_{\rm DM} = 10^{-17}$',
        "gluon": r'$\,\eta_{\rm DM} = 10^{-19}/10^5$'
    }

    for prefix in distance_scales:
        if filename.startswith(prefix):
            if coupling_order == "linear":
                pos = (5e-12, 4e-9) if coupling_type == "photon" else (2e-11, 4e-9)
            else:
                pos = parameter_list_positions[coupling_type][prefix]
            txt = parameter_list[coupling_type][prefix] + '\n' + eta_DM[coupling_type]
            ax.text(*pos, txt, bbox = bbox_style)
    

def plot_critical_screening(ax, K_E, K_atm, coupling_type, filename):
    """ Plot critical screening lines """
    distance_scales = ["10Mpc_", "10kpc_"]
    
    # Define labels for critical screening
    crit_screening_labels = {
        "photon": {
            "eth": r'$d_{e,{\rm crit}}^{(2)\, \rm\oplus}$',
            "atm": r'$d_{e,{\rm crit}}^{(2)\, \rm atm}$',
            "exp": r'$d_{e,{\rm crit}}^{(2)\, \rm app}$' # NOTE - Called this exp instead of app (more obvious to me)
        },
        "electron": {
            "eth": r'$d_{m_e,{\rm crit}}^{(2)\, \rm\oplus}$',
            "atm": r'$d_{m_e,{\rm crit}}^{(2)\, \rm atm}$',
            "exp": r'$d_{m_e,{\rm crit}}^{(2)\, \rm app}$'
        },
        "gluon": {
            "eth": r'$d_{g,{\rm crit}}^{(2)\, \rm\oplus}$',
            "atm": r'$d_{g,{\rm crit}}^{(2)\, \rm atm}$',
            "exp": r'$d_{g,{\rm crit}}^{(2)\, \rm app}$'
        }
    }
    
    # Define positions of text for critical screening
    crit_screening_positions = {
        "photon": {
            "10Mpc_": {
                "eth": (3e-13, 2e12/K_E),
                "atm": (1e-12, 5e21/K_atm),
                "exp": (5e-11, 7e25/K_E)
            },
            "10kpc_": {
                "eth": (7e-15, 2e13/K_E),
                "atm": (2e-12, 1e19/K_atm),
                "exp": (5e-11, 7e25/K_E)
            }
        },
        
        "electron": {
            "10Mpc_": {
                "eth": (2e-13, 2e11/K_E),
                "atm": (2e-12, 1e19/K_atm),
                "exp": (1e-10, 7e25/K_E)
            },
            "10kpc_": {
                "eth": (5e-15, 2e9/K_E),
                "atm": (2e-13, 5e21/K_atm),
                "exp": (5e-11, 7e25/K_E)
            }
        },
        
        "gluon": {
            "10Mpc_": {
                "eth": (2e-13, 2e11/K_E),
                "atm": (2e-12, 1e19/K_atm),
                "exp": (5e-12, 7e25/K_E)
            },
            "10kpc_": {
                "eth": (2e-14, 2e9/K_E),
                "atm": (2e-12, 1e19/K_atm),
                "exp": (5e-11, 7e25/K_E)
            }
        }
    }
    
    for prefix in distance_scales:
        if filename.startswith(prefix):
            pos = crit_screening_positions[coupling_type][prefix]
            lbl = crit_screening_labels[coupling_type]
            ax.text(*pos["eth"], lbl["eth"], fontsize = 35, color = 'tab:blue')
            ax.text(*pos["atm"], lbl["atm"], fontsize = 35, color = 'tab:blue')
            ax.text(*pos["exp"], lbl["exp"], fontsize = 35, color = 'tab:blue')


def plots(R, Etot, coupling_type, coupling_order):
    m_bench = 1e-21 # in eV
    m_bench2 = 1e-18
    ts_bench = 1 # in s
    ts_bench2 = 1e2

    mass = [[m_bench,m_bench2],
            [m_bench,m_bench2]]
    ts = [[ts_bench,ts_bench],
          [ts_bench2,ts_bench2]]
    
    distance_label = get_distance_label(R) # get distance label in kpc or Mpc

    wmp_contour = np.logspace(0,30,1000)

    K_space, K_E, K_atm, eta, ylabel = set_K_params(coupling_type, coupling_order)

    filename = distance_label+'_'+coupling_type+'_'+coupling_order+'_dilatoniccoupling.pdf'

    fig, ax = plt.subplots(2, 2, figsize = (30, 21), sharex = True, sharey = True)
    plt.rcParams['mathtext.fontset'] = 'cm'
    plt.rcParams.update({'font.size': 35,'font.family':'STIXGeneral'})
    axis_font = {'fontname':'Times New Roman', 'size':'35'}
    plt.subplots_adjust(wspace = 0, hspace = 0)

    plt.rcParams['hatch.color'] = 'lightgray'

    ax[0,0].set_yscale('log')
    ax[0,0].set_xscale('log')
    
    formatter = FuncFormatter(exponentlabel) # customize tick labels on axes to be in log10

    dt = 1 * 3.154e7 #year
    dt_100 = 100 * 3.154e7
    Elist = mass[0][0]*wmp_contour
    labels = [r'$\mathcal{E}_{\phi}= 10^5~m_{\phi}$',r'$\mathcal{E}_{\phi}= 10^7~m_{\phi}$']
    Lambda_earth_screen = [1e22 for i in range(len(Elist))] 
    d_earth_screen = [d_from_Lambda(1e22) for i in range(len(Elist))] 

    rho_E = 5.5 * K_E * GCM3_TO_EV4        
    R_atm = 1e4* 5.07e6
    rho_atm = 1e-3 * K_atm * GCM3_TO_EV4 
    R_exp = 1e0* 5.07e6
    rho_exp = 5.5 * K_E * GCM3_TO_EV4 

    # Initialize arrays
    Microscope_x, Microscope_y = [], []
    EotWashEP_x, EotWashEP_y = [], []
    FifthForce_x, FifthForce_y = [], []

    with open('Linear Scalar Photon/MICROSCOPE.txt', 'r') as f:
        for line in f:
            Microscope_x.append([float(value) for value in line.strip().split()][0])
            Microscope_y.append([float(value) for value in line.strip().split()][1])

    with open('Linear Scalar Photon/EotWashEP.txt', 'r') as f:
        for line in f:
            EotWashEP_x.append([float(value) for value in line.strip().split()][0])
            EotWashEP_y.append([float(value) for value in line.strip().split()][1])

    with open('Linear Scalar Photon/FifthForce.txt', 'r') as f:
        for line in f:
            FifthForce_x.append([float(value) for value in line.strip().split()][0])
            FifthForce_y.append([float(value) for value in line.strip().split()][1])

    Microscope_m = [Microscope_y[0] for i in range(len(Elist))]
    FifthForce_m = [FifthForce_y[0] for i in range(len(Elist))]

    rho_op_ex = 6.3e-4
    rho_E = 5.5 * rho_op_ex * GCM3_TO_EV4        
    R_atm = 1e4* 5.07e6
    rho_atm = 1e-3 * rho_op_ex * GCM3_TO_EV4 
    R_exp = 1e0* 5.07e6
    rho_exp = 5.5 * rho_op_ex * GCM3_TO_EV4 
    rho_ISM = 1.64*1.67e-24 * GCM3_TO_EV4

    if coupling_order == 'linear':    
        for i in range(2):
            for j in range(2):
                t = ts[i][j]
                m = mass[i][j]
                E_unc = E_from_uncert(t)
                axij = ax[i][j]

                setup_axes(axij, formatter, coupling_order)
                plot_MICROSCOPE(axij, Elist, Microscope_m)
                plot_FifthForce(axij, t, Elist, E_unc, FifthForce_m)
                plot_coupling(axij, Elist, t, m, R, eta, Etot, m_bench, wmp_contour, coupling_order)
                label_omega_lt_mass(axij, m, coupling_order)      
                plot_fill_region(axij, Elist, Microscope_m, t, m, R, eta, Etot, E_unc)
                plot_d_from_delta_t(axij, Elist, m, R, dt, 30e-6, K_space)
                plot_time_labels(axij, m, dt, R, coupling_order)
                plot_omega_ts(axij, E_unc, filename)
                plot_parameter_list(axij, i, j, coupling_type, coupling_order, filename)
                    
                ax[0,j].set_title(r'$\log_{10}(m_{\phi}/{\rm eV}) = $'+str(int(np.log10(mass[0][j]))), pad = 20)
                ax[i,1].set_ylabel(r'$t_*$ = '+str(int(ts[i][0]))+r' s',labelpad = 40,rotation = 270)
                ax[i,1].yaxis.set_label_position("right")

    if coupling_order == 'quad':    
        for i in range(2):
            for j in range(2):
                t = ts[i][j]
                m = mass[i][j]
                E_unc = E_from_uncert(t)
                axij = ax[i][j]
                
                setup_axes(axij, formatter, coupling_order)
                plot_couplings_screened(axij, Elist, m, K_E, K_atm, R_atm, rho_atm, R_exp, rho_exp)
                plot_E_unc(axij, E_unc)
                plot_coupling(axij, Elist, t, m, R, eta, Etot, m_bench, wmp_contour, coupling_order)
                label_omega_lt_mass(axij, m, coupling_order)
                plot_d_from_delta_t(axij, Elist, m, R, dt, 30e-6, K_space)
                plot_fill_region_quad(axij, Elist, t, m, R, eta, dt, Etot, E_unc, R_exp, rho_exp, K_E, K_space)
                plot_supernova(axij, Elist, coupling_type)
                plot_critical_screening(axij, K_E, K_atm, coupling_type, filename)

                plot_parameter_list(axij, i, j, coupling_type, coupling_order, filename)
                if filename == '10Mpc_'+coupling_type+'_quad_dilatoniccoupling.pdf':
                    plot_time_labels(axij, m, dt, R, coupling_order)

                if filename == '10kpc_'+coupling_type+'_quad_dilatoniccoupling.pdf':
                    if coupling_type == 'photon':
                        # These time labels are hard-coded
                        ax[i,j].text(3e-17,1e27,r'$\delta t\, \gtrsim \, 1~{\rm yr}~\uparrow$',rotation = 37, fontsize = 25, color = 'tab:red',bbox=dict(facecolor='white', alpha = 1, edgecolor='tab:red',boxstyle='round,pad=.1'))
                        ax[i,j].text(6e-16,9e26,r'$\delta t\, \gtrsim \, 1~{\rm day}~\uparrow$',rotation = 37, fontsize = 25, color = 'tab:purple',bbox=dict(facecolor='white', alpha = 1, edgecolor='tab:purple',boxstyle='round,pad=.1'))
                        

                    elif coupling_type == 'electron':
                        # These time labels are hard-coded
                        ax[i,j].text(3e-17,8.5e26,r'$\delta t\, \gtrsim \, 1~{\rm yr}~\uparrow$',rotation = 39, fontsize = 25, color = 'tab:red',bbox=dict(facecolor='white', alpha = 1, edgecolor='tab:red',boxstyle='round,pad=.1'))
                        ax[i,j].text(6e-16,8e26,r'$\delta t\, \gtrsim \, 1~{\rm day}~\uparrow$',rotation = 39, fontsize = 25, color = 'tab:purple',bbox=dict(facecolor='white', alpha = 1, edgecolor='tab:purple',boxstyle='round,pad=.1'))
                        
                    elif coupling_type == 'gluon':
                        ax[i,j].set_ylim(.5e5,8e30)
                        
                        # These time labels are hard-coded
                        ax[i,j].text(3e-17,6e23,r'$\delta t\, \gtrsim \, 1~{\rm yr}~\uparrow$',rotation = 38, fontsize = 25, color = 'tab:red',bbox=dict(facecolor='white', alpha = 1, edgecolor='tab:red',boxstyle='round,pad=.1'))
                        ax[i,j].text(6e-16,5e23,r'$\delta t\, \gtrsim \, 1~{\rm day}~\uparrow$',rotation = 38, fontsize = 25, color = 'tab:purple',bbox=dict(facecolor='white', alpha = 1, edgecolor='tab:purple',boxstyle='round,pad=.1'))
                        
                ax[0,j].set_title(r'$\log_{10}(m_{\phi}/{\rm eV}) = $'+str(int(np.log10(mass[0][j]))), pad = 20)
                ax[i,1].set_ylabel(r'$t_*$ = '+str(int(ts[i][0]))+r' s',labelpad = 40,rotation = 270)
                ax[i,1].yaxis.set_label_position("right")

    shadowaxes = fig.add_subplot(111, xticks=[], yticks=[], frame_on=False)
    shadowaxes.set_ylabel(ylabel,fontsize = 45)
    shadowaxes.set_xlabel(r'$\log_{10}(\omega/\rm{eV})$',fontsize= 45)
    shadowaxes.xaxis.labelpad=50
    shadowaxes.yaxis.labelpad=50
    
    plt.savefig(filename,dpi = 1500)
    #plt.show()
    
    
if __name__ == "__main__":
    coupling_types = ['photon','electron','gluon']
    coupling_orders = ['linear','quad']
    
    # Extragalactic
    R_EG = 1e7
    E_EG = 1
    
    # Galactic
    R_GC = 1e4
    E_GC = 1e-2
    
    start_time = time.time()
    for i in coupling_types:
        for j in coupling_orders:
            plots(R_GC,E_GC,i,j)
            plots(R_EG,E_EG,i,j)
            
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")