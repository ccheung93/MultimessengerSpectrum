import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import time

from propagation import *
from limits import *
from plots import *

COLORLIST = ["tab:red", "tab:orange", "tab:purple"]

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

def get_K_params(coupling_type, coupling_order):
    """ Set parameters based on coupling_type

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

def plot_time_labels(ax, m, omega_over_m_dt, omega_over_m_day, coupling_order):
    bbox_style = lambda color: dict(facecolor = 'white', alpha = 1, edgecolor = color, boxstyle = 'round,pad=.1')
    
    # Define time labels 
    time_labels = [
        {
            "omega_over_m": omega_over_m_dt, 
            "color": "tab:red", 
            "txt": r'$\delta t\, \gtrsim \, 1~{\rm yr}~ \uparrow $' # dt >~ 1 year
            },
        {
            "omega_over_m": omega_over_m_day, 
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
        pos_x = m*label["omega_over_m"]/4
        pos_y = pos_y_coupling[coupling_order]
        txt = label["txt"]
        ax.text(pos_x, pos_y, txt, rotation = 90, fontsize = 25, color = label["color"], bbox = bbox_style(label["color"]))

def plot_supernova(ax, Elist, coupling_type):
    supernova_config = {
        "photon": {
            "ylim": (.5e6, 8e32),
            "txt_y": 3e29,
            "lbl_y": 3e31,
            "txt": r'${\rm Supernova}~\gamma \gamma \rightarrow \phi \phi$',
            "line": [d_from_Lambda(1e12, 2)] * len(Elist)
        },
        "electron": {
            "ylim": (.5e9, 5e33),
            "txt_y": 3e29,
            "lbl_y": 1e32,
            "txt": r'${\rm Supernova}~e^+ e^- \rightarrow \phi \phi$',
            "line": [5e31] * len(Elist)
        },
        "gluon": {
            "ylim": (.5e4, 5e30),
            "txt_y": 3e26,
            "lbl_y": 3e29,
            "txt": r'${\rm Supernova}~N N \rightarrow N N \phi \phi$',
            "line": [d_from_Lambda(15e12, 2)] * len(Elist)
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
    """ Plot omega*t_star <~ 2*pi """
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
            txt = r'$\omega\,t_{*} \lesssim \, 2\pi$' # omega t_star <~ 2pi
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

def label_critical_screening(ax, K_E, K_atm, coupling_type, filename):
    """ Label critical screening lines """
    distance_scales = ["10Mpc_", "10kpc_"]
    
    # Define labels for critical screening
    crit_screening_labels = {
        "photon": {
            "eth": r'$d_{e,{\rm crit}}^{(2)\, \rm\oplus}$',
            "atm": r'$d_{e,{\rm crit}}^{(2)\, \rm atm}$',
            "exp": r'$d_{e,{\rm crit}}^{(2)\, \rm app}$'
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

def linear_plot(ax, i, j, coupling, m, Elist, t, dday, ddt, wm_dt, wm_day, Microscope_m, FifthForce_m, E_unc, m_bench, wmp_contour, coupling_type, filename):
    """ Plots for linear coupling_order """

    plot_MICROSCOPE(ax, Elist, Microscope_m)
    plot_E_unc(ax, E_unc)
    plot_FifthForce(ax, Elist, FifthForce_m)
    plot_coupling(ax, m_bench, coupling, wmp_contour)
    plot_mass_exclusion(ax, m, 'linear')
    label_omega_lt_mass(ax, m, 'linear')
    
    condition_mask = (Elist > E_unc) & (Elist > m * wm_day)
    fillregion_x = Elist[condition_mask]
    coupling_fill = coupling[condition_mask]
    fillregion_y = [Microscope_m[l] for l in range(len(fillregion_x))]
    plot_fill_region(ax, fillregion_x, fillregion_y, coupling_fill)

    plot_d_from_delta_t(ax, Elist, dday, ddt)
    
    plot_time_labels(ax, m, wm_dt, wm_day, 'linear')
    plot_omega_ts(ax, E_unc, filename)
    plot_parameter_list(ax, i, j, coupling_type, 'linear', filename)

def quad_plot(ax, i, j, coupling, m, Elist, d_screen_earth, d_screen_exp, d_screen_atm, dday1, ddt1, dday30, ddt30, wm_dt, wm_day, R, E_unc, m_bench, wmp_contour, K_E, K_atm, coupling_type, filename):
    """ Plots for quadratic coupling_order """
    
    plot_crit_couplings(ax, Elist, d_screen_earth, d_screen_exp, d_screen_atm)
    plot_E_unc(ax, E_unc)
    plot_coupling(ax, m_bench, coupling, wmp_contour)
    plot_mass_exclusion(ax, m, 'quad')
    label_omega_lt_mass(ax, m, 'quad')
    
    plot_d_from_delta_t(ax, Elist, dday30, ddt30)
    
    condition_mask = Elist > E_unc
    fillregion_x = Elist[condition_mask]
    coupling_fill = coupling[condition_mask]
    d_exp = d_screen_exp[condition_mask]
    
    plot_supernova(ax, Elist, coupling_type)
    label_critical_screening(ax, K_E, K_atm, coupling_type, filename)
    
    if R < 1e5:
        dday30_fill = dday30[condition_mask]
        fillregion_y = np.minimum(d_exp, dday30_fill)
        
        dt_lbl_yr = r'$\delta t\, \gtrsim \, 1~{\rm yr}~\uparrow$'
        dt_lbl_day = r'$\delta t\, \gtrsim \, 1~{\rm day}~\uparrow$'
        color_yr = 'tab:red'
        color_day = 'tab:purple'
        if coupling_type == 'photon':
            add_label(ax, 3e-17, 1e27, dt_lbl_yr, rotation=37, color=color_yr, edgecolor=color_yr)
            add_label(ax, 6e-16, 9e26, dt_lbl_day, rotation=37, color=color_day, edgecolor=color_day)
        elif coupling_type == 'electron':
            add_label(ax, 3e-17, 8.5e26, dt_lbl_yr, rotation=39, color=color_yr, edgecolor=color_yr)
            add_label(ax, 6e-16, 8e26, dt_lbl_day, rotation=39, color=color_day, edgecolor=color_day)
        elif coupling_type == 'gluon':
            ax.set_ylim(.5e5,8e30)
            add_label(ax, 3e-17, 6e23, dt_lbl_yr, rotation=38, color=color_yr, edgecolor=color_yr)
            add_label(ax, 6e-16, 5e23, dt_lbl_day, rotation=38, color=color_day, edgecolor=color_day)
    else:
        plot_d_from_delta_t(ax, Elist, dday1, ddt1)
    
        dday_fill = dday1[condition_mask]
        fillregion_y = np.minimum(d_exp, dday_fill)
        
        plot_fill_d_from_delta_t(ax, Elist, dday1, dday30, ddt1, ddt30)
        plot_time_labels(ax, m, wm_dt, wm_day, 'quad')
        
    plot_fill_region(ax, fillregion_x, fillregion_y, coupling_fill)
    plot_parameter_list(ax, i, j, coupling_type, 'quad', filename)

def plots(R, Etot, coupling_type, coupling_order, save_plots=True, show_plots=True):
    """Generate dilatonic coupling plots 

    Args:
        R (float): distance between the source and the experiment
        Etot (float): total energy of the burst
        coupling_type (str): type of coupling ('photon', 'electron' or 'gluon')
        coupling_order (str): coupling order ('linear' or 'quadratic')
    """
    
    # Benchmark parameters
    mass_benchmarks = [1e-21, 1e-18] # mass benchmarks in eV
    ts_benchmarks = [1, 1e2] # burst duration benchmarks in seconds
    m_bench = 1e-21 # in eV
    
    mass = [[mass_benchmarks[0], mass_benchmarks[1]],
            [mass_benchmarks[0], mass_benchmarks[1]]]
    
    ts = [[ts_benchmarks[0], ts_benchmarks[0]],
          [ts_benchmarks[1], ts_benchmarks[1]]]
    
    # Set up general parameters
    distance_label = get_distance_label(R) 
    filename = f"{distance_label}_{coupling_type}_{coupling_order}_dilatoniccoupling.pdf"
    wmp_contour = np.logspace(0,30,1000)
    dt = 1 * 3.154e7 
    Elist = mass[0][0]*wmp_contour

    K_space, K_E, K_atm, eta, ylabel = get_K_params(coupling_type, coupling_order)

    # Load constraints for linear coupling order
    if coupling_order == "linear":
        Microscope_m, FifthForce_m = load_linear_constraints(Elist)

    # Setup plot
    fig, ax = plt.subplots(2, 2, figsize = (30, 21), sharex = True, sharey = True)
    plt.rcParams.update({
        'mathtext.fontset': 'cm',
        'font.size': 35,
        'font.family': 'STIXGeneral',
        'hatch.color': 'lightgray'
    })
    plt.subplots_adjust(wspace = 0, hspace = 0)

    ax[0,0].set_yscale('log')
    ax[0,0].set_xscale('log')
    
    # Set tick labels on axes to be in log10
    formatter = FuncFormatter(exponentlabel) 

    for i in range(2):
        for j in range(2):
            t = ts[i][j]
            m = mass[i][j]
            E_unc = E_from_uncert(t)
            axij = ax[i][j]
            
            setup_axes(axij, formatter, coupling_order)
            
            rho, rescaling_factor = signal_duration(Etot, m, Elist, t, R, 1)
            
            wm_dt = omegaoverm_noscreen(dt, R)
            wm_day = omegaoverm_noscreen(DAY_TO_SEC, R)
            
            if coupling_order == 'linear':
                coupling = d_probe(Elist, rho, rescaling_factor, eta, 1)
                
                Dg = 30e3
                dday30 = d2_from_delta_t(DAY_TO_SEC, R, m, Elist, Dg, K_space)  # NOTE - is there a d1_from_delta_t and shouldn't that be used here?
                ddt30 = d2_from_delta_t(dt, R, m, Elist, Dg, K_space)
                
                linear_plot(axij, i, j, coupling, m, Elist, t, dday30, ddt30, wm_dt, wm_day, Microscope_m, FifthForce_m, E_unc, m_bench, wmp_contour, coupling_type, filename)
                
            elif coupling_order == 'quad':
                coupling = d_probe(Elist, rho, rescaling_factor, eta, 2)

                d_screen_earth = d2_screen(Elist, R_E, RHO_E, m, K_E)
                d_screen_atm = d2_screen(Elist, R_ATM, RHO_ATM, m, K_atm)
                d_screen_exp = d2_screen(Elist, R_EXP, RHO_EXP, m, K_E)
                
                Dg = 1e3
                dday1 = d2_from_delta_t(DAY_TO_SEC, R, m, Elist, Dg, K_space)
                ddt1 = d2_from_delta_t(dt, R, m, Elist, Dg, K_space)
                
                Dg = 30e3
                dday30 = d2_from_delta_t(DAY_TO_SEC, R, m, Elist, Dg, K_space)
                ddt30 = d2_from_delta_t(dt, R, m, Elist, Dg, K_space)
                
                quad_plot(axij, i, j, coupling, m, Elist, d_screen_earth, d_screen_exp, d_screen_atm, dday1, ddt1, dday30, ddt30, wm_dt, wm_day, R, E_unc, m_bench, wmp_contour, K_E, K_atm, coupling_type, filename)

            # Subplot axis labels
            ax[0,j].set_title(r'$\log_{10}(m_{\phi}/{\rm eV}) = $'+str(int(np.log10(mass[0][j]))), pad = 20)
            ax[i,1].set_ylabel(r'$t_*$ = '+str(int(ts[i][0]))+r' s',labelpad = 40,rotation = 270)
            ax[i,1].yaxis.set_label_position("right")

    # Shared axis labels
    shadowaxes = fig.add_subplot(111, xticks=[], yticks=[], frame_on=False)
    shadowaxes.set_ylabel(ylabel, fontsize = 45)
    shadowaxes.set_xlabel(r'$\log_{10}(\omega/\rm{eV})$', fontsize= 45)
    shadowaxes.xaxis.labelpad=50
    shadowaxes.yaxis.labelpad=50
    
    if save_plots: plt.savefig(filename,dpi = 1500)
    if show_plots: plt.show()
    
    
if __name__ == "__main__":
    coupling_types = ['photon','electron','gluon']
    coupling_orders = ['linear','quad']
    
    # Extragalactic
    R_EG = 1e7 # pc
    E_EG = 1   # solar-masses
    
    # Galactic
    R_GC = 1e4
    E_GC = 1e-2
    
    # computations
    # Inputs:
    #   density profile
    #   spectrum emitted by source, e.g. momentum spectrum
    #   experimental properties, e.g. sensitivity, t_int
    
    # plot
    
    start_time = time.time()
    for i in coupling_types:
        for j in coupling_orders:
            plots(R_GC,E_GC,i,j)
            plots(R_EG,E_EG,i,j)
            
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")