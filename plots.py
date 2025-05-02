import numpy as np

COLORLIST = ["tab:red", "tab:orange", "tab:purple"]

def setup_axes(ax, formatter, coupling_order):
    """ Set up axes for subplot (i, j) """
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)
    ax.set_xticks(np.logspace(-20,-6,7))
    ax.set_xlim(.3e-20,0.9e-6)
    if coupling_order == 'linear': 
        ax.set_ylim(1e-9,0.9e0)
    ax.tick_params(direction="in")
    
def add_label(ax, x, y, txt, rotation = 0, fontsize = 25, 
              color = 'black', edgecolor = 'black', facecolor = 'white', 
              alpha = 1 , boxstyle = 'round,pad=0.1'):
    """Adds a text box label at position (x, y)
    
    Args:
        ax (): matplotlib axis
        x, y (float): coordinates to place label
        rotation (float): angle in degrees to rotate
        fontsize (int): font size
        color (str): color
        edgecolor (str): box edge color
        facecolor (str): box fill color
        alpha (float): box transparency
        boxstyle (str): box style
    """
    ax.text(x, y, txt, 
            rotation = rotation, 
            fontsize = fontsize,
            color = color,
            bbox = dict(facecolor = facecolor,
                        alpha = alpha,
                        edgecolor = edgecolor,
                        boxstyle = boxstyle
                        )
            )
    
def plot_MICROSCOPE(ax, range_x, microscope_m):
    """ Plot MICROSCOPE EP violation limits """
    ax.plot(range_x, microscope_m, color = 'gray', linewidth = 2)
    
    # Shade in region indicating parameter space excluded by MICROSCOPE EP tests
    upper_bound = np.full_like(range_x, 1e50, dtype=float)
    ax.fill_between(range_x, microscope_m, upper_bound, color = 'gray', alpha = 0.1)
    
    # Place label for MICROSCOPE
    pos_x = 3.5e-11
    pos_y = microscope_m[0]*1.3
    ax.text(pos_x, pos_y, r'${\rm MICROSCOPE}$', color = 'k')

def plot_FifthForce(ax, range_x, fifthForce_m):
    """ Plot fifth-force limits """
    ax.plot(range_x, fifthForce_m)

def plot_d_from_delta_t(ax, range_x, dday, ddt):
    """ Plot the detection delays of 1 day and integration time relative to a light-speed signal """
    ax.plot(range_x, dday, color = COLORLIST[2], linewidth = 2, linestyle = '--'  )
    ax.plot(range_x, ddt, color = COLORLIST[0], linewidth = 2, linestyle = '--'  )
    
def plot_fill_d_from_delta_t(ax, range_x, ddt_day1, ddt_day30, ddt1, ddt30):
    """ Shade in area between dilatonic coupling curves"""
    ax.fill_between(range_x, ddt_day1, ddt_day30, color = COLORLIST[2], alpha = 0.1)
    ax.fill_between(range_x, ddt1, ddt30, color = COLORLIST[0], alpha = 0.1)
    
def plot_fill_region(ax, fillregion_x, fillregion_y, coupling):
    """ Shade in the viable parameter space """
    ax.fill_between(fillregion_x, coupling, fillregion_y, where = coupling < fillregion_y, color = 'tab:green', alpha = 0.3)

def plot_coupling(ax, m_bench, coupling, wmp_contour):
    """ Plot the projected sensitivity of future experiments """
    ax.plot(m_bench*wmp_contour, coupling, c = 'k', linewidth = 2, alpha = 1)
    
def plot_crit_couplings(ax, range_x, d_earth, d_exp, d_atm):
    """ Plot the critical screening from the Earth, atmosphere, and experimental apparatus """
    ax.plot(range_x, d_earth, color = 'tab:blue', linewidth = 3)
    ax.plot(range_x, d_atm, color = 'tab:blue', linestyle = 'dashed')
    ax.plot(range_x, d_exp, color = 'tab:blue', linestyle = 'dotted')
    
    # Shade in the region above the critical coupling at Earth
    ax.fill_between(range_x, d_earth, 1e100, color = 'tab:blue', alpha = .05)

def plot_E_unc(ax, E_unc):
    """ Plot the scalar energy calculated from the uncertainty principle"""
    ax.axvline(E_unc, color = 'chocolate', linestyle = '--')
    
    # Shade in the region up to E_unc
    ymin = 1e-50
    ymax = 1e50
    ax.fill_betweenx([ymin, ymax], x1=ymin, x2=E_unc, color = 'chocolate', alpha = 0.1)
    
def plot_mass_exclusion(ax, m, coupling_order):
    """ Plot region excluded due to the scalar energy being less than its mass """
    ax.axvline(m, c = 'k', linestyle = '--')

    # Shade in the region up to the mass constraint
    if coupling_order == "linear":
        min_y = 1e-50
    elif coupling_order == "quad":
        min_y = 1e-10 # NOTE - how is this decided?
    ax.fill_between([1e-30, m], min_y, 1e50, facecolor = 'none', hatch = "/", edgecolor = 'k', alpha = 0.3)
    
def label_omega_lt_mass(ax, m, coupling_order):
    """ Label region in parameter space where omega < scalar field mass """
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