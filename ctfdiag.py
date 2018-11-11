import sys
import numpy as np
from scipy.constants import physical_constants
from scipy.ndimage import zoom

from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import interp1d

import re

import os

import mrcfile

from bokeh import plotting as bplt
from bokeh import models as bmodels
from bokeh import layouts as blayouts
from bokeh import io as bio

def ctf(omega, defocus, lam, phase_shift, spherical_aberration, volta_phase_plate_spot_size):
    """Compute the CTF for a volta phase plate.

    Evaluates the CTF at `omega`, returning an array of shape `omega.shape[0:-1]`

    Parameters
    ----------
    omega : `numpy.ndarray` of shape (..., 2) or (...,)
        Spatial frequencies at which to evaluate the CTF

        If `omega.shape[-1]` is 2, then interpret `omega` as spatial
        frequencies `(omega_x, omega_y)`, otherwise, interpret `omega`
        as the modulus of spatial frequencies.

        In units of 1/A
    lam : float
        Electron wavelength

        In units of m
    phase_shift : float
        Phase plate shift

        In radians
    spherical_aberration : float
        Spherical aberration

        In units of mm
    volta_phase_plate_spot_size : float
        Phase plate spot size
    
        In units of nm

    Returns
    -------
    `numpy.ndarray` of shape `omega.shape[0:-1]`
        CTF evaluated at requested points

    """

    # Convert everything to Angstroms
    lam_A = lam * 1e10
    spherical_aberration_A = spherical_aberration * 10e-7

    # Convert 1/nm -> 1/A
    volta_phase_plate_spot_size_1A = volta_phase_plate_spot_size / 10

    if omega.shape[-1] == 2:
        omega_abs = np.sqrt(np.sum(omega*omega, axis=-1))
    else:
        omega_abs = omega

    gamma = (-phase_shift*(1 - np.exp(-(omega_abs**2)/(2*(volta_phase_plate_spot_size_1A**2)))) +
             np.pi*(-defocus*lam_A*(omega_abs**2) + 0.5*(spherical_aberration_A)*(lam_A**3)*(omega_abs**4)))

    return np.sin(gamma)

def ctf_2d(omega, defocus_U, defocus_V, astigmatism, amplitude_contrast, lam, phase_shift, spherical_aberration, volta_phase_plate_spot_size):
    """Compute the CTF for a volta phase plate.

    Evaluates the CTF at `omega`, returning an array of shape `omega.shape[0:-1]`

    Parameters
    ----------
    omega : `numpy.ndarray` of shape (..., 2) 
        Spatial frequencies at which to evaluate the CTF

        In units of 1/A
    defocus_U, defocus_V : float
        Maximal and minimal defocus
      
        In units of nm
    amplitude_contrast : float
        Amplitude contrast

        Unitless, in [0, 1]
    astigmatism : float
        Astigmatistm angle

        In units of radians
    lam : float
        Electron wavelength

        In units of m
    phase_shift : float
        Phase plate shift

        In radians
    spherical_aberration : float
        Spherical aberration

        In units of mm
    volta_phase_plate_spot_size : float
        Phase plate spot size
    
        In units of nm

    Returns
    -------
    `numpy.ndarray` of shape `omega.shape[0:-1]`
        CTF evaluated at requested points

    """

    # Convert everything to Angstroms
    lam_A = lam * 1e10
    spherical_aberration_A = spherical_aberration * 10e-7

    # Convert 1/nm -> 1/A
    volta_phase_plate_spot_size_1A = volta_phase_plate_spot_size / 10

    omega_abs = np.sqrt(np.sum(omega*omega, axis=-1))

    # Compute defocus term with astigmatism
    thetas = np.arctan(omega[..., 1], omega[..., 0])

    defocus = defocus_U*(np.cos(thetas - astigmatism)**2) + defocus_V*(np.sin(thetas - astigmatism)**2)
        
    gamma = (phase_shift*(1 - np.exp(-(omega_abs**2)/(2*(volta_phase_plate_spot_size_1A**2)))) +
             np.pi*(defocus*lam_A*(omega_abs**2) - 0.5*(spherical_aberration_A)*(lam_A**3)*(omega_abs**4)))

    return -np.sqrt(1 - amplitude_contrast**2)*np.sin(gamma) - amplitude_contrast*np.cos(gamma)

def get_electron_wavelength(voltage):
    """Compute the relativistic wavelength for an electron.

    Parameters
    ----------
    voltage : float
        Accelerating voltage of potential applied to the electron.

        In kilovolts

    Returns
    -------
    float
        Wavelength in meters
    """

    # Electron energy in electronvolts
    eV = 1e3*voltage*physical_constants['electron volt'][0]
    h = physical_constants['Planck constant'][0]
    m = physical_constants['electron mass'][0]
    c = physical_constants['speed of light in vacuum'][0]

    return h / np.sqrt(2*m*eV*(1 + eV/(2*m*c**2)))

def estimate_power_spectrum(image, periodigram_box_size, periodigram_overlap):
    """Estimate the 2D power spectrum of image by averaging periodigrams.
    """

    periodigram_box_ys = np.arange(0, image.shape[1], int(periodigram_box_size * periodigram_overlap))
    periodigram_box_xs = np.arange(0, image.shape[0], int(periodigram_box_size * periodigram_overlap))

    average_pw = np.zeros((periodigram_box_size, periodigram_box_size))
    count = 0

    for i in range(periodigram_box_ys.shape[0]):
        if periodigram_box_ys[i] + periodigram_box_size >= image.shape[0]:
            continue
        for j in range(periodigram_box_xs.shape[0]):
            if periodigram_box_xs[j] + periodigram_box_size >= image.shape[1]:
                continue

            average_pw += np.abs((1/(periodigram_box_size**2))*np.fft.fftshift(np.fft.fft2(image[periodigram_box_ys[i]:periodigram_box_ys[i]+periodigram_box_size,
                                                                                                 periodigram_box_xs[j]:periodigram_box_xs[j]+periodigram_box_size])))**2
            count += 1

    return average_pw / count

def rotationally_average_power_spectrum(power_spectrum, pixel_size, num_omegas):
    wys, wxs = np.meshgrid(np.linspace(-1/(2*pixel_size), 1/(2*pixel_size), power_spectrum.shape[0], endpoint=False),
                           np.linspace(-1/(2*pixel_size), 1/(2*pixel_size), power_spectrum.shape[1], endpoint=False),
                           indexing='ij')

    power_spectrum_f = RegularGridInterpolator((np.linspace(-1/(2*pixel_size), 1/(2*pixel_size), power_spectrum.shape[0], endpoint=False),
                                                np.linspace(-1/(2*pixel_size), 1/(2*pixel_size), power_spectrum.shape[1], endpoint=False)),
                                               power_spectrum, bounds_error=False, fill_value=0)

    omegas = np.linspace(0, min(wys.max(), wxs.max()), num_omegas, endpoint=False)

    average_ps_1d = np.zeros(omegas.shape)

    for i in range(omegas.shape[0]):
        num_thetas = int(np.ceil(2*np.pi*(np.ceil(omegas[i]/pixel_size)))) + 1
        thetas = np.linspace(0, 2*np.pi, num_thetas, endpoint=False)
        sample_points = np.array((omegas[i]*np.sin(thetas), omegas[i]*np.cos(thetas))).T
        average_ps_1d[i] = power_spectrum_f(sample_points).mean()

    return omegas, average_ps_1d

def parse_ctf_logs(data_dir):
    """Parse a directory containing CTF log files.

    Log files are expected to have the form::
    
        Micrographname_\d\d\d\d(_.+)?.[txt|log]

    This method will read both GCTF logs (group 1 = _gctf.log and
    _EPA.log) and CTFFIND (group 1 = .txt and _avrot.txt)

    Parameters
    ----------
    data_dir : str
        directory containing CTF log files

    Returns
    -------
    dict
        the micrograph data, indexed by micrograph number
    """

    micrograph_data = {}

    for filename in os.listdir(data_dir):
        if not filename.endswith(('.txt', '.log')):
            continue
        try:
            match = re.search(r'(.*?_(\d\d\d\d))', filename)
            micrograph_name = match.group(1)
            micrograph_number = int(match.group(2))
        except:
            # Skip this file
            continue

        print('Processing {}'.format(filename))

        if micrograph_name not in micrograph_data:
            micrograph_data[micrograph_name] = {}

        micrograph_datum = micrograph_data[micrograph_name]
            
        if filename.endswith('_gctf.log'):
            # Read GCTF defocus from the last line in the file with
            # 'FINAL VALUES'
            # resolution limit in last line with RES_LIMIT
            with open(os.path.join(data_dir, filename), 'r') as gctf_logfile:
                for line in reversed(gctf_logfile.readlines()):
                    if 'RES_LIMIT' in line:
                        line = line.split()
                        micrograph_datum['ctf_res_GCTF'] = float(line[-1])
                        
                    elif 'Final Values' in line:
                        line = line.split()
                        print(line)
                        defocusU = float(line[0])
                        defocusV = float(line[1])
                        astig = float(line[2])
                        phase = float(line[3])

                        micrograph_datum['defocusU_GCTF'] = defocusU
                        micrograph_datum['defocusV_GCTF'] = defocusV
                        micrograph_datum['astigmatism_GCTF'] = astig
                        micrograph_datum['VPP_shift_GCTF'] = phase
                        
                        break
        elif filename.endswith('_EPA.log'):
            # Read GCTF CTF plot, which starts in the second line of
            # the file
            epa = []
            
            with open(os.path.join(data_dir, filename), 'r') as gctf_epafile:
                for line in gctf_epafile.readlines()[1:]:
                    line = line.split()
                    q = float(line[0])
                    ctfsim = float(line[1])
                    epa_all = float(line[2])
                    epa_fg = float(line[3])

                    # GCTF reports reciprocal resolutions, so invert it
                    epa.append((1/q, ctfsim, epa_all, epa_fg))

            micrograph_datum['EPA_GCTF'] = epa
            
        elif filename.endswith('_avrot.txt'):
            # CTFFIND produces text files with this format
            #1 - spatial frequency (1/Angstroms);
            #2 - 1D rotational average of spectrum (assuming no astigmatism);
            #3 - 1D rotational average of spectrum;
            #4 - CTF fit;
            #5 - cross-correlation between spectrum and CTF fit;
            #6 - 2sigma of expected cross correlation of noise

            # data starts on line 6 of the file
            with open(os.path.join(data_dir, filename), 'r') as ctffind_avrot:
                lines = ctffind_avrot.readlines()
                qs = [ float(q) for q in lines[5].split() ]
                avrot_noastig = [ float(pw) for pw in lines[6].split() ]
                avrot = [ float(pw) for pw in lines[7].split() ]
                ctffit = [ float(ctf) for ctf in lines[8].split() ]
                ctfcorr = [ float(corr) for corr in lines[9].split() ]
                ctfsigma = [ float(sig) for sig in lines[10].split() ]

                micrograph_datum['avrot_CTFFIND'] = (qs, avrot_noastig, avrot, ctffit, ctfcorr, ctfsigma)
                
        elif filename.endswith('.txt'):
            # Assume this is a CTFFIND log file
            # Columns: #1 - micrograph number;
            #2 - defocus 1 [Angstroms];
            #3 - defocus 2;
            #4 - azimuth of astigmatism;
            #5 - additional phase shift [radians];
            #6 - cross correlation;
            #7 - spacing (in Angstroms) up to which CTF rings were fit successfully

            # data starts on line 5

            with open(os.path.join(data_dir, filename), 'r') as ctffind_file:
                line = ctffind_file.readlines()[5].split()

                defocusU = float(line[1])
                defocusV = float(line[2])
                astig = float(line[3])
                phase = float(line[4])
                corr = float(line[5])
                res = float(line[6])

                micrograph_datum['defocusU_CTFFIND'] = defocusU
                micrograph_datum['defocusV_CTFFIND'] = defocusV
                micrograph_datum['astig_CTFFIND'] = astig
                micrograph_datum['VPP_shift_CTFFIND'] = phase
                micrograph_datum['ctf_total_corr_CTFFIND'] = corr
                micrograph_datum['ctf_res_CTFFIND'] = res

    return micrograph_data
    
def parse_gctf_logs(data_dir):
    """Parse a directory containing GCTF log files

    Parameters
    ----------
    data_dir : str
        directory containing GCTF log files

    Returns
    -------
    list
        the micrograph data
    """

    micrograph_data = []
    
    print('Reading GCTF logs from {}'.format(data_dir))

    for root_name,dir_names,file_names in os.walk(data_dir):
        for file_name in sorted(file_names):
            if not file_name.endswith('_gctf.log'):
                continue

            try:
                micrograph_number = re.search(r'.*?_(\d+)_gctf.log',
                                              file_name).group(1)
            except AttributeError:
                raise ValueError('Expected GCTF log filenames ending in '
                                 '_micrographnumber_gctf.log, '
                                 'but got {}'.format(file_name))        

            with open(os.path.join(root_name, file_name), 'r') as gctf_log_file:
                for line in reversed(gctf_log_file.readlines()):
                    if 'Final Values' in line:
                        line = line.split()
                        defocusU = float(line[0])
                        defocusV = float(line[1])
                        astig = float(line[2])
                        phase = float(line[3])

                        # print('Read {}, {}, {}, '
                        #       '{} from {}'.format(
                        #           defocusU,
                        #           defocusV,
                        #           astig,
                        #           phase,
                        #           file_name))

                        # Read EPA file too
                        suffix_idx = file_name.find('_gctf.log')
                        base_name = file_name[0:suffix_idx]

                        omega_abs =[]
                        EPA = []
                        CTF = []

                        with open(os.path.join(root_name, base_name + '_EPA.log'), 'r') as gctf_epa_file:
                            for line in gctf_epa_file.readlines():
                                line = line.split()

                                try:
                                    omega_abs.append(1/float(line[0]))
                                    CTF.append(float(line[1]))
                                    EPA.append(float(line[2]))
                                except ValueError:
                                    # Skip text lines
                                    continue

                        micrograph_data.append((file_name,
                                                defocusU,
                                                defocusV,
                                                astig,
                                                phase,
                                                omega_abs,
                                                EPA,
                                                CTF))

                        break


    return micrograph_data

data_dir = sys.argv[1]
micrograph_data = parse_ctf_logs(data_dir)

phase_shift_source = bmodels.ColumnDataSource(data=dict(micrograph_number=[],phase_shift_gctf=[],phase_shift_ctffind=[],phase_shift_90=[],astigmatism_gctf=[],defocusU_gctf=[],defocusV_gctf=[],astigmatism_ctffind=[],defocusU_ctffind=[],defocusV_ctffind=[]))

ctf_source = bmodels.ColumnDataSource(data=dict(omega_abs=[],ctf_values=[], epa=[], ctffind=[], ctffind_ctf=[]))

micrograph_ps_source = bmodels.ColumnDataSource(data=dict(image=[]))

gctf_info_source = bmodels.ColumnDataSource(data=dict(param=["Defocus U (nm)", "Defocus V (nm)", "Astigmatism (deg)", "Phase shift (deg)"], values=[0, 0, 0, 0]))

cur_micrograph_number = -1

num_omegas = 500

phase_shift_fig = bplt.figure(plot_height=800, plot_width=600, title='Phase shift due to Phase Plate', tools="tap")

phase_shift_hover = bmodels.HoverTool(tooltips=[("micrograph number", "@micrograph_number"),
                                                ("phase shift", "@phase_shift"),
                                                ("astigmatism", "@astigmatism"),
                                                ("defocus U", "@defocusU"),
                                                ("defocus V", "@defocusV")])

phase_shift_fig.add_tools(phase_shift_hover)

phase_shift_fig.xaxis.axis_label = "Micrograph number"
phase_shift_fig.yaxis.axis_label = "Phase shift, degrees"

ctf_fig = bplt.figure(plot_width=1200, title="CTF", tools="ywheel_zoom", y_range=(0,1))

ctf_fig.extra_y_ranges = {"epa": bmodels.Range1d(start=10, end=20)}
ctf_fig.add_layout(bmodels.LinearAxis(y_range_name="epa"), 'right')

# print('Reading {}'.format(sys.argv[1]))

# micrograph = mrcfile.open(sys.argv[1], 'r', permissive=True)

# micrograph_ps = estimate_power_spectrum(micrograph.data, 512, 0.5)

pixel_size = 0.66

# wys, wxs = np.meshgrid(np.linspace(-1/(2*pixel_size), 1/(2*pixel_size), 512, endpoint=False),
#                        np.linspace(-1/(2*pixel_size), 1/(2*pixel_size), 512, endpoint=False),
#                        indexing='ij')

# mean = micrograph_ps[ np.sqrt(wys**2 + wxs**2) > 1/50 ].mean()

# micrograph_ps[ np.sqrt(wys**2 + wxs**2) < 1/50 ] = mean

micrograph_ps_fig = bplt.figure(plot_height=512,
                                plot_width=512,
                                 x_range=(0, 512),
                                 y_range=(0, 512),
                                 title="Micrograph FFT")

micrograph_ps_fig.image(image='image', x=0, y=0, dw=512, dh=512, source=micrograph_ps_source)

# omegas, average_ps_1d = rotationally_average_power_spectrum(micrograph_ps, 0.66, num_omegas)
# print(average_ps_1d.shape)

phase_shift_slider = bmodels.widgets.Slider(start=0, end=181, value=0, step=.1, title="Phase plate shift")
defocus_slider = bmodels.widgets.Slider(start=500, end=50000, value=5000, step=1, title="Defocus in Angstroms")
pixel_size_slider = bmodels.widgets.Slider(start=0.33, end=2, value=0.66, step=.01, title="Pixel size")

columns = [ bmodels.widgets.TableColumn(field="param", title="GCTF info"),
            bmodels.widgets.TableColumn(field="values", title="") ]

gctf_info = bmodels.widgets.DataTable(source=gctf_info_source, columns=columns)

def update_plot(new_micrograph_number=None):
    micrograph_names = list(micrograph_data.keys())
    micrograph_names.sort()

    # Update phase shift plot
    defocusU_GCTF = [micrograph_data[i]['defocusU_GCTF'] for i in micrograph_names]
    defocusV_GCTF = [micrograph_data[i]['defocusV_GCTF'] for i in micrograph_names]
    astigmatism_GCTF = [micrograph_data[i]['astigmatism_GCTF'] for i in micrograph_names]

    phase_shift_GCTF = [micrograph_data[i]['VPP_shift_GCTF'] for i in micrograph_names]
    # CTFFIND phase shift is in radians
    phase_shift_CTFFIND = [micrograph_data[i]['VPP_shift_CTFFIND']/np.pi*180 if 'VPP_shift_CTFFIND' in micrograph_data[i] else 'NaN' for i in micrograph_names]
    
    phase_shift_source.data = dict(micrograph_number=np.arange(len(micrograph_data)),
                                   phase_shift_gctf=phase_shift_GCTF,
                                   phase_shift_ctffind=phase_shift_CTFFIND,
                                   phase_shift_90=[90 for d in micrograph_data],
                                   defocusU_gctf=defocusU_GCTF,
                                   defocusV_gctf=defocusV_GCTF,
                                   astigmatism_gctf=astigmatism_GCTF)
    
    # # Update power spectrum plot
    # if new_micrograph_number is not None:
    #     print('Changing micrograph display to {}'.format(new_micrograph_number))
    #     micrograph_base_name = micrograph_data[new_micrograph_number][0][0:micrograph_data[new_micrograph_number][0].find('_gctf.log')]

    #     micrograph_ps_name = os.path.join(data_dir, micrograph_base_name) + '.ctf'
    #     print('Trying to load {}'.format(micrograph_ps_name))
    #     micrograph_ps_mrc = mrcfile.open(micrograph_ps_name, 'r', permissive=True)
    #     print('Loaded {}, {}'.format(micrograph_ps_name, micrograph_ps_mrc.data.shape))
    #     micrograph_ps_source.data = dict(image=[micrograph_ps_mrc.data[0,...]])

    #     micrograph_ps_fig.title.text = "Micrograph {} Power Spectrum".format(new_micrograph_number)
    
    # # Update CTF plot
    ctf_dict = ctf_source.data

    lam = get_electron_wavelength(300)
    volta_phase_plate_spot_size = .05
    spherical_aberration = 2.7
    
    # Compute omega range, out to Nyquist
    omega_abs = np.linspace(0, (1/(2*pixel_size)), num_omegas, endpoint=False)

    if new_micrograph_number is not None:
        md = micrograph_data[micrograph_names[new_micrograph_number]]

        defocus = 0.5*(md['defocusU_GCTF'] + md['defocusV_GCTF'])
        phase_shift = md['VPP_shift_GCTF']
        defocus_slider.value = defocus
        phase_shift_slider.value = phase_shift

        qs = [q[0] for q in md['EPA_GCTF']]
        epas = np.array([epa[1] for epa in md['EPA_GCTF']])
        epas = epas / epas.max()

        epa_f = interp1d(qs, epas, bounds_error=False)
        epa = epa_f(omega_abs)
        ctf_fig.extra_y_ranges['epa'].start = np.nanmean(epa[10:])-.1
        ctf_fig.extra_y_ranges['epa'].end = np.nanmean(epa[10:])+.1

        qs = md['avrot_CTFFIND'][0]
        avrot = md['avrot_CTFFIND'][1]

        avrot_f = interp1d(qs, avrot, bounds_error=False)
        avrot = avrot_f(omega_abs)

        ctffind_ctf = md['avrot_CTFFIND'][3]
        ctffind_ctf_f = interp1d(qs, ctffind_ctf, bounds_error=False)
        ctffind_ctf = ctffind_ctf_f(omega_abs)
        
        gctf_info_source.data = dict(param=["Defocus U (nm)", "Defocus V (nm)", "Astigmatism (deg)", "Phase shift (deg)"],
                                     values=[md['defocusU_GCTF'],
                                             md['defocusV_GCTF'],
                                             md['astigmatism_GCTF'],
                                             md['VPP_shift_GCTF']])
    else:
        defocus = defocus_slider.value
        phase_shift = phase_shift_slider.value
        epa = ctf_source.data['epa']
        if len(epa) == 0:
            epa = ['NaN' for i in range(len(omega_abs))]
        avrot = ctf_source.data['ctffind']
        if len(avrot) == 0:
            avrot = ['NaN' for i in range(len(omega_abs))]
        ctffind_ctf = ctf_source.data['ctffind_ctf']
        if len(ctffind_ctf) == 0:
            ctffind_ctf = ['NaN' for i in range(len(omega_abs))]

    phase_shift = phase_shift/180*np.pi
    
    #pixel_size = pixel_size_slider.value
            
    print('Update CTF, defocus: {}, phase shift: {}'.format(defocus, phase_shift))
    ctf_values = np.abs(ctf(omega_abs, defocus, lam, phase_shift, spherical_aberration, volta_phase_plate_spot_size))

    ctf_source.data = dict(omega_abs=omega_abs, ctf_values=ctf_values, epa=epa, ctffind=avrot, ctffind_ctf=ctffind_ctf)

widgets = [phase_shift_slider, defocus_slider]
for widget in widgets:
    widget.on_change('value', lambda attr, old, new: update_plot())

# Widgets without callbacks
widgets.append(gctf_info)

phase_shift_glyphs = phase_shift_fig.circle(x='micrograph_number', y='phase_shift_gctf', size=6, source=phase_shift_source)
phase_shift_glyphs = phase_shift_fig.circle(x='micrograph_number', y='phase_shift_ctffind', size=6, source=phase_shift_source, line_color='red')
phase_shift_fig.line(x='micrograph_number', y='phase_shift_90', source=phase_shift_source, line_color='orange')

ctf_fig.line(x='omega_abs', y='ctf_values', source=ctf_source)
ctf_fig.line(x='omega_abs', y='epa', y_range_name='epa', source=ctf_source, line_color='orange')
ctf_fig.line(x='omega_abs', y='ctffind', source=ctf_source, line_color='green')
ctf_fig.line(x='omega_abs', y='ctffind_ctf', source=ctf_source, line_color='red')
ctf_fig.xaxis.formatter = bmodels.FuncTickFormatter(code="""
return "1/" + (1/tick).toFixed(4);
""")

def phase_shift_on_select(attr, old, new):
    if len(new['1d']['indices']) > 0:
        new_micrograph_number = new['1d']['indices'][0]
        print("Selected micrograph {}".format(new_micrograph_number))
        update_plot(new_micrograph_number)

phase_shift_glyphs.data_source.on_change('selected', phase_shift_on_select)

layout = blayouts.layout([[phase_shift_fig, micrograph_ps_fig],[ctf_fig, blayouts.widgetbox(*widgets)]])

# Initial setup of the plot
update_plot()

bio.curdoc().add_root(layout)
