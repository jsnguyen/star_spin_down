import copy
import multiprocessing
import struct
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import truncnorm
import tqdm

####################
# SIMULATION UNITS #
####################

# MASS -> SOLAR MASS
# LENGTH -> SOLAR RADII
# TIME -> YEAR
# weird units, but this is a LOT easier than converting everything to cgs and back

# find nearest index and value
def find_nearest(arr, val):
    arr = np.asarray(arr)
    i = (np.abs(arr - val)).argmin()
    return i, arr[i]

class Star:
        
    def __init__(self, lut):
        self.lut = lut
        self.age = None
        self.mass = None
        self.radius = None
        self.inertia = None
        self.period = None
        self.rot_vel = None
        self.ang_mom = None

    # look up table, the BHAC15 data
    def set_lut(self, lut):
        self.lut = lut

    # interpolate between ages since we don't necessarily land on a specific age between the star
    def interpolate_between_age(self, x, key):
        return np.interp(x, self.lut[self.mass]['age'], self.lut[self.mass][key])

    # set both the mass and age during initialization
    def set_mass_age(self, m, age):
        self.mass = m
        self.set_age(age)

    # changin the age will change inertia and radius
    def set_age(self, age):
        self.age = age
        self.inertia = self.interpolate_between_age(age, 'inertia')
        self.radius = self.interpolate_between_age(age, 'radius')

    # changing the period will change the rotational velocity and angular momentum
    def set_period(self, p):
        self.period = p
        self.rot_vel = 2*np.pi/p # units of radians per year
        self.ang_mom = self.inertia*self.rot_vel 

    # changing the angular momentum will change the rotational velocity and period
    def set_ang_mom(self, ang_mom):
        self.ang_mom = ang_mom
        self.rot_vel = self.ang_mom/self.inertia
        self.period = 2*np.pi/self.rot_vel # units of radians per year

    # calculate the wind torque according to Kawaler 1998, Matt 2015 empirical relations
    def calc_wind_torque(self):
        K = 6.924e-10 # 6.7e30 ergs in our solar units
        solar_rot_vel = 91.56  # rad / year
        return  - K * np.power(self.rot_vel/solar_rot_vel, 3) * np.power(self.mass, -0.5) * np.power(self.radius, 0.5)

    # calculate the change in angular momentum directly
    # this is the equation we are solving
    def calc_change_in_ang_mom(self, dt):
        inertia_old = self.interpolate_between_age(self.age, 'inertia')
        inertia_new = self.interpolate_between_age(self.age+dt, 'inertia')
        dy = inertia_new - inertia_old
        change_in_inertia = dy/dt
        return  (self.calc_wind_torque() - self.rot_vel * change_in_inertia)

    # advances the age of the star by some timestep
    def advance(self, dt):
        d_ang_mom = self.calc_change_in_ang_mom(dt)
        # dont allow adding angular momentum
        if d_ang_mom > 0:
            d_ang_mom = 0

        self.set_age(self.age+dt)
        self.set_ang_mom(self.ang_mom + d_ang_mom*dt)

    def __str__(self):
        return str('{:.3f} {:.3f} {:.3f} {:.3f} {:.3f}'.format(self.mass, self.inertia, self.period, self.rot_vel, self.ang_mom))

# read in the BHAC15 data, its a little messy and has lots of log space stuff
# convert everything from log units to not-log units
def read_in_BHAC15_data(filename):
    n_header = 46

    data = {}
    with open(filename, 'r') as f:

        for _ in range(n_header):
            f.readline()

        for line in f:
            l  = line.strip()

            if l == '':
                continue
            if l[0] == '!':
                continue

            l = [float(el) for el in l.split()]
            mass = l.pop(0)

            if mass not in data.keys():
                data[mass] = {'age': [],
                              't_eff': [],
                              'lum': [],
                              'log_g': [],
                              'radius': [],
                              'li_ratio': [],
                              't_cen': [],
                              'dens_cen': [],
                              'mass_radc': [],
                              'radius_radc': [],
                              'k2conv': [],
                              'k2rad': [],
                              'inertia': []}

            data[mass]['age'].append(np.power(10, l[0]))
            data[mass]['t_eff'].append(l[1])
            data[mass]['lum'].append(np.power(10, l[2]))
            data[mass]['log_g'].append(l[3])
            data[mass]['radius'].append(l[4])
            data[mass]['li_ratio'].append(np.power(10, l[5]))
            data[mass]['t_cen'].append(np.power(10, l[6]))
            data[mass]['dens_cen'].append(np.power(10, l[7]))
            data[mass]['mass_radc'].append(l[8])
            data[mass]['radius_radc'].append(l[9])
            data[mass]['k2conv'].append(l[10])
            data[mass]['k2rad'].append(l[11])
            data[mass]['inertia'].append(calc_inertia(mass, l[4], l[10], l[11])) # inertia is a derived quantity

    for mass in data.keys():
        for key in data[mass].keys():
            data[mass][key] = np.array(data[mass][key]) # convert to numpy arrays

        # convert ages to relative ages since we only need that
        for key in data[mass].keys():
            data[mass]['age'] = data[mass]['age'] - data[mass]['age'][0]


    return data

# calculate the inertia according to the equation found in the header to the BHAC15 data
def calc_inertia(mass, radius, k2rad, k2conv):
    k2_sq = k2conv*k2conv + k2rad*k2rad
    return k2_sq * mass * radius*radius

# unnormalized version of the kroupa 2013 IMF
def imf_kroupa2013_unnorm(mass):
    if 0.07 <= mass <= 0.5:
        return np.power((mass/0.07), -1.3)

    elif 0.5 < mass <= 150:
        return np.power((0.5/0.07), -1.3) * np.power((mass/0.5), -2.3)

    else:
        return 0

# multiprocesssing function for parallelization
# each process simulates the star's evolution individually
def mp_advance(i, s, dt, n_iter):
    output_filename = './data/star_{}.dat'.format(i)
    star_time_series = []
    star_time_series.append(copy.copy(s))
    for i in range(n_iter): 
        s.advance(dt)
        star_time_series.append(copy.copy(s))

    return star_time_series

def main():

    # LOAD BHAC15 MODEL DATA
    lut = read_in_BHAC15_data('BHAC15_tracks+structure')
    print('Model mass range: {}'.format(list(lut.keys())))

    # INSTANTIATE STARS WE ARE SIMULATING
    print('Instantiate stars...')
    n_stars = int(1e6) # set the total number of stars in the simulation HERE
    stars = []
    for i in range(n_stars):
        stars.append(Star(lut))

    # CALCULATE IMF
    print('Calculate IMF...')
    v_imf = np.vectorize(imf_kroupa2013_unnorm, otypes=[float])

    # load the kroupa2013 normalization constant
    # has to be calculated in 'calc_kroupa2013_imf_norm.py'
    with open('kroupa2013_norm.pickle', 'rb') as f:
        norm = pickle.load(f)

    width = 1e-3
    masses = np.arange(0.07, 150, width)
    pdf = v_imf(masses)/norm # don't forget to normalize!

    # double check the normalzation constant is okay
    # total area under IMF should be 1
    print('Integral of PDF: {:.6f}'.format(np.trapz(pdf, masses)))

    # cumulative density function
    # need this to sample the IMF
    cdf = np.zeros(masses.shape)
    for i,m in enumerate(masses):
        # this works b/c the last value in the array starts at zero anyways...
        cdf[i] += cdf[i-1]+(width*v_imf(m)/norm)

    # SAMPLE MASS DISTRIBUTION
    # sample the mass distribution using inverse function sampling
    print('Sample IMF...')
    mass_bins = np.arange(0.3, 1.4+0.1, 0.1)
    mass_bin_width = 0.1
    print('Sample IMF mass bins: {}'.format(mass_bins))

    star_ind = 0
    pbar = tqdm.tqdm(total = n_stars) # progress bar since this actually takes a while
    while star_ind < n_stars:
        
        random_val = np.random.uniform()
        nearest_index, _ = find_nearest(cdf, random_val)
        imf_rand_mass = masses[nearest_index]

        for mb in mass_bins:
            if mb-mass_bin_width/2 < imf_rand_mass <= mb+mass_bin_width/2:
                mb = np.round(mb, decimals=1)
                stars[star_ind].set_mass_age(mb, 0)
                star_ind += 1
                pbar.update(1)
                break

    pbar.close()
                
    # SAMPLE PERIOD DISTRIBUTION
    # just a simple truncated gaussian

    print('Sample period...')
    days_to_years = 1/365
    mean_period = 8 * days_to_years
    std_period = 6 * days_to_years
    lower_lim = 0.5 * days_to_years
    upper_lim = 18.5 * days_to_years
    a, b = (lower_lim - mean_period) / std_period, (upper_lim - mean_period) / std_period

    for s in tqdm.tqdm(stars):
        p_rvs = truncnorm.rvs(a, b, loc=mean_period, scale=std_period)
        s.set_period(p_rvs)
        
    # ACTUAL SIMULATION

    print('Running simulation...')

    # simulation parameters
    # n_iter * dt = total time

    n_proc = 64 # number of processes for multiprocessing
    n_iter = int(1.5e2)
    dt = 1e5

    # write header to file
    output_filename = 'stars.dat'
    with open(output_filename, 'wb') as f:
        f.write(struct.pack('>I', n_iter))
        f.write(struct.pack('>d', dt))
        f.write(struct.pack('>I', n_stars))

    # start multiprocessing loop, evolving stars through time
    with multiprocessing.Pool(processes=n_proc) as pool:
        res = pool.starmap(mp_advance, tqdm.tqdm([(i, s, dt, n_iter) for i,s in enumerate(stars)]))

        # use the result, stored in memory and write to file
        print('Writing to file...')
        with open(output_filename, 'ab') as f:
            for star_time_series in res:
                for frame in star_time_series:
                    f.write(struct.pack('>d', frame.age))
                    f.write(struct.pack('>d', frame.mass))
                    f.write(struct.pack('>d', frame.radius))
                    f.write(struct.pack('>d', frame.inertia))
                    f.write(struct.pack('>d', frame.period))
                    f.write(struct.pack('>d', frame.rot_vel))
                    f.write(struct.pack('>d', frame.ang_mom))

if __name__=='__main__':
    main()
