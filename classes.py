#astrofunctions final copy 
import matplotlib.pyplot as plt #plotting functions
import glob #global file inports for fits files
import scipy.signal # signal processing
import numpy as np #basic mathematical functions and utilities
import astropy.io.fits as pyfits #for specialisted astronomy functions
from scipy.signal import savgol_filter 
from scipy.signal import lombscargle
import pylab  #additional plotting functions
import pandas as pd # data management 
from scipy.optimize import curve_fit
import seaborn as sns
mykepler = '4'

files = glob.glob('Data/Object%slc/kplr*.fits'%(mykepler))

class smoothing:
    def __init__(self):
        self._test = 'testrun'
    def lightcurve_smoothing(*args):
        "Inputs: Directory containing kepler lightcurve data for a singular star, separated into multiple files "
        "with integer file numbering system, file type: .fits"
        "Function: combines files into singular DataFrane, having removed NaN values and outliers above the dataset"
        "Function smooths and flattens lightcurve, to aid detection of exoplanets"
        "Additionally, sorts all datapoints by time, to account for possible inconsistencies in data source's ordering "
            
        T_ALL = []          #establish empty arrays to use as temporary memory dumps for kepler data
        F_ALL = []          #to be combined into until it is sorted into a final DataFrame 
        ERR_ALL =[]
        for file_name in files:
            data = pyfits.getdata(file_name)    #get data from specified file 
            name = file_name[:-len('.fits')]    #aquire file name and remove .fits extension, name string can be used
                                                    # for identification during debugging or individual plotting
                                                                                

            TIME_RAW = data['TIME']             #separate data from .fits files into time flux, and error arrays
            FLUX_RAW = data['PDCSAP_FLUX']
            ERR_RAW = data['PDCSAP_FLUX_ERR']
            indexes = ~np.isnan(FLUX_RAW)       #use indexes to locate all values in flux that are 
            time_nan = TIME_RAW[indexes]        #not NaN values, and keep only these values and their
            FLUX_NAN = FLUX_RAW[indexes]        #corresponding time values
            err_nan = ERR_RAW[indexes]
            stdflux = np.nanstd(FLUX_NAN)       #calculate mean and standard deviation to remove
            fluxmean = np.nanmean(FLUX_NAN)     #outliers above 3 std above the mean
            condition = fluxmean + (stdflux)

            indices = np.where(FLUX_NAN>condition)  #remove outlier points which are one standard deviation above
            XPLOT = np.delete(time_nan, indices)    # the mean of the flux values
            Y_out = np.delete(FLUX_NAN, indices)
            err_out = np.delete(err_nan, indices)
                
            #use a savgol filter to normalise and smooth the data, reducing the impact of background noise
            #apply filter to data, and calculate normalised value around 1
            #window length of 1 selected so that noise is reduced, but transits are not reduced in this process.
            interp_savgol = savgol_filter( Y_out, window_length=501, polyorder=5)
            interp_savgolmean = (interp_savgol/(np.nanmean(interp_savgol)))
            divplot = Y_out/interp_savgol   #creating normalised datasets
            err_out = err_out/interp_savgol  #normalisation of data 
            T_ALL.extend(XPLOT)
            F_ALL.extend(divplot)       #applying normalised datasets to temporary 'memory dump' arrays, in 
            ERR_ALL.extend(err_out)     #unsorted form
                

            # plotting of each .fits files left for debugging purpouses
            #plt.figure(figsize=(10,3), dpi=200)
            #plt.plot(XPLOT,divplot, 'k+')               #plot normalised data points and filter
            #plt.plot(XPLOT, interp_savgolmean, 'r-')

            #plt.xlabel('Time(days)')
            #plt.ylabel('Brightness (e-/s)')
            #plt.title(name)        #use of name without .fits file as seen above 

            #plt.show()
            #plt.close()            
            
            
            
        df = pd.DataFrame({                 # creation of dataframe, to output smoothed lightcurve 
            'time': T_ALL,                  # dataset in compact, easily accessed format 
            'flux': F_ALL,
            'error': ERR_ALL
            })
        df.sort_values(by='time', inplace = True)       #dataframe sorted by ascending time, resolves inconsistencies
        return(df)                                   # in directory (i.e. fits files 10,11 listed above 3,4,5 e.t.c)
    def runfunction(self):
        self.df = self.lightcurve_smoothing(files)
    def lc_plot(*args):
        "Separate plot function to allow lightcurve data to exist in both function and ipynb file without"
        "completing lightcurve smoothing or plotting twice"                                               
        plt.figure(figsize=(30,10), dpi=200)   
        plt.xlabel('Time(days)', fontsize = 20)
        plt.errorbar(df['time'],df['flux'], yerr = df['error'], ls = None, marker = 'x', markersize = 7)    #plot normalised data points 
        plt.ylabel('Brightness (e-/s)', fontsize = 20)
        plt.title("Normalised, Flattened Lightcurve for Kepler Object 4", fontsize = 20)    
        plt.show()
        plt.close()
        return print('')
    

df= smoothing.lightcurve_smoothing(files)  #call and run smoothing function so that datasets can be more conveniently 
                                 #referenced in later functions



def periodicity_detection(df, cond1, cond2):
    "Inputs, time conditions to pass as scan range for lomb-scargle filter"
    "function passed only once in final notebook to preserve run time and system resources" 
    "comprehensive masking of exoplanets and subsequent removal of these were done by use of lightkurve model"
    timeall = df['time']
    fluxall= df['flux'] #separate dataframe into arrays
    errorall = df['error']
        
    freqscan = np.linspace((1/cond2),(1/cond1),10000)
    lomb = scipy.signal.lombscargle(timeall, fluxall, freqscan, precenter=True)
       
    
    period = np.linspace(0.01,80,10000)
    lomb2 = scipy.signal.lombscargle(freqscan,lomb,period)
    
    
    pylab.figure(figsize=(12,5))
    pylab.subplot(1,2,1, ylabel='Lomb-Scargle Power', xlabel='Frequency (days$^-1$)', title='Initial lomb-scargle periodogram')
    pylab.plot(freqscan,lomb)

    pylab.subplot(1,2,2, ylabel='Lomb-Scargle Power', xlabel='Period (days)',ylim=(0,0.6e-11),title='Second lomb-scargle periodogram')
    pylab.plot(period,lomb2)
    plt.tight_layout()
    
    periods  = scipy.signal.find_peaks(lomb2, height = 2e-12)
    period_h = period[periods[0]]               #periods detected through SciPy's signal analysis module
    index = np.diff(period_h)                   #set a threshold determined through inspection, then run detection
    lombvals = np.interp(period_h, period, lomb2)       
    
    plt.figure(figsize= (6,5))
    plt.xlim(10,80)
    plt.plot(period, lomb2)
    plt.plot(period_h, lombvals, 'rx')
    plt.show()
    plt.close()

    return (period_h)

class folding:
    def __init__(self, period,a,ttz):
        self.period = period
        self._a = a
        self._ttz = ttz
        self. baseline = 1.0002
        #constructing folding class with basic parameters to be called later
    def lightcurve_folding(time, flux, error, period):
        data = pd.DataFrame({'time': time, 'flux': flux, 'error': error})
        data['phase'] = data.apply(lambda x: ((x.time/ period) - np.floor(x.time / period)), axis=1)
        

        phase_long = np.concatenate((data['phase'], data['phase'] +1.0, data['phase'] +2.0))
        flux_long = np.concatenate((flux, flux, flux))
        err_long = np.concatenate((error, error, error))
        
        return(phase_long, flux_long, err_long)
    
    #lightcurve folding function which can be applied to any set of data inputs, as seen in arguments
    def lightcurve_application(*args):
        #applying lightcurve_smoothing to all periods detected, saving into combined dataframe, masking transits 
        # and creating individual dataframes for each transit
        df= smoothing.lightcurve_smoothing(files)
        p1,p_flux1,p_error1 = folding.lightcurve_folding(df['time'],df['flux'],df['error'],21.77745575)
        p2,p_flux2,p_error2 = folding.lightcurve_folding(df['time'],df['flux'],df['error'],41.022302)
        p3,p_flux3,p_error3 = folding.lightcurve_folding(df['time'],df['flux'],df['error'],31.786679)
        p4,p_flux4,p_error4 = folding.lightcurve_folding(df['time'],df['flux'],df['error'],13.176418)
        allfoldeddf = pd.DataFrame({
            'p1': p1,
            'p_flux1': p_flux1,
            'p_error1': p_error1,
            'p2': p2,
            'p_flux2': p_flux2,
            'p_error2': p_error2,
            'p3': p3,
            'p_flux3': p_flux3, 
            'p_error3': p_error3,
            'p4': p4,
            'p_flux4': p_flux4,
        'p_error4': p_error4 })
        exo1mask = (p1>1.67) & (p1<1.74)
        exo2mask = (p2>2.17) & (p2<2.25)
        exo3mask = (p3> 1.24) & (p3 < 1.31)
        exo4mask = (p4>1.85) & (p4<1.98)
        exo1df = pd.DataFrame({'phase': p1[exo1mask], 'flux': p_flux1[exo1mask], 'error': p_error1[exo1mask]})
        exo2df = pd.DataFrame({'phase': p2[exo2mask], 'flux': p_flux2[exo2mask], 'error': p_error2[exo2mask]})
        exo3df = pd.DataFrame({'phase': p3[exo3mask], 'flux': p_flux3[exo3mask], 'error': p_error3[exo3mask]})
        exo4df = pd.DataFrame({'phase': p4[exo4mask], 'flux': p_flux4[exo4mask], 'error': p_error4[exo4mask]})
        exo1df.sort_values(by='phase', inplace = True)
        exo2df.sort_values(by='phase', inplace = True)
        exo3df.sort_values(by='phase', inplace = True)
        exo4df.sort_values(by='phase', inplace = True)
        fig = plt.figure(figsize = (25,12))
        fig.subplots_adjust(hspace=0.3, wspace=0.2)
        plt.subplot(2,3,1, ylabel = 'Normalised Flux', xlabel = 'Phase', title = 'folded lightcurve for exoplanet of period 21.7776 days')
        plt.plot(allfoldeddf['p1'],allfoldeddf['p_flux1'], c='r', marker='o',ls='None', ms=0.7 )

        plt.subplot(2,3,2, ylabel = 'Normalised Flux', xlabel = 'Phase', title = 'folded lightcurve for exoplanet of period 41.03 days')
        plt.plot(allfoldeddf['p2'],allfoldeddf['p_flux2'],c='r', marker='o',ls='None', ms=0.7 )

        plt.subplot(2,3,4, ylabel = 'Normalised Flux', xlabel = 'Phase', title = 'folded lightcurve for exoplanet of period 31 days')
        plt.plot(allfoldeddf['p3'],allfoldeddf['p_flux3'],c='r', marker='o',ls='None', ms=0.7 )

        plt.subplot(2,3,5, ylabel = 'Normalised Flux', xlabel = 'Phase', title = 'folded lightcurve for exoplanet of period 13.17 days')
        plt.plot(allfoldeddf['p4'],allfoldeddf['p_flux4'],c='r', marker='o',ls='None', ms=0.7 )
        return(allfoldeddf, exo1df, exo2df, exo3df, exo4df)
    def quartic_transit(t,a, depth,ttz,baseline):
        return (a*(t-ttz)**4 + 0.5*depth) - abs((a*(t-ttz)**4 + 0.5*depth)) +baseline
    def exo_fit_errorbar(perioddfsegment, fluxdfsegment, a, depth, ttz, p, p_flux, p_error, xblim, xmlim, Title):
        popt,pcov = curve_fit(folding.quartic_transit, perioddfsegment, fluxdfsegment,p0 =[ a, -(depth), ttz, 1.0002])
        plt.figure(figsize=(20,10))
        plt.xlim(xblim, xmlim)
        plt.plot(perioddfsegment, folding.quartic_transit(perioddfsegment, *popt), label = 'Fit')
        plt.errorbar(p, p_flux, yerr = p_error, c='r', marker='o',ls='None', ms=0.7, label = 'Folded Data')
        plt.legend(loc = 'upper right')
        plt.xlabel('Phase')
        plt.ylabel('Flux')
        plt.title(Title)
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        plt.text(ttz+0.02, 0.99925, 'transit depth ={:10.7f}'.format(popt[1]), bbox = props, fontsize =15)
        plt.show()
        plt.close()
        Error = np.nanmean(p_error)
        baseline = popt[3]
        Serror = (0.07)
        exoError = (((np.sqrt(abs(popt[1])))*Serror )**(2) + (((Serror)/(2*np.sqrt(abs(popt[1]))))*Error)**(2))**(1/2)
        percent = exoError * 100
        print("Absolute error:", exoError)
        print("percent = ", percent,"%")
        print('baseline', baseline)
        return(folding.quartic_transit(perioddfsegment, *popt), popt)
        
    def exo_fit(perioddfsegment, fluxdfsegment, a, depth, ttz, p, p_flux, p_error, xblim, xmlim, Title):
        popt,pcov = curve_fit(folding.quartic_transit, perioddfsegment, fluxdfsegment,p0 =[ a, -(depth), ttz, 1.0002])
        plt.figure(figsize=(20,10))
        plt.xlim(xblim, xmlim)
        plt.plot(perioddfsegment, folding.quartic_transit(perioddfsegment, *popt), label = 'Fit')
        plt.plot(p, p_flux, c='r', marker='o',ls='None', ms=0.7, label = 'Folded Data')
        plt.legend(loc = 'upper right')
        plt.xlabel('Phase')
        plt.ylabel('Flux')
        plt.title(Title)
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        plt.text(ttz+0.02, 0.99925, 'transit depth ={:10.7f}'.format(popt[1]), bbox = props, fontsize =15)
        plt.show()
        plt.close()
        Error = np.nanmean(p_error)
        baseline = popt[3]
        Serror = (0.07)
        exoError = (((np.sqrt(abs(popt[1])))*Serror )**(2) + (((Serror)/(2*np.sqrt(abs(popt[1]))))*Error)**(2))**(1/2)
        ErrorAbs = exoError * 100
        print("Absolute Error = ", ErrorAbs)
        return(folding.quartic_transit(perioddfsegment, *popt), popt)



# Now exoplanets have been detected, folded and fitted, analysis of exoplanets. 
# To prevent extensive confusing numberical noise on notebook, use classes to make object for each exoplanet
# define list of constant values to be used 

plist = [21.776, 41.022302, 13.176818,31.786679] #periods of exoplanets
star_radius = 1155360000 #stellar radius(actual)
starmass =1.1*1.99e30   #stellar mass (actual)
starunc = 0.1 #uncertainty in star 
#data pertaining to the exoplanets
exo1mass = 4.1*5.972e24 #mass of exoplanets
exo2mass = 9.6 *5.972e24
exo3mass = 0.390*5.972e24
exo4mass = 5.5 *5.972e24

class exoplanet:
    def __init__(self, planetmass, starmass, baseline, transitf, stellar_radius, period,
                 starunc, planetplus, planetminus, midpoint):
        from scipy.constants import G 
        self._G = G
        self._mp = planetmass
        self._ms = starmass
        self._bl = baseline
        self._tr = transitf
        self._rs = stellar_radius
        self._P = period
        self._unS = starunc*6.96e8
        self._plplus = planetplus
        self._plmin = abs(planetminus)
        self._rp = np.sqrt(((self._bl - self._tr)/self._tr)*(self._rs**2))/6.37e6
        self.planetmax = self._mp + self._plplus
        self.planetmin = self._mp - planetminus
        self.starmax = self._rs+self._unS
        self.starmin = self._rs - self._unS
        self.midpoint = midpoint
        self.pcdrop = (self._bl-self._tr)/self._bl
        self._a = ((6.674e-11*self._ms*((self._P*24*3600)**2))/((4*(np.pi))**2))**(1/3)
        self._Areal = self._a / 1.496e11
    def semimajor(self):
        #calculates semi-major axis of planet
        a = ( self._G * self._ms * (self._P/(2*np.pi))**2)**(1/3)
        return a
    def radius(self):
        # calculates the radius of the planet
        _rp = np.sqrt(((self._bl - self._tr)/self._tr)*(self._rs**2))/6.37e6
        return _rp
    def redmass(self):
        ## calculates the reduced mass of the system
        return self._mp/(self._ms + self._mp)
    def sminAU(self):
        a = ( self._G * self._ms * (self._P/(2*np.pi))**2)**(1/3)
        areal = a/149817679298.79666
        return areal       
    def uncertaintypc(self):
        _plpercent = (self.planetmax-self._mp)/self._mp
        self.spercent = (self.starmax - self._ms)/self._ms
        return(_plpercent)
    def temperature(self):
        AU = 1.496e11
        so_lum = 3.828e26 
        st_lum = 0.507 * so_lum
        bzm_constant = 5.67e-8
        self.planet_temp = ((st_lum/(16*np.pi*bzm_constant*(self._a* 24 *3600)**2)))**(1/4)
        pl_temp = ((st_lum)/(16*np.pi*bzm_constant*((self._a *24 *3600)**2)))**(1/4)
        return(pl_temp)
    def flux_incident(self):
        AU = 1.496e11
        semireal= self._a /AU
        so_lum = 3.828e26
        st_lum = 0.507 * so_lum
        flux = st_lum/semireal
        return flux


# create exoplanet objects to exist within functions file 
exo_1 = exoplanet(exo1mass, starmass , 0.9997135088737736, 0.9993989278983916,star_radius , 21.7776,starunc, 1.7,2, 1.705)
exo_2 = exoplanet(exo2mass, starmass, 1,0.9993989278983916, star_radius,  41.022302, starunc, 1.7, 1.8,2.21)
exo_3 = exoplanet(exo4mass, starmass, 0.99999, 0.99925, star_radius, 31.786679, starunc, 1.2,1.1,1.93)
exo_4 = exoplanet(exo3mass, starmass, 0.9999713, 0.99975, star_radius, 13.176418, starunc, 1.24, 0.32,1.75)


class exoplanet_lists:
    def __init__(self):
        # exoplanet properties are collated into lists; can be called anywhere throughout notebook by creating object 
        self.planet_radius=[exo_1.radius(), exo_2.radius(),exo_3.radius(), exo_4.radius()]
        self.planet_period = np.array([(exo_1._P), (exo_2._P),(exo_3._P), (exo_4._P)])
        self.planet_mass = np.array ([(4.1),(9.6),(0.390), (5.5)])
        self.planetabs = [1.7, 1.8, 0.32, 1.1]
        self.planet_err =np.array([(1.85), (1.75), (0.78), (1.15)])
        #planet_err = [exo_1.uncertaintypc(), exo_2.uncertaintypc(), exo_3.uncertaintypc(),exo_4.uncertaintypc()]
        self.planet_temps = [exo_1.temperature(), exo_2.temperature(), exo_3.temperature(), exo_4.temperature()]

class comparison:
    def __init__(self, allsystemsdf, exorad, exoperiod, exoerror, HZdf):
        self._allsys = allsystemsdf
        self._colourplot = ('black', 'black', 'darkorange', 'red', 'darkkhaki', 'violet', 'green', 'slategray', 'dodgerblue')
        self._exorad = exorad
        self._exoperiod = exoperiod
        self._exoerror = exoerror
        self.Lstar = 0.504
        self._HZr_i = np.sqrt(self.Lstar/1.1)
        self.HZr_o = np.sqrt(self.Lstar/0.53)
        self.HZdf = HZdf
        self.mass_solarsys = [0.330, 4.87,5.97,0.073,0.642,1898]
        self.temp_solarsys = [440, 737, 288, 210, 165]
        self.temp = [5904,5904,5904,5904] # star temperature
        self.temp1 = [5704, 5704, 5704, 5704, 5704] #solar temperature
    def comparisonplot(self):
        plt.figure(figsize = (20,10))
        for i in range(1,8):
            plotvals = self._allsys[self._allsys['np']==i]
            plt.plot( np.log(plotvals['radius']), np.log(plotvals['period']),marker = 'o', color = self._colourplot[i],
            markersize = '5', ls = 'None', label = 'number of planets in system: {}'.format(i))
        plt.errorbar( np.log(self._exorad), np.log(self._exoperiod), yerr = (np.log(self._exoerror)),
                    marker = '*', color = 'blue', markersize = '15', ls = 'None',
        label = 'Exoplanets detected from object 4')
        plt.legend(loc = 'upper right')
        plt.xlabel('log(Period[days])')
        plt.ylabel('log(Planet Mass / Earth Mass )')
        plt.show()
        plt.close()
        return 1
    def habitable_zone_analysis():
        #calculates and plots habitable zone plot, as well as density plot with lines for earth water and rock
        plist = [21.776, 41.022302, 13.176818,31.786679]
        star_radius = 1155360000 
        starmass =1.1*1.99e30
        starunc = 0.1
        #data pertaining to the exoplanets
        exo1mass = 4.1*5.972e24
        exo2mass = 9.6 *5.972e24
        exo3mass = 0.390*5.972e24
        exo4mass = 5.5 *5.972e24

        def semi_calc(p):
            valuetest = ((4*2.189e+30*(p)**2)/(4*np.pi**2))
            a = valuetest**(1/3)
            return a 
        smlist = []
        for work in plist:
            pls = semi_calc(work)
            smlist.append(pls/1.496e11)
    #
        habzDF = pd.read_csv('habz.csv')
        habzDF.dropna(inplace = True)
        
        def temp_eq(a):
            AU = 1.49e11
            stellar_lum = 0.507 * 3.827e26
            sbzm = 5.67e-8
            t= (stellar_lum/(16*np.pi*sbzm*(a*AU)**2))**(1/4)
            return t
        temparray=[]
        for n in smlist:
            va = temp_eq(n)
            temparray.append(va)


        mass_solarsys = [0.330,	4.87,5.97,0.073,0.642,1898]
        temp_solarsys = [440, 737, 288, 210, 165]
        planet_mass = np.array ([(4.1),(9.6), (5.5), (0.390)])
        planet_radius=[exo_1.radius(), exo_2.radius(),exo_3.radius(), exo_4.radius()]
        temp = [5904,5904,5904,5904]
        temp1 = [5704, 5704, 5704, 5704, 5704]

        plt.figure(figsize = (25,15))
        plt.tick_params(which = 'major', direction='in', length=6, width=2, colors='k',
               grid_color='grey', grid_alpha=0.5)
        plt.tick_params(which = 'minor', direction='in', length=6, width=2, colors='k',
               grid_color='grey', grid_alpha=0.5)
        plt.grid(True, linestyle='-.')
        plt.yticks(fontsize = 15)
        plt.xticks(fontsize = 15)
        plt.title(" Radius of Exoplanets as a Function Semi Major Axis  ", fontsize = 25)
        plt.plot((habzDF['pl_eqt']), (habzDF['st_teff']), marker = 'o', color = 'k', markersize = '2',
                ls = 'None', label = "Nasa Exoplanet Archive Planets" )
        plt.plot( (temparray),temp,  marker = 'v', color = 'b', markersize = '15', 
                ls = 'None', label = "Object 4 Exoplanets" )
        plt.plot((temp_solarsys),temp1, marker = 'D', color = 'g',
                markersize = '15', ls = 'None', label = "Solar System Planets" )
        plt.axvline(x = 273), plt.axvline(x=373)
        z = np.linspace(-2000,100000, 5)
        plt.fill_betweenx(z, 273, 373, color = 'darkkhaki', alpha = 0.3)
        plt.xscale('log'), plt.ylim(1000,8000)
        #main planet data plotted for exoplanets, archive and slar system
        #set limits
        # label solar system planets
        plt.xlabel("Equilibrium Temperature [K]", fontsize = 15), plt.ylabel("Stellar Temperature", fontsize = 20)
        #plot and fill habitable zone for star
        #fill habitable zone of star
        plt.legend(loc = 'upper right', fontsize = 12, fancybox = True, prop = {'size': 20})
        plt.show()
        plt.close()
        # all.csv is nasa exoplanet archive data for systems with any number of exoplanets. 
        allplanetdata = np.loadtxt("all.csv", delimiter = ',', skiprows = 1)
        num_planets = allplanetdata[:,0]
        all_period = allplanetdata[:,1]
        all_radius = allplanetdata[:,2]
        all_mass = allplanetdata[:,3]

        allsys = pd.DataFrame({
            'np': num_planets,
            'period': all_period,
            'radius': all_radius,
            'mass': all_mass
        })
        allsys.sort_values(by = 'np', inplace = True)

        densitydf = pd.read_csv('densityplot.csv')
        densitydf.dropna(inplace = True)
        

        def compisition_func(density, mass): 
            #function to plot radius needed to meet density criteria across mass array

            r_earth = 6.371e6
            #define constants values 
            M_earth = 5.972e24

            r = (3/(4 * np.pi * density))**(1/3) * ((mass*M_earth)**(1/3))
            return r/r_earth
        
        mass = np.array(allsys['mass'])
        sorted_mass = np.sort(mass)
        #create sorted mass 
        rockline = compisition_func(2980, sorted_mass)
        waterline = compisition_func(999, sorted_mass)
        airline = compisition_func(1.225, sorted_mass)

        plt.figure(figsize = (25,15))
        plt.title("Radius as a function of Mass with relative density compisitions", fontsize = 15)
        plt.ylim(0,25), plt.yticks(fontsize = 15)
        plt.xlim(1,1000), plt.yticks(fontsize = 15)
        plt.xscale('log')
        plt.tick_params(which = 'major', direction='in', length=6, width=2, colors='k',
                       grid_color='grey', grid_alpha=0.5)
        plt.tick_params(which = 'minor', direction='in', length=6, width=2, colors='k',
                       grid_color='grey', grid_alpha=0.5)
        plt.grid(True, linestyle='-.')
        #plot exoplanet and nasa data 

        ax = sns.scatterplot(densitydf['pl_masse'], densitydf['pl_rade'], marker='o', hue=densitydf['pl_eqt'],linewidth=0,
                             palette='viridis')
        plt.plot(planet_mass, planet_radius, marker = 'D', color = 'r', markersize =10, ls = 'None')
        legend1 = ax.legend(title=' Equilibrium Temperature (K)', loc=2)

        #plot lines as separate variable to separate into two legends for clarity 
        water, = ax.plot(sorted_mass, waterline, label='Water', c='r', ls='--')
        rock, = ax.plot(sorted_mass, rockline, label='Rock', c='orange', ls='--')
        air, = ax.plot(sorted_mass, airline, label='air', c='y', ls='--')
        #plot legends
        legend2 = ax.legend([water, rock, air],['Water', 'Rock', 'Air'], title='Density Function Lines', loc=4)
        ax.add_artist(legend1)
        plt.xlabel("$log_{10}$ Planet Mass / Earth Mass", fontsize = 15)
        plt.ylabel("Planet Radius / Earth Radius", fontsize = 15)
        #show and close figure
        plt.show()
        plt.close()
        