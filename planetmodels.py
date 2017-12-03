from __future__ import print_function
from astropy import units as u
from scipy.interpolate import interp1d
import os.path as osp
import numpy as np
import pandas as pd
import pickle
import glob



class PlanetModels:

    def cullModels(self, folder, re_string='model.*.txt', outfile='PlanetModels.pickle'):
        """
        Gathers up all the models in a folder and strucutures into a multi-dimenional dictionary.

        modelDict = {'file1': {'age1': {'hdr1': [ ],...,'hdrN': [ ]}
                                .... ,
                               'ageN':{'hdr1': [ ],...,'hdrN': [ ]}
                               },
                       ...
                     'fileN': {'age1': {'hdr1': [ ],...,'hdrN': [ ]}
                                .... ,
                               'ageN':{'hdr1': [ ],...,'hdrN': [ ]}
                               }
                     }

        So, for each file, there's a dictionary. WIthin that dict, each age is a dictionary
        with the data for that age.. as a dictionary.


        Parameters
        ----------
        folder: Folder path to where the model files are
        re_string: regex string to obtain models.
        outfile: what pickle file you want the dictionary to be written out


        """

        self.model_files = glob.glob(osp.join(folder, re_string))

        self.modelDict = {}

        for mfile in self.model_files :

            # ADD NEW DICTIONARY ENTRY W/ FILENAME W/O EXTENSION
            fkey = osp.splitext(osp.basename(mfile))[0]
            self.modelDict[fkey] = {}

            # LOAD ALL DATA
            with open(mfile, 'r') as myfile:
                data = myfile.read().split('\n\n\n')

            hdr = None

            # EACH BLOCK IN THE FILE
            for section in data:
                if section != '':
                    # LIST OF EACH LINE
                    section = section.split('\n')
                    split_section = [line for line in section if line != '']
                    age_string = split_section.pop(0)
                    hdr_string = split_section.pop(0)

                    if hdr is None:
                        hdr = hdr_string.strip(' ').split()

                    age_Byr = age_string.strip(' ').split('=')
                    age_Byr = float(age_Byr[1])

                    # CREATE CUBE FROM THE LIST OF STRINGS
                    modCube = np.array([np.array(line.split(), dtype='float')
                                        for line in split_section])

                    modelDict_agei_fi = dict(zip(hdr, modCube.T))

                    self.modelDict[fkey][age_Byr] = modelDict_agei_fi

        self.modelNames = self.modelDict.keys()

        # DUMP THE FULL DICITONARY
        outfile = osp.join(folder, outfile)

        pickle.dump(self.modelDict, open(outfile, "wb"))

        print('Pickle file dumped to {}'.format(outfile))



    def loadPickelModels(self,pfile):
        """
        Load pickle'd file of dictionary of all the models in the folder you have all your .. umm...
        models in.

        Parameters
        ----------
        pfile: path/filename of pickle file.

        """

        self.modelDict = pickle.load( open( pfile, 'rb' ) )
        self.modelNames = self.modelDict.keys()



    def ageIntp(self, ageStar, modname, xcol, ycol):

        """
        Creates a interpolation object (an interpolator, if you will) for
        a given stellar age, modelname, xcol, ycol (e.g., Mass and L-band mag)
        and whatever xrange you want to create the interpolation object for.

        Parameters
        ----------
        ageStar : ASTROPY quantity for age of star
        modname : Which model name to use (e.g., model.AMES-dusty.M-0.0.NaCo.txt)
        xcol : Column name to use for the x-interpolation (e.g., M/Ms)
        ycol : Column name to use for final y-axis interpolation (e.g., L')

        Returns
        -------
        intpFinal : scipy.interpolate.interpolate.interp1d object of x and y
                    (e.g., Mass and L'-band) at the age of the star for a
                    particular model.

        intpFinal_inv : scipy.interpolate.interpolate.interp1d object of y and x
                    (e.g., L'-band and Mass) at the age of the star for a
                    particular model.
        """

        # CREATE RANGE OF ARTIFICIAL X VALUES TO USE FOR INTERPOLATION.
        # E.G., for mass range

        # CONVERT AGE TO GYR
        ageStar = ageStar.to('Gyr').value

        # SELECT MODEL CUBE
        model = self.modelDict[modname]

        # GET ALL AGES IN THAT MODEL CUBE AND SORT
        ages = np.array(model.keys())
        ages.sort()

        # GET +/- 2 MODEL CUBES AROUND AGE OF STAR
        indage = np.searchsorted(ages,ageStar)
        indi = indage - 2
        indf = indage + 2

        if indi<0: indi = 0
        if indf > len(ages)-1: indf = -1

        agesUse = ages[indi:indf]


        # INTERPOLATE/EXTRAPOLATE FOR EACH Y (E.G., L MAG) AT EACH AGE
        # ALONG X (E.G., MASS) AXIS
        # THEN INTERPOLATE Y CUBE TO STELLAR AGE TO GET 1D ARRAY OF Y AT
        # AGE OF STAR.
        # THEN GET INTERPOLATION OBJECT FOR X AND Y AT AGE OF STAR (E.G., MASS & L MAG)
        yCube = []

        # GETS LIMITS BASED ON THE AVAILABLE RANGE IN THE MODELS, NOT FROM USER INPUT
        allys = np.array([model[agei][xcol] for agei in agesUse])

        xi = np.array(map(np.min,allys)).min()
        xf = np.array(map(np.max,allys)).max()
        # MASS RANGE
        xUsearr = np.logspace(np.log10(xi), np.log10(xf), 70)

        for agei in agesUse:

            ydati = model[agei][ycol]
            xdati = model[agei][xcol]


            intpi = interp1d(xdati, ydati,
                             fill_value= (ydati[0],ydati[-1]),
                             bounds_error=False )

            ydat_newi = intpi(xUsearr)
            yCube.append(ydat_newi)

        # GET THE Y ARRAY (E.G., L'BAND MAG)
        yCube = np.array(yCube)
        intpY = interp1d(agesUse, yCube.T)
        yFinal_IntOvAge = intpY(ageStar)

        intpFinal_inv = interp1d(yFinal_IntOvAge[::-1], xUsearr[::-1], kind='linear',
                                 fill_value=(xUsearr[-1], xUsearr[0]), bounds_error=False)


        return intpFinal_inv



    def getMasses(self,intpObj,contrasts, starMag, starDist):
        """

        Parameters
        ----------
        intpObj
        contrasts
        starMag
        starDist

        Returns
        -------

        """

        m_planet = -2.5 * np.log10(contrasts) + starMag

        M_planet = m_planet - 5 * np.log10(starDist/10.)


        massPl = intpObj(M_planet) * u.Msun


        return massPl


    def treatSpiegelBrurrowModels(self,infile,outfolder):
        """

        Parameters
        ----------
        infile
        outfolder

        Returns
        -------

        """
        # e.g., infile = '/Users/rpatel/Dropbox/Research/Interpolation_Files/PlanetModels/BurrowsSpiegel.csv'
        datsb = pd.read_csv(infile)

        header = datsb.columns
        header_st = '  '.join(map(str, header))

        S0Unique = np.unique(datsb.S0.values)
        ageGyrUni = np.unique(datsb.ageGyr.values)
        atmUnique = np.unique(datsb.atm.values)
        # e.g., outfolder = '/Users/rpatel/Dropbox/Research/Interpolation_Files/PlanetModels/'
        sb_savefolder = outfolder
        for atmi in atmUnique:
            for S0i in S0Unique:
                fname = osp.join(sb_savefolder, 'model.SpBu_atm{}_S0{}.txt'.format(atmi, S0i))

                with open(fname, 'a') as f_handle:

                    for agei in ageGyrUni:
                        dat = datsb[(datsb.S0 == S0i) &
                                    (datsb.atm == atmi) &
                                    (datsb.ageGyr == agei)]
                        if dat.size != 0 :
                            np.savetxt(f_handle, dat.values, header=header_st, fmt='%f',
                                       comments='\n\n\n\n   t (Gyr) = {}\n'.format(agei))


