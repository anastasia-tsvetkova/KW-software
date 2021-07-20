__author__ = 'Anastasia Tsvetkova'
__email__  = 'tsvetkova.lea@gmail.com'

import re, os, pymysql, yaml, paramiko, zipfile, shutil, inspect, glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class KW_data(object):
    """
    A class for processing KW light curves and spectral data 
    
    Parameters:
    :ID: Burst ID in the KW database
    :path2yaml: Path to the yaml-file with logins, pswds ,etc.
    :path2data: Path to the folder where the data on burst will be stored
    :path2soft: Path to the folder where all proprietary KW software is stored
    :path2lz: Path to source lz-file 
    :path2ld: Path to libcfitsio.so
    :lz_name: Name of the quicklook file if the daily lz-file is not downloaded to the datastore yet
    """
    

    def __init__(self, ID,
                 path2yaml='./conf/application.yml',
                 path2data=None,
                 path2soft=None,
                 path2lz=None,
                 path2ld="",
                 lz_name=None,
                 verbose=False
                ):
        
        if isinstance(ID, int):
            self._id = ID
        else:
            raise TypeError("Burst ID should be an integer number!")
            
        self._path2yaml = path2yaml
        
        self._path2data = path2data
        if self._path2data is not None:
            if path2data.endswith('/'):
                self._path2data = path2data
            else:
                self._path2data = path2data + '/'
       
        self._path2lz = path2lz
        if self._path2lz is not None:
            if not self._path2lz.endswith('/'):
                self._path2lz += '/'
 
        self._lz_name = lz_name 

        self._path2soft = path2soft
        if self._path2soft is not None:
            if not self._path2soft.endswith('/'):
                self._path2soft += '/'
        else:
            self._path2soft = "./"
            
        self._path2LD = path2ld

        self._current_dir = os.getcwd()
        if not self._current_dir.endswith('/'):
            self._current_dir += '/'
        
        self._spectra = list(map(float, os.popen(f"{self._path2soft}kw-ph phtime --id={ID}").read().split()))

        self._verbose = verbose
 

    def retrieve_burst_from_KW_DB(self):
        """Extracting info about burst from the KW database"""

        if self._verbose:
            print(self.retrieve_burst_from_KW_DB.__doc__)

        conf = yaml.load(open(self._path2yaml), Loader=yaml.FullLoader)
        login = conf['mysql']['user']['login']
        pswd = conf['mysql']['user']['password']
        host = conf['mysql']['host']
        database = conf['mysql']['database']

        """Making open database connection"""
        if self._verbose:
            print("Making open database connection")

        db = pymysql.connect(host=host, user=login, password=pswd, database=database)
 
        """Preparing a cursor object using cursor() method"""
        cursor = db.cursor()

        """Preparing SQL query to SELECT a record from the database"""
        sql = f"SELECT GCN, UT, msec, type, trigdet, comment, event FROM wind WHERE ID = {self._id}"

        try:
            """Executing the SQL command"""
            cursor.execute(sql)

            """Fetching all the rows in a list of lists"""
            results = cursor.fetchone()

            (name, UT, self._msec, self._burst_type, self._det, 
             self._comment, self._event) = results

        except:
            print ("Error: unable to fetch data from the konuswind database")

        """Disconnecting from server"""
        db.close()

        expr = re.match(r"(\d{2}(\d{2}))\D+(\d+)\D+(\d+)\D+((\d+)\D+(\d+)\D+(\d+))", str(UT))
        self._date = expr[1] + expr[3] + expr[4]
        self._short_date = expr[2] + expr[3] + expr[4]
        self._UT = str(expr[5]) + '.' + '{:03d}'.format(int(self._msec))
        self._short_time = '{:05d}'.format(int(expr[6])*3600 + int(expr[7])*60 + int(expr[8]))
        self._time = self._short_time + '.' + '{:03d}'.format(int(self._msec))
        
        if name: self._burst_name = name.replace('GRB', '')
        else: self._burst_name = self._date + 'T' + self._short_time
            
        expr = re.search(r"(\d{2})(\d{2})(\d{2})\d{2}", self._date) 
        self._year = expr[1] + expr[2]
        self._year_month = expr[2] + expr[3]
    
        if self._lz_name is None:
            self._lz_name = self._year_month+"lz.zip"
            
    
    def download_lz_from_gamma(self):
        """Copying zip-files from gamma to the local host"""

        if self._verbose:
            print(self.download_lz_from_gamma.__doc__)

        conf = yaml.load(open(self._path2yaml), Loader=yaml.FullLoader)
        login = conf['server']['user']['login']
        pswd = conf['server']['user']['password']
        host = conf['server']['host']

        if self._path2lz is None:
            self._path2lz = f"/home/{login}/konus/wind/LZ/{self._year}/" 

        if self._lz_name is None:
            self._lz_name = f"{self._year_month}lz.zip"

        if self._path2data is None:
             self._path2data = self._path2lz

        if not os.path.exists(self._path2data): 
            os.mkdir(self._path2data)

        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(hostname=host, username=login, password=pswd)
        sftp = ssh.open_sftp()
        self._local_path = self._path2data + self._lz_name
        self._remote_path = self._path2lz + self._lz_name
        sftp.get(self._remote_path, self._local_path)
        sftp.close()
        ssh.close()
        
        if self._lz_name.endswith('.zip'):
            self.unpack_lz()
    
        if self._verbose:
            print(inspect.getdoc(self.download_lz_from_gamma))


    def unpack_lz(self, delete_extra_files=True):
        """Unpacks zip-files 

        :delete_extra_files: delete files for the unrelated months
        """

        if self._verbose:
            print("Unpacking zip-files")
            
        with zipfile.ZipFile(self._path2data + self._lz_name, 'r') as zip_ref:
            zip_ref.extractall(self._path2data)

        if delete_extra_files:
            file_list = glob.glob(f"{self._path2data}*.lz")
            file_list.remove(f"{self._path2data}{self._short_date}.lz")
            for f in file_list:
                os.remove(f)
    
        
    def lzlist(self):
        """Generating trigger-mode data from lz-files"""

        if self._verbose:
            print(self.lzlist.__doc__)

        os.chdir(self._path2data)

        """Extracting data from lz-file"""

        if self._verbose:
            print("Extracting data from lz-file")

        os.system(f"{self._current_dir}{self._path2soft}lzlist --shortbga {self._short_date}.lz")

        """Approximating LC background"""

        if self._verbose:
            print("Approximating LC background")

        os.system(f"wine {self._current_dir}{self._path2soft}/BGdetermination.exe ./kw{self._date}_{self._short_time}.bga {self._date} {self._short_time} \
                    -2500 -100")

        """Making LCs from pla-file"""

        if self._verbose:
            print("Making LCs from pla-file")

        os.system(f"wine {self._current_dir}{self._path2soft}/PLA2THR.exe kw{self._date}_{self._short_time}.pla")

        """Rebinning LCs"""

        if self._verbose:
            print("Rebinning LCs")

        os.system(f"wine {self._current_dir}{self._path2soft}/Wind_rebin_all.exe kw{self._date}_{self._short_time}.thr 1 1")

        os.chdir(self._current_dir)
        
        """Removing unrelated triggers from the directory"""

        if self._verbose:
            print("Removing unrelated triggers from the directory")

        file_list = os.listdir(self._path2data)
        for file in file_list:
            time = re.search(r"kw\d{8}_(\d{5})", file) 
            if time and time[1] != self._short_time:
                os.remove(self._path2data+file)  

    
    def bg_approx(self, ti=-2600, tf=-50):
        """ Approximates background for the defined time interval

        :ti: start of the bg approximation time interval
        :tf: stop of the bg approximation time interval
        """

        if self._verbose:
            print("Approximating LC background")

        os.chdir(self._path2data)

        os.system(f"wine {self._current_dir}{self._path2soft}/BGdetermination.exe ./kw{self._date}_{self._short_time}.bga {self._date} {self._short_time} \
                    {ti} {tf}")

        os.chdir(self._current_dir)


    def calib(self, sp_i = 42, sp_f=50, recalibrate=False):
        """ Calibrates a pla file

        :sp_i: start of the calibration time interval
        :sp_f: stop of the calibration time interval
        :recalibrate: recalibrate the pla-file if a calibration file already exists
        """

        if self._verbose:
            print("Calibrating pla file")

        os.chdir(self._path2data)
        
        if not os.path.exists(f"kw{self._date}_{self._short_time}_nomin.pla"):
            shutil.copyfile(f"kw{self._date}_{self._short_time}.pla",f"kw{self._date}_{self._short_time}_nomin.pla")
        else: print("_nomin.pla already exists!")

        if recalibrate or not os.path.exists("calib_coeff.dat"):
            os.system(f"{self._current_dir}{self._path2soft}kw_calib --filename=kw{self._date}_{self._short_time}_nomin.pla --SpBegin={sp_i} \
                    --SpEnd={sp_f} --noFile > calib_coeff.dat")
        
        if recalibrate or not os.path.exists(f"kw{self._date}_{self._short_time}_nomin_{sp_i}_{sp_f}_calib.log"):
            os.system(f"wine {self._current_dir}{self._path2soft}/Calib.exe kw{self._date}_{self._short_time}_nomin.pla {sp_i} {sp_f} -v")

        chi = 0
                
        with open(f"kw{self._date}_{self._short_time}_nomin_{sp_i}_{sp_f}_calib.log", 'r') as file:
            next(file)
            line = file.readline().split()
            chi2 = float(line[0])
            
        if chi2 < 20:
            print("Calibration chi2 =", chi2)
        else:
            print("Warning: high calibration chi2 =", chi2)
            
        with open("./calib_coeff.dat", 'r') as file:
            self._calib_coeff_PHA1 = file.readline().split()
            self._calib_coeff_PHA2 = file.readline().split()

        print(f"Calibration coefficients: PHA1 = {self._calib_coeff_PHA1[0]} PHA2 = {self._calib_coeff_PHA2[0]} for sp = {sp_i}-{sp_f}")
        
        """Making a new calibrated pla-file"""

        if self._verbose:
            print("Making a new calibrated pla-file")

        os.system(f"{self._path2soft}lzlist --shortbga {self._short_date}.lz")

        os.chdir(self._current_dir)


    def G1G2G3_bounds(self):
        """ Calculates the channel boundaries 

        :returns: the bounds of spectral channels
        """

        if self._verbose:
            print("Calculating channel boundaries")

        """ From Tsvetkova et al. (2021) """
        self._nominal_bounds = {'S1': [10.59, 44.72, 185.18, 757.32],
                                'S2': [10.99, 46.41, 191.31, 757.32]
                               }
        
        bounds = map(lambda x: x * float(self._calib_coeff_PHA1[0]), self._nominal_bounds[self._det])
        bounds = list(map(lambda x: '{:.0f}'.format(x), bounds))

        return bounds 

    
    def plot_bg(self, limits=[-2600,-10]):
        """ Plots background LC to allow its visual check 

        :limits: boundaries of the plotting area
        """
        
        lc_file = f"{self._path2data}kw{self._date}.bga"
        data = pd.read_csv(lc_file, sep='\s+',header=0)
        data = pd.DataFrame(data)
        data['T_shift'] = data['T'] - float(self._time) - 2.944
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(20,15)) # sharex=True
        fig.suptitle(f"KONUS-WIND GRB {self._burst_name}\nT0 = {self._time} s UT ({self._UT})\n{self._det}", fontsize=20)
        plt.xlabel("$T - T_{0}$ (s)", fontsize=14)
        y_title = "counts/2.944 s"

        plt.locator_params(axis='y', nbins=10)
        plt.locator_params(axis='x', nbins=10)

        (G1, G2, G3, G4) = self.G1G2G3_bounds()

        text = [f"G1: {G1}-{G2} keV", f"G2: {G2}-{G3} keV", f"G3: {G3}-{G4} keV"]

        idx = [self._det + 'G1', self._det + 'G2', self._det + 'G3']

        x_lim = limits[1] - limits[0]
                
        for i, ax in enumerate([ax1, ax2, ax3]):
            ax.set_xlim(limits)

            x = data['T_shift']
            y = data[idx[i]]
            ax.step(x, y, where='pre')
            (bottom, top) = ax.get_ylim()

            ax.text(1.0, 1.0, text[i], horizontalalignment='left', verticalalignment='top', transform=ax.transAxes)

        plt.show()
        
        
    def plot_LC(self, res=64, x_lim=30, limits=[[4600,5500],[2800,3700],[900,1350],[400,700]], savefig=True, show_spectra=False):
        """ Plots LCs 
        
        :res: time resolution of LC. Can be 2, 16, 64, 256, or 2944 (waiting-mode) ms.
        "x_lim: the right boundary of the plotting area
        :limits: list of boundaries [[bottom, top],...] on the y-axes 
        :savefig: saves the figure for GCN
        """

        assert res in [2,16,64,256,2944], "The time resolution should be 2, 16, 64, 256, or 2944 ms!"

        y_title = "counts/{:0.3f} s".format(res/1000.)
        self._time = float(self._time)

        """ Reading LC """

        if res == 2944:
            lc_file = f"{self._path2data}kw{self._date}_{self._short_time}.bga"
            data = pd.read_csv(lc_file, sep='\s+',header=0)
            data = pd.DataFrame(data)
            data['T_shift'] = data['T'] - self._time -2.944
            data['Sum_S1'] = data['S1G1'] + data['S1G2'] + data['S1G3']
            data['Sum_S2'] = data['S2G1'] + data['S2G2'] + data['S2G3']

        else:
            lc_file = f"{self._path2data}kw{self._date}_{self._short_time}_rc{res}.thr"
            data = pd.read_csv(lc_file, sep='\t',header=None)
            data = pd.DataFrame(data)

        """ Reading the bg levels """
        bg_G1 = dict()
        bg_G2 = dict()
        bg_G3 = dict()
        bg_G1G2G3 = dict()

        with open(f"{self._path2data}kw{self._date}_{self._short_time}_BG.log", 'r') as f:
            next(f)
            (undef, undef, undef, undef, bg_G1['S1'], undef, undef) = map(float, f.readline().split())
            (undef, undef, undef, undef, bg_G2['S1'], undef, undef) = map(float, f.readline().split())
            (undef, undef, undef, undef, bg_G3['S1'], undef, undef) = map(float, f.readline().split())
            next(f)
            next(f)
            (undef, undef, undef, undef, bg_G1['S2'], undef, undef) = map(float, f.readline().split())
            (undef, undef, undef, undef, bg_G2['S2'], undef, undef) = map(float, f.readline().split())
            (undef, undef, undef, undef, bg_G3['S2'], undef, undef) = map(float, f.readline().split())

        for i in ('S1', 'S2'): 
            bg_G1[i] *= res/1000.
            bg_G2[i] *= res/1000.
            bg_G3[i] *= res/1000.
            bg_G1G2G3[i] = bg_G1[i] + bg_G2[i] + bg_G3[i]

        """ Plot set up """
        fig, (ax123, ax1, ax2, ax3) = plt.subplots(4, figsize=(10,15)) # sharex=True
        fig.suptitle(f"KONUS-WIND GRB {self._burst_name}\nT0 = {self._time} s UT ({self._UT})\n{self._det}", 
                     fontsize=20)
        plt.xlabel("$T - T_{0}$ (s)", fontsize=14)

        plt.locator_params(axis='y', nbins=10)
        plt.locator_params(axis='x', nbins=10)

        fig.text(0.04, 0.5, y_title, va='center', rotation='vertical', fontsize=14)

        (G1, G2, G3, G4) = self.G1G2G3_bounds()

        text = [f"G1G2G3: {G1}-{G4} keV", f"G1: {G1}-{G2} keV", f"G2: {G2}-{G3} keV", f"G3: {G3}-{G4} keV"]

        bg_ch = [bg_G1, bg_G2, bg_G3, bg_G1G2G3]
        
        if res == 2944:
            idx = [self._det + 'G1', 
                   self._det + 'G2',
                   self._det + 'G3',
                   'Sum_' + self._det
                  ]
            
            for i, ax in enumerate([ax1, ax2, ax3, ax123]):
                ax.set_xlim(-200, 245)
                ax.set_ylim(limits[(i+1)%4])

                x = data['T_shift']
                y = data[idx[i]]
                ax.step(x, y, where='pre')

                ax.hlines(y=bg_ch[i][self._det], xmin=-200, xmax=245, colors='b', linestyles='dashed')
                ax.text(-150, limits[(i+1)%4][1] - 0.15*(limits[(i+1)%4][1] - limits[(i+1)%4][0]), text[(i+1)%4], fontsize=14)

                if show_spectra:
                    for sp in 
                    ax.vlines(x=sp, xmin=-200, xmax=245, colors='b', linestyles='dashed')


                for time in self._spectra:
                    ax.axvline(x=time, color='r', linestyle='dotted')
                                
        else:
            plt.autoscale()
            
            coeff = [0.7, 0.7, 0.7, 0.6]
            
            for i, ax in enumerate([ax1, ax2, ax3, ax123]):
                x = data.iloc[:,0]
                y = data.iloc[:,i+1]
                ax.step(x, y, where='pre')
                (bottom, top) = ax.get_ylim()
                ax.set_xlim(-1, x_lim)

                ax.hlines(y=bg_ch[i][self._det], xmin=-1, xmax=x_lim, colors='b', linestyles='dashed')
                ax.text(x_lim*coeff[i], top - 0.15*(top - bottom), text[(i+1)%4], fontsize=14)

                for time in self._spectra:
                    ax.axvline(x=time, color='r', linestyle='dotted')

        plt.show()

        """ Saving to files """
        if  savefig:
            fig.savefig(f"{self._path2data}kw{self._date}_{self._short_time}_{res}.png")
            fig.savefig(f"{self._path2data}kw{self._date}_{self._short_time}_{res}.eps")


    def plot_bg_spectra(self, sp_i=50, sp_f=60, limits=[], savefig=False):
        """ Plots LCs made from the spectral data to help choose the background spectra

        :sp_i: start of the background time interval
        :sp_f: stop of the calibration time interval
        :limits: list of boundaries [[bottom, top],...] on the y-axes 
        :savefig: saves the figure
        """

        data = pd.read_csv(f"{self._path2data}kw{self._date}_{self._short_time}_PHA1_3W.thr", sep='\t',header=0)
        data = pd.DataFrame(data)
        
        data2 = pd.read_csv(f"{self._path2data}kw{self._date}_{self._short_time}_PHA2_4W.dat", sep='\t',header=0, skiprows=[1])
        data2 = pd.DataFrame(data2)
        
        data['G4rate'] = data2['G1rate']
        data['G5rate'] = data2['G2rate']
        data['G6rate'] = data2['G3rate']
        data['G7rate'] = data2['G4rate']
        data['dT_PHA2'] = data2['dT']
        
        data.set_index('Nsp',inplace=True)
        
        if data['dT'].equals(data['dT_PHA2']):
            print("The sp. acc. time for PHA1 coincides with the sp. acc. time for PHA2")
        else:
            diff = data['dT'] - data['dT_PHA2']
            print("Discrepancy in sp. acc. time for PHA1 and PHA2:")
            print(diff[diff != 0])
            
        """ Calculating the mean CR """
        bg_rate = []
        
        for ch in range (0, 7):
            acc_time = 0
            total_count = 0
            
            for i in range(sp_i, sp_f + 1):
                idx = 'G' + str(ch+1) + 'rate'
                total_count += data[idx][i] * data['dT'][i]
                acc_time += data['dT'][i]
            
            bg_rate.append(total_count / acc_time)
      
        """ Plotting """
        if len(limits) == 0:
            nomin_scale = [0,50,40,30,25,20,15,15]
            scale = [max(a*0.12, b) for a, b in zip(bg_rate, nomin_scale)]

            bottom = [a - b for a, b in zip(bg_rate, scale)]
            top = [a + b for a, b in zip(bg_rate, scale)]
            limits = [[a, b] for a, b in zip(bottom, top)]

        fig, (ax1, ax2, ax3,  ax4, ax5, ax6, ax7) = plt.subplots(7, figsize=(10,15)) # sharex=True
        ax = [ax1, ax2, ax3,  ax4, ax5, ax6, ax7]
        fig.suptitle(f"{self._burst_name} background spectra {sp_i}-{sp_f}", fontsize=20)

        plt.xlabel("$T - T_{0}$ (s)", fontsize=14)
        plt.autoscale()
            
        x = data['Tstart']
        
        for i in range(0, 7):
            idx = 'G' + str(i+1) + 'rate'
            y = data[idx]
            ax[i].set_ylim(limits[i])
            ax[i].step(x, y, where='pre')
            ax[i].hlines(y=bg_rate[i], xmin=-10, xmax=500, colors='r', linestyles='dashed')
            ax[i].text(10, limits[i][1] - 0.15*(limits[i][1] - limits[i][0]), f'G{i}', fontsize=14)
            ax[i].axvline(x=data['Tstart'][sp_i], color='r')
            ax[i].axvline(x=data['Tstop'][sp_f], color='r')
            
        plt.show()

        """ Saving to files """
        if savefig: 
            fig.savefig(f"{self._path2data}kw{self._date}_{self._short_time}_bg_spectra.pdf")


    def plot_bg_spectra_interactive(self, sp_i=50, sp_f=60, limits=[]):
        """ Makes interactive plot of the LC obtained from spectral data to facilitate the background spectra selection 
        
        :sp_i: start of the background time interval
        :sp_f: stop of the background time interval
        :limits: list of boundaries [[bottom, top],...] on the y-axes 
        """
     
        self._data = pd.read_csv(f"{self._path2data}kw{self._date}_{self._short_time}_PHA1_3W.thr", sep='\t',header=0)
        self._data = pd.DataFrame(self._data)
        
        data2 = pd.read_csv(f"{self._path2data}kw{self._date}_{self._short_time}_PHA2_4W.dat", sep='\t',header=0, skiprows=[1])
        data2 = pd.DataFrame(data2)
        
        self._data['G4rate'] = data2['G1rate']
        self._data['G5rate'] = data2['G2rate']
        self._data['G6rate'] = data2['G3rate']
        self._data['G7rate'] = data2['G4rate']
        self._data['dT_PHA2'] = data2['dT']
        
        self._data.set_index('Nsp',inplace=True)
        
        if self._data['dT'].equals(self._data['dT_PHA2']):
            print("The sp. acc. time for PHA1 coincides with the sp. acc. time for PHA2")

        else:
            diff = self._data['dT'] - self._data['dT_PHA2']
            print("Discrepancy in sp. acc. time for PHA1 and PHA2:")
            print(diff[diff != 0])
            
        """ Calculating the mean CR """
        bg_rate = []
        bg_rate.append(0)
        
        for ch in range (1, 8):
            acc_time = 0
            total_count = 0
            
            for i in range(sp_i, sp_f + 1):
                idx = 'G' + str(ch) + 'rate'
                total_count += self._data[idx][i] * self._data['dT'][i]
                acc_time += self._data['dT'][i]
            
            bg_rate.append(total_count / acc_time)
            
        """ Plotting """
        fig = make_subplots(rows=7, cols=1, 
                            subplot_titles=("G1", "G2", "G3", "G4", "G5", "G6", "G7"), 
                            vertical_spacing=0.02
                           )

        for i in range (1, 8):
            idx = 'G' + str(i) + 'rate'

            fig.append_trace(go.Scatter(
                x=self._data['Tstart'],
                y=self._data[idx],
                text='sp '+self._data[idx].index.astype(str),
                line_shape='hv',
                ), 
                row=i, col=1
                )

            fig.add_hline(y=bg_rate[i], 
                          line_color='yellow', 
                          row=i, col=1
                         )
            
            fig.add_vline(x=self._data['Tstart'][sp_i], 
                          line_color='powderblue', 
                          row=i, col=1
                         )

            fig.add_vline(x=self._data['Tstop'][sp_f], 
                          line_color='powderblue', 
                          row=i, col=1
                         )

            fig.update_xaxes(range=[-10, 500], row=i, col=1)
            if len(limits) > 0:
                fig.update_yaxes(range=limits[i-1], row=i, col=1)

        fig.update_layout(height=2800, width=1000, title_text="LC from spectral data", showlegend=False)
        fig.show()

        
    def pla2fits(self, RA, Dec, reference=None, mission=None):
        """ Makes fits files from pla 

        :RA: GRB coordinate
        :Dec: GRB coordinate
        :reference: a reference to the GCN or paper with the localization 
        :mission: the telescope that localized the burst
        """

        if self._verbose:
            print("Making fits files from pla")

        assert reference is not None, "Please introduce a reference to the localization!"
        assert mission is not None, "Please introduce the telescope that localized the event!"
        
        if not os.path.exists(f"{self._path2data}/spectra"):
            os.mkdir(f"{self._path2data}/spectra")

        os.chdir(f"{self._path2data}/spectra")
        
        os.system(f"{self._path2LD} {self._current_dir}{self._path2soft}/KonusPLA2FITSconsole_mu \
                  --PLA={self._current_dir}{self._path2data}/kw{self._date}_{self._short_time}.pla \
                  --KBM={self._current_dir}{self._path2soft}/wind_al2_forFITS.kbm \
                  --ORB={self._current_dir}{self._path2soft}/BIN/  --RA={RA} --Dec={Dec} \
                  --Type=GRB --LocRef=\'{reference}\' --LocMission={mission}")     

        os.chdir(self._current_dir)

        
    def sum_spectra(self, spectra, bg_spectra):
        """Sums the spectra 

        :spectra: a list of pairs of spectra, e.g., [[1,4], [1,5]] for sp1_4 and sp1_5
        :bg_spectra: background spectra, e.g., [50,60]
        """

        os.chdir(f"{self._path2data}/spectra")

        """Summing the burst spectra"""

        if self._verbose:
            print("Summing the burst spectra")

        for sp in spectra:
            sp_i, sp_f = sp

            os.system(f"{self._path2LD} {self._current_dir}{self._path2soft}SumKonusSpectra --pha=KW{self._date}_T{self._short_time}_1.pha \
                      --sp={sp_i}:{sp_f}")
            os.system(f"{self._path2LD} {self._current_dir}{self._path2soft}SumKonusSpectra --pha=KW{self._date}_T{self._short_time}_2.pha \
                      --sp={sp_i}:{sp_f}")

        """Summing the background spectra"""

        if self._verbose:
            print("Summing the background spectra")

        bg_i, bg_f = bg_spectra

        os.system(f"{self._path2LD} {self._current_dir}{self._path2soft}SumKonusSpectra --pha=KW{self._date}_T{self._short_time}_1.pha \
              --sp={bg_i}:{bg_f} --bg")
        os.system(f"{self._path2LD} {self._current_dir}{self._path2soft}SumKonusSpectra --pha=KW{self._date}_T{self._short_time}_2.pha \
              --sp={bg_i}:{bg_f} --bg")

        os.chdir(self._current_dir)
