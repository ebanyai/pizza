  # -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 09:34:58 2015

@author: ebanyai
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import re

# set the float numbers' display style
pd.options.display.float_format = '{:,.5e}'.format

#########################################
##########  SOURCE FOLDERS   ############
#########################################
fName_species = r"species.dat"
fName_solar = r"initial_abundAsplund09.dat"

####################################
########## Create Table ############
####################################
def table(objs):
    """
    Creates a merged table from separate data blocks.
    
    Parameters
    ----------
    objs: list or dict of Series, DataFrame, or Panel objects.
        See pd.concat() function for more details.
        
    Returns
    -------
        Pandas DataFrame
        
    Example
    -------
    >>> table([data1,data2,data3]) returns data1,data2,data3 in table format something like this:
                        data1 data2 data3
                    1    23     38    23
                    2   231     93    83
                    3   737     74    881
                    4    73     973   556
    """
    new = pd.concat(objs,axis=1)
    return new

####################################
##########   Get Data   ############
####################################

def grabData(path,column_mass,column_envelope,column_model,header=None):
    """ 
    A function to import the solar abundances, the list of species, the recquired model numbers
    and the .srf files for the given model data set.
    
    The function recquieres one argument, the path to the source files (.srf, species etc.). 
    
    Parameters
    ----------
    column_mass: Index of the column containing the total mass in the interpulse.dat file

    column_envelope: Index of the column containing the envelope mass in the interpulse.dat file
    
    column_model: Index of the column containing the model number in the interpulse.dat file
    
    header: None or line number of header row. Default is None. Starts from 1.
    
    Returns
    -------
    ModelSet - a pandas DataFrame object (easy to use table) with the recquired models and solar abundacnes.
    
    Example
    -------
    >>> model = grabData("/models/m3z1.04/",0,2,5,)    
    
    Example 2
    ---------
    >>> model = grabData("/models/m3z1.04/",1,3,6,header=1)
    """
    
    if not path.endswith("/"):
        path += "/"
    # define file names of the several input data files (solar abund, species etc.)
    global fName_species
    global fName_solar
    fName_models = path+"interpulse.dat"
    fList = sorted([f for f in os.listdir(path) if f.endswith(".srf")])
    
    # grab data from the species file
    print("Importing the list of the species...   ",end="")  
    cols_species = ["mass","element","isotope","neutrons"]    
    data_species = pd.read_table(fName_species, names=cols_species, sep="\s+", skiprows=1,header=None, skipinitialspace=True)
    print("DONE")  
    
    # grab data from interpulse file
    print("Importing the list of the recquired models...   ",end="")
    data_models = pd.read_table(fName_models,sep="\s+",header=header,skipinitialspace=True)
    print("DONE")
    
     # grab solar abundaces 
    cols_solar = ["isotope","values","mass"]
    data_solar = pd.read_table(fName_solar,names=cols_solar,sep="\s+",skipinitialspace=True,index_col=None)
    dump = dict(zip(data_solar["isotope"].tolist(),data_solar["values"].tolist()))
    data_solar = pd.DataFrame.from_dict({0:dump},orient="index")
    
    # grab data from .srf files
    frames = []
    print("Importing .srf files...\n Imported: ",end="")
    for item in fList:
        frames.append(get_srf(path+item,data_species["isotope"].values))
        print(item+", ",end="")
    
    data_srf = pd.concat(frames,ignore_index=True)
    print(" ...DONE")
    
    # finding closest model numbers
    print("Finding closest models...   ",end="")
    all_models = data_srf["model number"].values.astype(int)
    needed_models = data_models.iloc[:,column_model].values.astype(int)
    closest_models = [all_models[0]]
    for model in needed_models:
        closest_models.append(min(all_models, key=lambda x:abs(x-model)))
    closest_models.append(all_models[-1])
    print("DONE")
    
    
    # selecting closest models from the srf data    
    init = pd.DataFrame.from_dict({0:{"total mass":np.ceil(data_models.iloc[0,column_mass])}},orient='index')
    last = pd.DataFrame.from_dict({0:{"total mass":data_models.iloc[-1,column_mass]}},orient='index')
    mass = pd.DataFrame()
    mass["total mass"] = data_models.iloc[:,column_mass]
    ext_models = pd.concat([init,mass,last])
    ext_models["model number"] = closest_models
    ext_models.set_index(["model number"],inplace=True)
    
    
    # get recquired models
    print("Gathering required data by model numbers... ",end="")
    dump = data_srf[data_srf["model number"].isin(closest_models)].sort_values(["model number","age"])
    sorted_data = pd.DataFrame(columns=data_srf.columns)
    
    sorted_data = pd.concat([dump.iloc[[0]]],ignore_index=True)
    for i in range(1,len(dump)-1):
        if int(dump["model number"].iloc[i]) != int(dump["model number"].iloc[i-1]):
           sorted_data = pd.concat([sorted_data,dump.iloc[[i]]]) 
    
    sorted_data["model number"] = sorted_data["model number"].astype("int64") 
    sorted_data.set_index(["model number"],inplace=True)
    needed_data = pd.concat([sorted_data,ext_models],axis=1)    
    print("DONE")
    
    print("\nClosest model numbers found in "+path+": ",end="")
    print(closest_models,"\n")
    
    data_species.set_index("isotope", inplace = True)
    envelope_mass = data_models.iloc[:,column_envelope]
    return ModelSet(needed_data,data_solar,data_species,envelope_mass)


####################################
########    Function to    #########
########   get .srf files  #########
####################################    
def get_srf(file_name,species):
    data = []
    block = []
    block_start = str(len(species))
    cols = ["species","model number","age"]+species.tolist()
    
    with open(file_name) as f:
        for line in f:
            sline = line.split()
            if not sline:
                pass
            elif sline[0] == block_start and block:
                flattened = [item for subblock in block for item in subblock]
                data.append(flattened)
                del block[:]
                block.append(sline)
            else:
                block.append(sline)
    df = pd.DataFrame(data,columns=cols,dtype=float)
    return df



####################################
########   ModelSet Class  #########
####################################
class ModelSet():
    """
    A class for models stored as a pandas DataFrame.
    
    Methods
    -------
        - sumIsotopes(element): sums isotopes for an element 
        - ratioSol(element): calculates the abundace ratio of an element between the model and the solar values
        - ratioIni(element): calculates the abundace ratio of an element between the different model stages and the initial values
        - ratioRelSolar(): calculates two elements ratio relative to their solar ratio
        - delta(element_1,element_2): calculates: { [ ( element_1/element_2 )_model / ( element_1/elemen_2 )_solar ] - 1 } * 1000
        - epsilon(C): calculates the value of log10( C / p ) + 12
        - cPerH(C): calculates the value of log10( C / p_model ) - log10( C/ p_solar)
        - cPerFe(C): calculates the value of log10( C / Fe_model ) - log10( C / Fe_solar )
		- eYield(isotope): Calculates the total mass of isotope lost in the winds.
        
    Examples
    --------
    # Assuming m is a ModelSet object.
    >>> m.model["c12"] --> returns a DataFrame with the values of C12 for every model numbers.
    >>> m.solar["c12"] --> returns a DataFrame withe the solar abundace value of C12
    >>> m.model["ba138"].loc[8443] --> returns the value of Ba138 in model 8443
    >>> m.model["ba136"].iloc[10] --> returns the value of Ba136 in the 11th row of the model set
    """
    
    def __init__(self,models_df,solar_df,species_df,envelope_mass):
        """Initializing class with two attributes: model and solar."""
        
        self.model = models_df
        self.solar = solar_df
        self.species = species_df
        self.envelope_mass = envelope_mass
    
    def sumIsotopes(self,element,solar=False):
        """ 
        Sums isotopes of an element. The default is to sum isotpes for the model.
        
        Parameters
        ----------
        element: str
            Element to search for.
        solar: boolean, optional
            Use True for solar values.
            
        Returns
        -------
        result : pandas DataFrame
        
        Examples
        --------
        (Assuming m is a ModelSet object.)
        >>> m.sumIsotopes("h") --> returns the sum of H (column "p" and "d")
        >>> m.sumIsotopes("c",solar=True) --> returns the sum of the isotopes of C for solar abundaces.
        """
        
        if solar:
            if element == "h":
                result = self.solar["p"]
            else:
                regexp = "^"+element+"[0-9]{1,3}"
                result = self.solar.filter(regex=regexp).sum(axis=1)
                return result
        else:
            if element == "h":
                result = self.model[["p","d"]].sum(axis=1)
            else:
                regexp = "^"+element+"[0-9]{1,3}"
                result = self.model.filter(regex=regexp).sum(axis=1)
                return result

    def ratioSol(self,element,summed=False):
        """ 
        Calculates the ratio of an element abundace in the model to the solar value.
        
        Parameters
        ----------
        element: str
            Element to search for.
        summed: boolean, optional
            Use True if summed isotopes are used.
            
        Returns
        -------
        ratio : pandas DataFrame
        
        Examples
        --------
        # Assuming m is a ModelSet object.f
        >>> m.ratioSol("c12") --> returns the ratio for C12 
        >>> m.ratioSol("c",summed=True) --> returns ratio for the summed isotopes of C 
        """
        
        if summed:
            ratio = self.sumIsotopes(element) / self.sumIsotopes(element,solar=True).iloc[0]
        else:
            ratio = self.model[element] / self.solar[element].iloc[0]
        return ratio
    
    def ratioIni(self,element,summed=False):
        """ 
        Calculates the ratio of an element abundace in the model to the initial value.
        
        Parameters
        ----------
        element: str
            Element to search for.
        summed: boolean, optional
            Use True if summed isotopes are used.
            
        Returns
        -------
        ratio : pandas DataFrame
        
        Examples
        --------
        # Assuming m is a ModelSet object.
        >>> m.ratioIni("c13") --> returns the ratio for C13
        >>> m.ratioIni("c",summed=True) --> returns ratio for the summed isotopes of C 
        """
        
        if summed:
            ratio = self.sumIsotopes(element) / self.sumIsotopes(element).iloc[0]
        else:
            ratio = self.model[element] / self.model[element].iloc[0]
        return ratio
    
    def ratioRelSol(self,element1,element2):
        """ 
        Calculates two elements ratio relative to their solar ratio.
        
        Parameters
        ----------
        element1: str
            Element in the numerator.
        element2: str
            Element in the denominator.
            
        Returns
        -------
        ratio : pandas DataFrame
        
        Examples
        --------
        # Assuming m is a ModelSet object.
        >>> m.ratioRelSol("c13","c12") --> returns the relative ratio of C13/C12 for all model numbers 
        >>> m.ratioRelSol("ba138","ba136").loc[4020] --> returns the relative ratio of Ba138/BaC136 for model number 4020
        """
        
        ratio = self.model[element1]/self.model[element2]*self.solar[element2].loc[0]/self.solar[element1].loc[0]
        return ratio     

    def logratioRelSol(self,element1,element2):
        """ 
        Calculates two elements ratio relative to their solar ratio and then takes the log10.
        
        Parameters
        ----------
        element1: str
            Element in the numerator.
        element2: str
            Element in the denominator.
            
        Returns
        -------
        ratio : pandas DataFrame
        
        Examples
        --------
        # Assuming m is a ModelSet object.
        >>> m.logratioRelSol("c13","c12") --> returns the relative ratio of C13/C12 for all model numbers in log10
        >>> m.logratioRelSol("ba138","ba136").loc[4020] --> returns the relative ratio of Ba138/BaC136 in log10 for model number 4020
        """
        
        ratio = np.log10(self.model[element1]/self.model[element2]*self.solar[element2].loc[0]/self.solar[element1].loc[0])
        return ratio     

    def ratioTwo(self,element1,element2):
        """ 
        Calculates two elements ratios.
        
        Parameters
        ----------
        element1: str
            Element in the numerator.
        element2: str
            Element in the denominator.
            
        Returns
        -------
        result : pandas DataFrame
        
        Examples
        --------
        # Assuming m is a ModelSet object.
        >>> m.ratioTwo("c13","c12") --> returns the ratio of C13/C12 for all model numbers 
        >>> m.ratioTwo("ba138","ba136").loc[4020] --> returns the ratio of Ba138/BaC136 for model number 4020
        """
        
        ratio = self.model[element1]/self.model[element2]
        return ratio     
        
    def delta(self,element1,element2):
        """ 
        Calculates delta for two elements.
        
        Parameters
        ----------
        element1: str
            Element in the numerator.
        element2: str
            Element in the denominator.
            
        Returns
        -------
        result : pandas DataFrame
        
        Examples
        --------
        # Assuming m is a ModelSet object.
        >>> m.delta("c13","c12") --> returns the delta value of C13/C12 for all model numbers 
        >>> m.delta("ba138","ba136").loc[4020] --> returns the delta value of Ba138/BaC136 for model number 4020
        """
        
        delta = (self.model[element1]/self.model[element2]*self.solar[element2].loc[0]/self.solar[element1].loc[0]-1)*1000
        return delta 
     
    def deltaspinel(self,element1,element1a,element2):
        """
        Calculates delta for Mg26 including 25*al26

        Parameters
        ----------
        element1: str
            Element in the numerator.
        element1a: str
            Element to add in the numerator.
        element2: str
            Element in the denominator.

        Returns
        -------
        result : pandas DataFrame

        Examples
        --------
        # Assuming m is a ModelSet object.
        >>> m.delta("mg26","al-6","mg24") --> returns the delta value of mg26+25*al-6 for all model numbers
        """

        deltaspinel = ((self.model[element1]+(25.*self.model[element1a]))/self.model[element2]*self.solar[element2].loc[0]/self.solar[element1].loc[0]-1)*1000
        return deltaspinel

    def epsilon(self,c):
        """ 
        Calculates epsilon for the given element. log10(C/p) + 12
        
        Parameters
        ----------
        c: str
            String value of element.
            
        Returns
        -------
        result : pandas DataFrame.
        
        Examples
        --------
        # Assuming m is a ModelSet object.
        >>> m.epsilon("ca") --> returns epsilon value of Ca
        >>> m.epsilon("ca").iloc[0] --> returns epsilon value of Ca for the first model in model set.

        """
        
        
        epsilon = np.log10(self.sumIsotopes(c)/self.model["p"])+12
        return epsilon
    
    def cPerH(self,c):
        """ 
        Calculates [C/H] = log10(C/H)_model - log10(C/H)_solar for the given C value.
        
        Parameters
        ----------
        c: str
            String value of element.
            
        Returns
        -------
        result : pandas DataFrame.
        
        Examples
        --------
        # Assuming m is a ModelSet object.
        >>> m.cPerH(2.3e-03) --> returns [C/H] for C = 2.3e-03
        >>> m.cPerH(2.3e-03).iloc[0] --> returns [C/H] for C = 2.3e-03 for the first model in model set.

        """
        
        mC = self.sumIsotopes(c,solar=True).iloc[0]
        sC = self.sumIsotopes(c)
        result = np.log10(mC/self.model["p"]) - np.log10(sC/self.solar["p"].iloc[0])
        return result
        
    def cPerFe(self,c):
        """ 
        Calculates [C/Fe] = log10(C/Fe)_model - log10(C/Fe)_solar for the given element.
        
        Parameters
        ----------
        c: str
            String value of element.
            
        Returns
        -------
        result : pandas DataFrame
        
        Examples
        --------
        # Assuming m is a ModelSet object.
        >>> m.cPerFe(na) --> returns [Na/Fe] 
        >>> m.cPerFe(2.3e-03).iloc[0] --> returns [C/Fe] for C = 2.3e-03 for the first model in model set.

        """
        sC = self.sumIsotopes(c,solar=True).iloc[0]
        sFe = self.sumIsotopes("fe",solar=True).iloc[0]
        mC = self.sumIsotopes(c)
        mFe = self.sumIsotopes("fe")
        result = np.log10(mC/mFe) - np.log10(sC/sFe)
        return result
    
    def eYield(self,e):
        """ 
        Calculates the total mass of each isotopes lost in the winds.
        
        Parameters
        ----------
        e: str
            String value of element.
            
        Returns
        -------
        result : pandas DataFrame
        
        Examples
        --------
        # Assuming m is a ModelSet object.
        >>> m.eYield("fe60") --> returns sum over i for m["fe60"].iloc[i] * (m["total mass"].iloc[i] - m["total mass"].iloc[i-1]) * 60
        >>> m.eYield("pb207") --> returns sum over i for m["pb207"].iloc[i] * (m["total mass"].iloc[i] - m["total mass"].iloc[i-1]) * 207

        """
        result = 0
        for i in range(2,len(self.model)-1):
            env_mass = self.model[e].iloc[-1] *  self.envelope_mass.iloc[-1]*self.species.loc[e]["mass"]
            tot_mass = self.model[e].iloc[i] * (self.model["total mass"].iloc[i-1] - self.model["total mass"].iloc[i]) * self.species.loc[e]["mass"]
            result = tot_mass + env_mass
        return result
    
    def cSMP(self,x):
        """
        C_SMP = C_solar + (C_SM * x)
        
        Calculates C_SMP for every element. Excludes elements missing from either the solar or the model abundances.

        Parameters
        ----------
        x : float
            Value of x parameter in equation C_SMP = C_solar + (C_SM * x).

        Returns
        -------
        result: pandas DataFrame
        
        Examples
        --------
        # Assuming m is a ModelSet object.
        >>> m.cSMP(0.9)

        """
        isotopes = self.species.index.tolist()[1:]
        
        result = self.model.copy(deep=True)
        result[isotopes] = result[isotopes] * x
        
        temp = result[isotopes].apply(lambda x: x+self.solar.iloc[0], axis=1)
        result[isotopes] = temp[isotopes]
        result = result.dropna(axis=1, how='all')
        
        
        return result
    
    def cSMPiso(self,x,iso,m=1):
        """
        C_SMPiso(x,iso) = C_solar[iso] + (C_SM[iso] * x * m)
        
        Calculates C_SMP for only the selected isotope.

        Parameters
        ----------
        x : float
            Value of x parameter in equation C_SMPiso(x,iso) = C_solar[iso] + (C_SM[iso] * x * m).
        iso : string
            String value of the isotope.
        m : float
            Correction value for the given isotope

        Returns
        -------
        result:  pandas DataFrame
        
        Examples
        --------
        # Assuming m is a ModelSet object.
        >>> m.cSMP(0.9,"bi209")

        """
        
        
        try:
            result = pd.DataFrame()
            result[iso] = self.model[iso] * x * m
            #result[iso] = result[iso].apply(lambda x: x+self.solar[iso])
            result[iso] = result[iso] + self.solar[iso].iloc[0]
            return result
        except KeyError:
            print("Isotope {} is missing.".format(iso))
        
    
    def rSMP(self,x,iso_i,iso_j,m_i=1,m_j=1):
        """
        rSMP(x,iso_i,iso_j) = C_SMPiso(x,iso_i) / C_SMPiso(x,iso_j)
        
        Calculates the ratio of C_SMPiso(iso_i) and C_SMPiso(iso_j). 

        Parameters
        ----------
        x : float
            Value of x parameter in equation C_SMPiso = C_solar[iso] + (C_SM[iso] * x).
        iso_i, iso_j : string
            String values of isotopes (i, j)
        m_i, m_j : float
            Correction value for the given isotope

        Returns
        -------
        result : pandas DataFrame
        
        Examples
        --------
        # Assuming m is a ModelSet object.
        >>> m.rSMP(1.9,"pb205","pb206")
        """
        csmp_i = pd.DataFrame(columns=["ratio"])
        csmp_j = pd.DataFrame(columns=["ratio"])
        csmp_i["ratio"] = self.cSMPiso(x,iso_i,m=m_i)
        csmp_j["ratio"] = self.cSMPiso(x,iso_j,m=m_j)
        
        
        result = csmp_i / csmp_j
        
        return result
    
        
    def rSTD(self,iso_i,iso_j):
        """
        rSTD(iso_i,iso_j) = STD(iso_i) / STD(iso_j)
        
        Calculates the ratio of solar abundance of iso_i and iso_j.

        Parameters
        ----------
        iso_i, iso_j : string
            String values of isotopes (i, j)
 
        Returns
        -------
        result : pandas DataFrame.
        
        Examples
        --------
        # Assuming m is a ModelSet object.
        >>> m.rSTD(1.9,"pb205","pb206")

        """
       
        #TODO iso check
       
        rstd_i = pd.DataFrame(columns=["ratio"])
        rstd_j = pd.DataFrame(columns=["ratio"])
       
        rstd_i["ratio"] = self.solar[iso_i]
        rstd_j["ratio"] = self.solar[iso_j]
        result =  rstd_i / rstd_j
       
        return result
   
    def Q(self,iso_i,iso_j,iso_k):
        """
        Substracts i, j and k values from the isotope names. Then calculates 
        Q = (ln(i) - ln(j)) / (ln(k)- ln(j)) 

        Parameters
        ----------
        iso_i, iso_j, iso_k : string
            String values of isotopes (i, j, k)

        Returns
        -------
        result : float
            
        
        Examples
        --------
        # Assuming m is a ModelSet object.
        >>> m.Q("pb205","pb206","pb204")
        0.4987804829683338
        """
        
        i = self.getIsoNum(iso_i)
        j = self.getIsoNum(iso_j)
        k = self.getIsoNum(iso_k)
        
        result = (np.log(i) - np.log(j)) / (np.log(k) - np.log(j)) 
        return result
    
    
        
    def epsi(self,iso_i,iso_j,iso_k,x,m_i=1,m_j=1,m_k=1):
        """
        epsi(iso_i,iso_j,iso_k,x) = [ rSMP(iso_i,iso_j) / rSTD(iso_i,iso_j) *
                                      rSMP(iso_k,iso_j) / rSTD(iso_k,iso_j)^(-Q) ] * 10^4

        Parameters
        ----------
        iso_i, iso_j, iso_k : string
            String values of isotopes (i, j, k)
        x : float
            Value of x parameter in equation C_SMPiso(x,iso) = C_solar[iso] + (C_SM[iso] * x * m).
        m_i,m_j_,m_k: float
            multiplication value for iso_i, iso_j and iso_k

        Returns
        -------
        result : pandas DataFrame
        
        Examples
        --------
        # Assuming m is a ModelSet object.
        >>> m.epsi("pb208","pb207","bi209",0.8)
        
        >>> m.epsi("pb208","pb207","bi209",0.8,m_i=1.3,m_k=0.4)
            
        """
        q = self.Q(iso_i,iso_j,iso_k)
        
        rR_ij = self.rSMP(x,iso_i,iso_j,m_i=m_i,m_j=m_j) / self.rSTD(iso_i,iso_j)["ratio"].iloc[0]
        rR_kj = self.rSMP(x,iso_k,iso_j,m_i=m_k,m_j=m_j) / self.rSTD(iso_k,iso_j)["ratio"].iloc[0]
        
        rR_kj_Q = rR_kj ** (-q)
        
        result =(rR_ij*rR_kj_Q - 1) * 10e+4
        
        return result
    
    
    def epsiLin(self,iso_i,iso_j,iso_k,x,m_i=1,m_j=1,m_k=1):
        """
        epsi(iso_i,iso_j,iso_k,x) = [ (rSMP(iso_i,iso_j) / rSTD(iso_i,iso_j) - 1) -
                                      Q * (rSMP(iso_k,iso_j) / rSTD(iso_k,iso_j)) -1) ] * 10^4

        Parameters
        ----------
        iso_i, iso_j, iso_k : string
            String values of isotopes (i, j, k)
        x : float
            Value of x parameter in equation C_SMPiso(x,iso) = C_solar[iso] + (C_SM[iso] * x * m).
        m_i,m_j_,m_k: float
            multiplication value for iso_i, iso_j and iso_k
        

        Returns
        -------
        result : pandas DataFrame
            
        Examples
        --------
        # Assuming m is a ModelSet object.
        >>> m.epsiLin("pb208","pb207","bi209",0.8)
        
        >>> m.epsiLin("pb208","pb207","bi209",0.8,m_i=1.3,m_k=0.4)
        """
        q = self.Q(iso_i,iso_j,iso_k)
        
        rR_ij = self.rSMP(x,iso_i,iso_j,m_i=m_i,m_j=m_j) / self.rSTD(iso_i,iso_j)["ratio"].iloc[0]
        rR_kj = self.rSMP(x,iso_k,iso_j,m_i=m_k,m_j=m_j) / self.rSTD(iso_k,iso_j)["ratio"].iloc[0]
        
        result = ((rR_ij-1) - q*(rR_kj-1))*10e+4
        
        
        return result
    
        
    def getIsoNum(self,isotope):
        """
        Substracts the isotope number from the isotope string.

        Parameters
        ----------
        isotope : string
            String value of the isotope.

        Returns
        -------
        result : int
            
        Examples
        --------
        >>> m.getIsoNum("pb205")
        205

        """
        x = re.search("[0-9]+",isotope)
        result = int(x.group())
        
        return result
    
    def addIsotope(self,iso_name,iso_value):
        """
        

        Parameters
        ----------
        iso_name : string
            string value of isotope.
        iso_value : float
            abundance value of the isotope.

        Returns
        -------
        None.

        """
        self.model[iso_name] = iso_value
        return

    

print("\n Use the grabData() function to import data. For example:\n my_model = grabData('/path/to/data/', column_mass, column_model,header=1)")