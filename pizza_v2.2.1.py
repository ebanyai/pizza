# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 09:34:58 2015

@author: ebanyai
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

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

def grabData(path):
    """ 
    A function to import the solar abundances, the list of species, the recquired model numbers
    and the .srf files for the given model data set.
    
    The function recquieres one argument, the path to the source files (.srf, species etc.). 
    
    Returns
    -------
    ModelSet - a pandas DataFrame object (easy to use table) with the recquired models and solar abundacnes.
    
    Example
    -------
    >>> model = grabData("/models/m3z1.04/")    
    """
    
    if not path.endswith("/"):
        path += "/"
    # define file names of the several input data files (solar abund, species etc.)
    global fName_species
    global fName_solar
    fName_models = path+"interpulse.dat"
    fList = [f for f in os.listdir(path) if f.endswith(".srf")]
    
    # grab data from the species file
    print("Importing the list of the species...   ",end="")  
    cols_species = ["mass","element","isotope","neutrons"]    
    data_species = pd.read_table(fName_species, names=cols_species, sep=" ", skiprows=1,header=None, skipinitialspace=True)   
    print("DONE")  
    
    # grab data from interpulse file
    print("Importing the list of the recquired models...   ",end="")
    cols_models = ["thermal pulse number","total mass","core mass","envelope mass","?","??","model number"]
    data_models = pd.read_table(fName_models,names=cols_models,sep=" ",header=None,skipinitialspace=True)
    print("DONE")
    
    
     # grab solar abundaces 
    cols_solar = ["isotope","values","mass"]
    data_solar = pd.read_table(fName_solar,names=cols_solar,sep=" ",skipinitialspace=True,index_col=None)
    dump = dict(zip(data_solar["isotope"].tolist(),data_solar["values"].tolist()))
    data_solar = pd.DataFrame(columns=data_solar["isotope"].tolist())
    data_solar = data_solar.append(dump,ignore_index=True)
    
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
    all_models = data_srf["model number"].values.astype(np.int)
    needed_models = data_models["model number"].values.astype(np.int)
    closest_models = [all_models[0]]
    for model in needed_models:
        closest_models.append(min(all_models, key=lambda x:abs(x-model)))
    closest_models.append(all_models[-1])
    print("DONE")
    
    
    # selecting closest models from the srf data    
    init = pd.DataFrame.from_dict({0:{"total mass":np.ceil(data_models["total mass"].iloc[0])}},orient='index')
    last = pd.DataFrame.from_dict({0:{"total mass":data_models["total mass"].iloc[-1]}},orient='index')
    ext_models = pd.concat([init,data_models[["total mass"]],last])
    ext_models["model number"] = closest_models
    ext_models.set_index(["model number"],inplace=True)
    
    # get recquired models
    print("Gathering required data by model numbers... ",end="")
    dump = data_srf[data_srf["model number"].isin(closest_models)].sort(["model number","age"])
    sorted_data = pd.DataFrame(columns=data_srf.columns)
    
    sorted_data = sorted_data.append(dump.iloc[[0]],ignore_index=True)
    for i in range(1,len(dump)-1):
        if int(dump["model number"].irow(i)) != int(dump["model number"].irow(i-1)):
           sorted_data = sorted_data.append(dump.iloc[[i]]) 
    
    sorted_data["model number"] = sorted_data["model number"].astype("int64") 
    sorted_data.set_index(["model number"],inplace=True)
    needed_data = pd.concat([sorted_data,ext_models],axis=1)    
    print("DONE")
    
    print("\nClosest model numbers found in "+path+": ",end="")
    print(closest_models,"\n")
    
    data_species.set_index("isotope", inplace = True)
    return ModelSet(needed_data,data_solar,data_species)
    
    

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
    (Assuming m is a ModelSet object.)
    >>> m.model["c12"] --> returns a DataFrame with the values of C12 for every model numbers.
    >>> m.solar["c12"] --> returns a DataFrame withe the solar abundace value of C12
    >>> m.model["ba138"].loc[8443] --> returns the value of Ba138 in model 8443
    >>> m.model["ba136"].iloc[10] --> returns the value of Ba136 in the 11th row of the model set
    """
    
    def __init__(self,models_df,solar_df,species_df):
        """Initializing class with two attributes: model and solar."""
        
        self.model = models_df
        self.solar = solar_df
        self.species = species_df
    
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
        A pandas DataFrame.
        
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
        A pandas DataFrame.
        
        Examples
        --------
        (Assuming m is a ModelSet object.)
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
        A pandas DataFrame.
        
        Examples
        --------
        (Assuming m is a ModelSet object.)
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
        A pandas DataFrame.
        
        Examples
        --------
        (Assuming m is a ModelSet object.)
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
        A pandas DataFrame.
        
        Examples
        --------
        (Assuming m is a ModelSet object.)
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
        A pandas DataFrame.
        
        Examples
        --------
        (Assuming m is a ModelSet object.)
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
        A pandas DataFrame.
        
        Examples
        --------
        (Assuming m is a ModelSet object.)
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
        A pandas DataFrame.

        Examples
        --------
        (Assuming m is a ModelSet object.)
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
        A pandas DataFrame.
        
        Examples
        --------
        (Assuming m is a ModelSet object.)
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
        A pandas DataFrame.
        
        Examples
        --------
        (Assuming m is a ModelSet object.)
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
        A pandas DataFrame.
        
        Examples
        --------
        (Assuming m is a ModelSet object.)
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
        c: str
            String value of element.
            
        Returns
        -------
        A pandas DataFrame.
        
        Examples
        --------
        (Assuming m is a ModelSet object.)
        >>> m.eYield("fe60") --> returns sum over i for m["fe60"].iloc[i] * (m["total mass"].iloc[i] - m["total mass"].iloc[i-1]) * 60
        >>> m.eYield("pb207") --> returns sum over i for m["pb207"].iloc[i] * (m["total mass"].iloc[i] - m["total mass"].iloc[i-1]) * 207

        """
        result = 0
        for i in range(1,len(self.model)-1):
            result += self.model[e].iloc[i] * (self.model["total mass"].iloc[i-1] - self.model["total mass"].iloc[i]) * self.species.loc[e]["mass"]
        return result

print("\n Use the grabData() function to import data. For example:\n my_model = grabData('/path/to/data/')")
