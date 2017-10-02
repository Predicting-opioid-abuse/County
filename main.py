# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
Chase Ginther
"""

import pandas as pd
import os
import pandas as pd
import matplotlib.pyplot as plt

os.chdir("C:\\Users\\Chase\\Desktop\\School\\Fall 2017\\CMDA 4684\\County")

#read in data sources
data = pd.read_csv("data\ohio.csv")
unemployment = pd.read_csv("data\OhioUnemploymentReport.csv")
education = pd.read_csv("data\EducationReport.csv")


#Remove any comma's
