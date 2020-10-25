#
# This is the minimal Biogeme script file in order to estimate the
# parameters of the simple example. The variable "loglike" must be
# properly defined by replacing the ... by the proper formula.
# It is strongly advised to use intermediary variables so that the
# code is easier to read and debug.
#
# Michel Bierlaire
# Tue Aug  4 19:46:01 2020
#

# Import the packages
import pandas as pd
import biogeme.database as db
import biogeme.biogeme as bio
from biogeme.expressions import Beta, log

pandas = pd.read_table("small.dat")
database = db.Database("small", pandas)

# Import the name of the columns to be used as variables
globals().update(database.variables)

# Define the parameters to be estimated
pi1 = Beta('pi1', 0.5, 0.0001, 1, 0)
pi2 = Beta('pi2', 0.5, 0.0001, 1, 0)
pi3 = Beta('pi3', 0.5, 0.0001, 1, 0)

# You may want to define here intermediary variables
# Example: 
one = pi1 + pi2 + pi3

# The contribution of each observation to the log likelihood function
# must be defined here
loglike = ...

# We create an instance of Biogeme, combining the model and the data
biogeme  = bio.BIOGEME(database,loglike)
biogeme.modelName = "maxlike"

# We estimate the parameters
results = biogeme.estimate()

# Get the results in a pandas table
pandasResults = results.getEstimatedParameters()
print(pandasResults)