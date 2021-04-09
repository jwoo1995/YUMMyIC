import numpy
import camb

pars=camb.CAMBparams()

om = 0.3111
omb = 0.04897  #baryonic matter
omc = om - omb #cold dark matter
omL = 0.6889 # omega Lambda
omk = 1 - om - omL
"""
pars = camb.CAMBparams()
pars.set_cosmology(H0=67.66, ombh2=0.02242, omch2=  0.11933, tau = 0.0561)
#pars.InitPower.set_params(ns=0.9649, As=2.1*10**-9)

#print(calc_transfers(parmas= pars))
data= camb.get_transfer_functions(pars)

print(data)
"""
pars = camb.CAMBparams()
pars.set_cosmology(H0=67.66, ombh2=0.02242, omch2=0.11933, tau=0.0561)
pars.InitPower.set_params(ns=0.965)
pars.set_matter_power(redshifts=[100], kmax=10)
results= camb.get_results(pars)

trans = results.get_matter_transfer_data()
transdat = trans.transfer_data
twod_transdat = transdat.reshape((212,13,))
print(np.shape(twod_transdat))

np.savetxt('cmab_trans.txt',twod_transdat)
