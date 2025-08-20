# PROTAC-shotgun
Shotgun approch for modeling of POI-PROTAC-E3 ternary complex.

Scripts for FFT-based protein-protein docking is modified on the codes from github repository https://github.com/shannontsmith/fft-docking. The original idea is from the following 1992 PNAS paper:

Katchalski-Katzir, E., et al. (1992). "Molecular surface recognition: determination of geometric fit between proteins and their ligands by correlation techniques." Proc Natl Acad Sci U S A 89(6): 2195-2199.

The code of DSDP is modified to use the input coordinates as the initial guess for conformation sampling. You can find the instructions for installment of DSDP from https://github.com/PKUGaoGroup/DSDP.

The code of ColabDock is also modified to use the input coordinates as the initial guess for conformation sampling. The output from each iteration is saved, and the second stage (prediction stage) of ColabDock is not run. You can find the instructions for installment of ColabDock from https://github.com/JeffSHF/ColabDock.

