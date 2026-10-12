# *************************************************************************************************
# Physical constants
R: float = 8.3145       # Ideal gas constant in J / (mol K)
T: float = 298.15       # Room temperature in Kelvin
F: float = 96485.3329   # Faraday's constant in Coulombs per mole
V_T: float = R*T*1E3/F  # Thermal voltage in millivolts
ne: int = 2             # number of electrons
c0: float = 10.0E-3     # Total AQDS concentration in moles per liter (was 20.0E-3; 2026-08-04)
diff: float = 4.0E-6    # The diffusivity of AQDS in cm^2 / sec
# Kinetic rate constant, AQDS on Au. Units are cm/sec (velocity), confirmed 2026-08-04.
# Was 7.2E-3. Measured at 10 mM; a 1 mM check gave a similar result, so dimerization is
# not significantly affecting the value at 10 mM.
k0: float = 2.0E-5      # The kinetic rate constant of AQDS in cm / sec
