""" synchrotron emission model from Barniol and Duran (2013) """
import numpy as np
from sympy import solve
from sympy import symbols


class SEM:
    def __init__(
        self,
        vp=4.0,
        Fvp=1.14,
        p=3,
        dL=90,
        z=0.0206,
        t=246,
        geo='spherical',
        va_gtr_vm=True,
        va=None,
        vm=None,
        save=False,
        name=None,
    ):
        """ 
        This class calculates physical TDE system parameters from observed parameters from radio spectral observations. It uses the equations from Barniol and Duran (2013). These equations assume the Newtonian case in which Gamma = 1

        Parameters:
            - vp is peak frequency in GHz
            - Fvp is peak flux density in mJy
            - p is the powerlaw index
            - dL is luminosity distance in Mpc
            - z is redshift 
            - t is time since jet was launched in days
            - geo, str, is the assumed geometry, can be spherical or conical 
            - va_gtr_vm set True if the synchrotron self absorption frequency, va, is above or equal to vm, the synhcrotron frequency at which the electrons emit. Else, set false. If va and vm cannot be identified in the spectrum set va_gtr_vm = False
            - va and vm only requred if va_gtr_vm = False
            - save: option to write parameters to text file 
            - name: str, name for text file, only required if save = True
        """

        # constants and conversions
        Mpctocm = 3.0857e24
        self.c = 2.998e10  # cm/s
        self.msun = 1.989e33  # g

        if geo == 'spherical':
            self.fA = 1.0
            self.fV = 4.0 / 3.0
        elif geo == 'conical':
            self.fA = 0.1
            self.fV = 4.0 / 3.0

        self.vp = vp  # GHz
        self.Fvp = Fvp  # mJy
        self.d = dL * Mpctocm  # cm
        self.z = 0.0206
        self.t = t * 24 * 60 * 60
        self.geo = geo
        self.save = save
        self.name = name
        self.p = p

        if va_gtr_vm:
            self.eta = 1.0
        else:
            self.eta = va / vm

    def get_Req(self):
        me = 9.10938356e-31
        mp = 1.672621e-27
        eps_e = 0.1
        chi_e = (self.p - 2 / self.p - 1) * eps_e * (mp / me)
        LF = 2 / chi_e + 1

        d = self.d
        z = self.z
        xi = 1 + (1 / eps_e)
        Fp = self.Fvp
        vp = self.vp / 10
        fA = self.fA
        p = self.p

        # Req = (
        #     3.2e15
        #     * self.Fvp ** (9 / 19)
        #     * (self.d / 1e26) ** (18 / 19)
        #     * (self.vp / 10) ** (-1)
        #     * (1 + self.z) ** (-10 / 19)
        #     * self.fA ** (-8 / 19)
        #     * self.fV ** (-1 / 19)
        #     * 4 ** (1 / 19)
        # )
        fV = 4 / 3  # * (1 - 0.9 ** 3)

        eta = 1

        prefac = (
            1e17
            * (21.8 * 525 ** (p - 1)) ** (1 / (13 + 2 * p))
            * chi_e ** ((2 - p) / (13 + 2 * p))
            * LF ** ((p + 8) / (13 + 2 * p))
            * (LF - 1) ** ((2 - p) / (13 + 2 * p))
            * xi ** (1 / (13 + 2 * p))
        )  # * 4**(1/19)

        Req = (
            prefac
            * Fp ** ((6 + p) / (13 + 2 * p))
            * (d / 1e28) ** (2 * (p + 6) / (13 + 2 * p))
            * vp ** (-1)
            * (1 + z) ** (-(19 + 3 * p) / (13 + 2 * p))
            * fA ** (-(5 + p) / (13 + 2 * p))
            * fV ** (-1 / (13 + 2 * p))
            * 4 ** (1 / (13 + 2 * p))
        )
        return Req

    def get_Eeq(self):
        me = 9.10938356e-31
        mp = 1.672621e-27
        eps_e = 0.1
        chi_e = (self.p - 2 / self.p - 1) * eps_e * (mp / me)
        LF = 2 / chi_e + 1
        fA = self.fA
        fV = 4 / 3  # * (1 - 0.9 ** 3)

        d = self.d
        z = self.z
        xi = 1 + (1 / eps_e)
        Fp = self.Fvp
        vp = self.vp / 10
        xi = 1 + (1 / eps_e)
        p = self.p
        # Eeq = (
        #     1.9e46
        #     * self.Fvp ** (23 / 19)
        #     * (self.d / 1e26) ** (46 / 19)
        #     * (self.vp / 10) ** (-1)
        #     * (1 + self.z) ** (-42 / 19)
        #     * self.fA ** (-12 / 19)
        #     * self.fV ** (8 / 19)
        #     * 4 ** (11 / 19)
        # )
        prefac2 = (
            1.3e48
            * 21.8 ** ((-2 * (p + 1)) / (13 + 2 * p))
            * (525 ** (p - 1) * chi_e ** (2 - p)) ** (11 / (13 + 2 * p))
            * LF ** ((-5 * p + 16) / (13 + 2 * p))
            * (LF - 1) ** (-11 * (p - 2) / (13 + 2 * p))
            * xi ** (11 / (13 + 2 * p))
        )

        Eeq = (
            prefac2
            * Fp ** ((14 + 3 * p) / (13 + 2 * p))
            * (d / 1e28) ** (2 * (3 * p + 14) / (13 + 2 * p))
            * vp ** (-1)
            * (1 + z) ** ((-27 + 5 * p) / (13 + 2 * p))
            * fA ** (-(3 * (p + 1)) / (13 + 2 * p))
            * fV ** ((2 * (p + 1)) / (13 + 2 * p))
            * 4 ** (11 / (13 + 2 * p))
        )
        return Eeq

    def get_LF(self):

        return LF

    def get_LF_e(self):

        return LF_e

    def get_Bfield(self, Req):
        B = (
            1.3e-2
            * (
                self.Fvp ** (-2)
                * (self.d / 1e28) ** (-4)
                * (self.vp / 10) ** 5
                * self.eta ** (-10 / 3)
                * (1 + self.z) ** 7
            )
            * (self.fA ** 2 * (Req / 1e17) ** 4)
        )
        return B

    def get_Ne(self, Req):
        Ne = (
            1e54
            * (
                self.Fvp ** 3
                * (self.d / 1e28) ** 6
                * (self.vp / 10) ** (-5)
                * self.eta ** (10 / 3)
                * (1 + self.z) ** (-8)
            )
            * (1 / (self.fA ** 2 * (Req / 1e17) ** 4))
        )
        return Ne

    def get_ambientden(self, Ne, Req):
        V = (
            4 / 3 * np.pi * ((Req ** 3 - (0.9 * Req) ** 3))
        )  # for a spherical shell with radius 0.1Req
        ne = Ne / V
        return ne

    def get_outflow_velocity(self, Req):
        fac = Req * (1 + self.z) / (self.c * self.t)
        x = symbols('x')
        res = solve(x / (1 - x) - fac, x)
        beta_ej = res[0]

        return beta_ej

    def get_outflow_mass(self, Eeq, beta_ej):
        # kinetic energy: E = 0.5 m v^2
        M_ej = 2 * Eeq / (beta_ej * self.c) ** 2
        return M_ej

    def do_analysis(self):

        Req = self.get_Req()
        Eeq = self.get_Eeq()
        Ne = self.get_Ne(Req)
        ne = self.get_ambientden(Ne, Req)
        # beta_ej = self.get_outflow_velocity(Req)
        # M_ej = self.get_outflow_mass(Eeq, beta_ej)
        B = self.get_Bfield(Req)

        print(f'Assuming ' + self.geo + ' geometry..')
        print(f'At time t = {self.t/(24*60*60)} d')
        print('--------------------------------------------------')
        print(f'The energy is: {Eeq} erg')
        print(f'The radius is: {Req} cm')
        print('--------------------------------------------------')
        print(f'For this radius and energy, I find:')
        # print(f'Electron Lorentz factor = {LF_e}')
        # print(f'Bulk source Lorentz factor: {LF}')
       # print(f'Outflow velocity: {beta_ej} c')
       # print(f'Outflow mass: {M_ej/self.msun} msun')
        print(f'Ambient density: {ne} cm^-3')
        print(f'Magnetic field: {B} G')
        print('--------------------------------------------------')

        if self.save:
            print('Writing to text file ' + self.name + '.txt..')
            np.savetxt(
                self.name + '.txt',
                [self.t, Req, Eeq, beta_ej, M_ej, B, Ne],
                header='t (d), Req (cm), Eeq (erg), velocity (c), Mass (g), Ambient density (cm^-3), B field (G), electron density',
            )

        return Eeq, Req
