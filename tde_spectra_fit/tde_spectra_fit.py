# -*- coding: utf-8 -*-
import numpy as np
import emcee
import matplotlib.pyplot as plt
import corner
from IPython.display import display, Math


"""Main module."""

# Example data:
flux_density = np.array([32.1, 28.6, 47, 51, 51, 67.6, 61.5]) * 1e-3  # mJy
u_flux_density_up = np.array([1, 1, 16, 11, 10, 7.3, 9.1]) * 1e-3  # mJy
u_flux_density_low = np.array([30, 30, 16, 11, 10, 7.3, 9.1]) * 1e-3  # mJy
frequency = np.array([2.25, 3.5, 5.04, 6.13, 7.06, 9, 11])  # GHz
delta_t = np.array([20, 20, 20, 20, 20])  # days

# flux_density = np.array([2.18, 2.12, 2.13, 2.00, 1.84, 1.56, 1.26, 1.06, 0.84,\
#                0.73, 0.59, 0.44, 0.30]) #mJy
# u_flux_density = np.array([0.08, 0.10, 0.09, 0.05, 0.03, 0.03, 0.03, 0.02, 0.04,\
#                  0.02, 0.02, 0.09, 0.04])
# frequency = np.array([1.4, 1.5, 1.8, 2.6, 3.4, 5.0, 7.1, 8.5, 11.0, 13.5, 16.0,\
#             19.2, 24.5]) #GHz
# delta_t = np.array([246.25, 246.25, 246.25, 246.25, 246.25, 246.25, 246.25,\
#           247.21, 247.21, 247.21, 247.21, 247.21, 247.21]) #days


class TDE_fit:
    def __init__(
        self,
        fd=flux_density,
        fd_err_low=u_flux_density_low,
        fd_err_up=u_flux_density_up,
        frequency=frequency,
        break_number=5,
        quiescent_flux_density=None,
        name='High_Sparrow_Oct2020',
        nsteps=10000,
        nwalkers=400,
        initial=(2.21, 2.68, 2.8),
    ):

        """ This class takes in a radio TDE spectrum and uses emcee to fit a powerlaw to the data to determine the peak frequency, peak flux density, and powerlaw index, p.

        Parameters:
        fd: array, flux_density measurements. Unit: mJy. 
        fd_err_low: array, lower flux density errors. Unit: mJy. 
        fd_err_up: array of upper flux density errors, can be the same as lower array. Unit: mJy
        frequency: array, frequencies corresponding to the flux densities. Unit: GHz
        break_number: Integer, can take value of 1-11 that corresponds to the spectral break you wish to model from Granot & Sari 2002, ApJ, 568, 2, Figure 1. Usually break 2 or 5 are used. 
        quiescent_flux_density: array or None, option to subtract non-TDE radio emission from host galaxy from the input flux densities. Requires an array of quiescent flux densities the same length as the fd array. Set to None if no quiescent emission. Unit: mJy
        name: string, name that you want output to be saved under 
        nsteps: integer, number of steps you want to run emcee for
        nwalkers: integer, number of walkers you want emcee to use
        initial: the initial guess for Fvb, vb, and p for the spectrum

        """

        self.fd = fd
        self.fd_err_up = fd_err_up
        self.fd_err_low = fd_err_low
        self.frequency = frequency
        self.break_number = break_number
        self.quiescent_flux_density = quiescent_flux_density
        self.name = name
        self.nsteps = nsteps
        self.nwalkers = nwalkers
        self.ndim = 4
        self.burnin = 150
        self.initial = initial

        if self.quiescent_flux_density is not None:
            self.flux_emission = self.fd - self.quiescent_flux_density
        else:
            self.flux_emission = self.fd

    def init_break(self):
        return self.break_number

    def plot_initial_data(self):

        f = plt.figure(figsize=(8, 7))

        plt.scatter(self.frequency, self.flux_emission, color='k')
        plt.plot(self.frequency, self.flux_emission, color='k')
        plt.errorbar(
            self.frequency,
            self.flux_emission,
            yerr=[self.fd_err_low, self.fd_err_up],
            fmt='.',
            capsize=2,
            color='k',
        )

        plt.legend(loc='best')
        plt.xscale('log')
        plt.yscale('log')

        # plt.axis([1.8, 15,2e-2, 0.1])
        plt.xlabel('Frequency (GHz)')
        plt.ylabel('Flux Density (mJy)')

        plt.savefig(f'{self.name}_rawdata.pdf')

    def run_emcee(self):

        break_number = self.break_number

        def powerlaw(v, Fvb, vb, p):

            if break_number == 1:
                # if vsa < vm (slow cooling)
                print(
                    '**warning** p is not being fitted for this choice of break number'
                )
                beta1 = 2
                beta2 = 1 / 3
                s = 1.64

            if break_number == 2:
                beta1 = 1 / 3
                beta2 = (1 - p) / 2
                s = 1.84 - (0.4 * p)

            if break_number == 3:
                beta1 = (1 - p) / 2
                beta2 = -p / 2
                s = 1.15 - (0.06 * p)

            if break_number == 4:
                beta1 = 2
                beta2 = 5 / 2
                s = 3.44 * p - 1.41

            if break_number == 5:
                # if vm < vsa < vs (slow cooling)
                beta1 = 5 / 2
                beta2 = (1 - p) / 2
                s = 1.47 - (0.21 * p)

            if break_number == 6:
                # if vsa > vm (could be slow or fast cooling)
                beta1 = 5 / 2
                beta2 = -p / 2
                s = 0.94 - 0.14 * p

            if break_number == 7:
                beta1 = 2
                beta2 = 11 / 8
                s = 1.99 - 0.04 * p

            if break_number == 8:
                # if vc < vsa < vm (fast cooling)
                print(
                    '**warning** p is not being fitted for this choice of break number'
                )
                beta1 = 11 / 8
                beta2 = -1 / 2
                s = 0.907

            if break_number == 9:
                beta1 = -1 / 2
                beta2 = -p / 2
                s = 3.34 - 0.82 * p

            if break_number == 10:
                print(
                    '**warning** p is not being fitted for this choice of break number'
                )
                beta1 = 11 / 8
                beta2 = 1 / 3
                s = 1.213

            if break_number == 11:
                # if vsa < vc (fast cooling)
                print(
                    '**warning** p is not being fitted for this choice of break number'
                )
                beta1 = 1 / 3
                beta2 = -1 / 2
                s = 0.597

            Fv1 = Fvb * ((v / vb) ** (-beta1 * s) + (v / vb) ** (-beta2 * s)) ** (
                -1 / s
            )

            return Fv1

        def log_likelihood(theta, x, y, yerr):
            yerrup = yerr[1]
            yerrlow = yerr[0]
            Fvb, vb, p, log_f = theta
            model = powerlaw(x, Fvb, vb, p)
            sigma2 = (yerrup ** 2 + model ** 2 * np.exp(2 * log_f)) + (
                yerrlow ** 2 + model ** 2 * np.exp(2 * log_f)
            ) / 2
            return -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(sigma2))

        def log_prior(theta):
            Fvb, vb, p, log_f = theta
            if 0.1 < Fvb < 1e4 and 0.1 < vb < 5 and 1 < p < 3.5 and -10 < log_f < 10:
                return 0.0
            return -np.inf

        # combine prior and likelihood for log proability:
        def log_probability(theta, x, y, yerr):

            lp = log_prior(theta)
            if not np.isfinite(lp):
                return -np.inf
            return lp + log_likelihood(theta, x, y, yerr)

        nwalkers = self.nwalkers
        nsteps = self.nsteps
        ndim = 4

        # set up data input for emcee:
        x = self.frequency
        y = self.flux_emission
        yerr = [self.fd_err_low, self.fd_err_up]

        # set initial position:
        # Fvb, vb, p, f
        # sol = (1, 9, 2.5, 1)
        sol = (self.initial[0], self.initial[1], self.initial[2], 1)
        pos = sol + 1e-4 * np.random.randn(nwalkers, ndim)
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, log_probability, args=(x, y, yerr)
        )
        sampler.run_mcmc(pos, nsteps, progress=True)

        return sampler

    def powerlaw(self, v, Fvb, vb, p):
        break_number = self.break_number

        if break_number == 1:
            # if vsa < vm (slow cooling)
            print('**warning** p is not being fitted for this choice of break number')
            beta1 = 2
            beta2 = 1 / 3
            s = 1.64

        if break_number == 2:
            beta1 = 1 / 3
            beta2 = (1 - p) / 2
            s = 1.84 - (0.4 * p)

        if break_number == 3:
            beta1 = (1 - p) / 2
            beta2 = -p / 2
            s = 1.15 - (0.06 * p)

        if break_number == 4:
            beta1 = 2
            beta2 = 5 / 2
            s = 3.44 * p - 1.41

        if break_number == 5:
            # if vm < vsa < vs (slow cooling)
            beta1 = 5 / 2
            beta2 = (1 - p) / 2
            s = 1.47 - (0.21 * p)

        if break_number == 6:
            # if vsa > vm (could be slow or fast cooling)
            beta1 = 5 / 2
            beta2 = -p / 2
            s = 0.94 - 0.14 * p

        if break_number == 7:
            beta1 = 2
            beta2 = 11 / 8
            s = 1.99 - 0.04 * p

        if break_number == 8:
            # if vc < vsa < vm (fast cooling)
            print('**warning** p is not being fitted for this choice of break number')
            beta1 = 11 / 8
            beta2 = -1 / 2
            s = 0.907

        if break_number == 9:
            beta1 = -1 / 2
            beta2 = -p / 2
            s = 3.34 - 0.82 * p

        if break_number == 10:
            print('**warning** p is not being fitted for this choice of break number')
            beta1 = 11 / 8
            beta2 = 1 / 3
            s = 1.213

        if break_number == 11:
            # if vsa < vc (fast cooling)
            print('**warning** p is not being fitted for this choice of break number')
            beta1 = 1 / 3
            beta2 = -1 / 2
            s = 0.597

        Fv1 = Fvb * ((v / vb) ** (-beta1 * s) + (v / vb) ** (-beta2 * s)) ** (-1 / s)

        return Fv1

    def do_fit(self,):

        # run emcee:
        sampler = self.run_emcee()

        # plot chains:
        fig, axes = plt.subplots(self.ndim, figsize=(10, 7), sharex=True)
        samples = sampler.get_chain()
        labels = ["Fvb", "vb", "p", "log(f)"]
        for i in range(self.ndim):
            ax = axes[i]
            ax.plot(samples[:, :, i], "k", alpha=0.3)
            ax.set_xlim(0, len(samples))
            ax.set_ylabel(labels[i])
            ax.yaxis.set_label_coords(-0.1, 0.5)

        axes[-1].set_xlabel("step number")

        # plot 2D parameter posterior distributions:
        burnin = self.burnin
        flat_samples = sampler.get_chain(discard=burnin, thin=15, flat=True)
        print(flat_samples.shape)
        fig = corner.corner(flat_samples, labels=labels)
        fig.savefig(f'{self.name}_2dposteriors.pdf')
        print('----------------------------------------------------------')
        print('MCMC results:')

        # get autocorrelation time:
        try:
            tau = sampler.get_autocorr_time()

        except:
            print(
                '**Warning** The chain is shorter than 50 times the integrated autocorrelation time for 4 parameter(s). Use this estimate with caution and run a longer chain!'
            )
            tau = np.inf

        print(
            f'The autocorrelation time is {tau}. You should run the chains for at least 10 x steps as this.'
        )

        # extract p plus uncertainties:
        results = []
        results_up = []
        results_low = []
        print('The MCMC fit parameters are:')
        for i in range(self.ndim):
            mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
            q = np.diff(mcmc)
            txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
            txt = txt.format(mcmc[1], q[0], q[1], labels[i])
            results.append(mcmc[1])
            results_up.append(q[1])
            results_low.append(q[0])
            display(Math(txt))

        print('----------------------------------------------------------')

        # plot the observed and model spectra:

        # plot flux density SED:

        f = plt.figure(figsize=(8, 8))

        vs = np.linspace(0, 30, 100)

        emcee_flux = self.powerlaw(vs, results[0], results[1], results[2])
        emcee_flux_up = self.powerlaw(vs, results_up[0], results_up[1], results_up[2])
        emcee_flux_low = self.powerlaw(
            vs, results_low[0], results_low[1], results_low[2]
        )

        plt.scatter(self.frequency, self.flux_emission)
        plt.errorbar(
            self.frequency,
            self.flux_emission,
            yerr=[self.fd_err_up, self.fd_err_low],
            fmt='.',
            capsize=2,
        )

        plt.plot(vs, emcee_flux)
        plt.plot(vs, emcee_flux + emcee_flux_up, color='grey', alpha=0.5)
        plt.plot(vs, emcee_flux - emcee_flux_low, color='grey', alpha=0.5)
        # plt.plot(frequency[5:], 4*frequency[5:]**-3)

        # plt.axhline(y=np.max(flux_emission),label=r'F_p')

        plt.xscale('log')
        plt.yscale('log')

        # plt.axis([1, 30, 0.5e-2,12e-2])

        plt.xlabel('Frequency (GHz)')
        plt.ylabel('Flux Density (mJy)')

        plt.axvline(
            x=vs[np.where(emcee_flux == np.max(emcee_flux))], ls='--', color='grey'
        )
        plt.axhline(y=np.max(emcee_flux), ls='--', color='grey')

        plt.savefig(f'{self.name}_model_spectrum.pdf')

        # print peak flux, peak frequency, and p of spectrum:
        print('----------------------------------------------------------')
        print('The peak flux, peak frequency, and p of the spectrum are:')
        up = emcee_flux_up[np.where(emcee_flux == np.max(emcee_flux))]
        low = emcee_flux_low[np.where(emcee_flux == np.max(emcee_flux))]
        Ferror = (up + low) / 2
        print(f'Fp = {np.max(emcee_flux):.2f} +/- {Ferror[0]:.2f} mJy')
        print(f'vp = {vs[np.where(emcee_flux == np.max(emcee_flux))][0]:2f} GHz')
        print(f'p = {results[2]:.2f} +{results_up[2]:.2f} - {results_low[2]:.2f} ')
        print('----------------------------------------------------------')
        Fvb = results[0]
        Fvb_u = (results_up[0] + results_low[0]) / 2
        vb = results[1]
        vb_u = (results_up[1] + results_low[1]) / 2
        p = results[2]
        p_u = (results_up[2] + results_low[2]) / 2
        Fp = np.max(emcee_flux)
        Fp_u = Ferror[0]
        vp = vs[np.where(emcee_flux == np.max(emcee_flux))][0]
        return Fvb, vb, p, Fp, vp, Fvb_u, vb_u, p_u, Fp_u

