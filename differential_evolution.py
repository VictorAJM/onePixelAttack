from __future__ import division, print_function, absolute_import
from xml.dom.minicompat import StringTypes
import numpy as np
from scipy.optimize import OptimizeResult, minimize
# from scipy.optimize.optimize import _status_message
from scipy._lib._util import check_random_state
import warnings


__all__ = ['differential_evolution']

_MACHEPS = np.finfo(np.float64).eps


def differential_evolution(func, bounds, args=(), strategy='best1bin',
                           maxiter=1000, popsize=15, tol=0.01,
                           mutation=(0.5, 1), recombination=0.7, seed=None,
                           callback=None, disp=False, polish=True,
                           init='latinhypercube', atol=0):

    solver = DifferentialEvolutionSolver(func, bounds, args=args,
                                         strategy=strategy, maxiter=maxiter,
                                         popsize=popsize, tol=tol,
                                         mutation=mutation,
                                         recombination=recombination,
                                         seed=seed, polish=polish,
                                         callback=callback,
                                         disp=disp, init=init, atol=atol)
    return solver.solve()


class DifferentialEvolutionSolver(object):

    # Dispatch of mutation strategy method (binomial or exponential).

    def __init__(self, func, bounds, args=(),
                strategy='best1bin', maxiter=1000, popsize=15,
                tol=0.01, mutation=(0.5, 1), recombination=0.7, seed=None,
                maxfun=np.inf, callback=None, disp=False, polish=True,
                init='latinhypercube', atol=0):

        self.mutation_func = getattr(self, '_best1')
        self.strategy = strategy
        self.callback = callback
        self.polish = polish
        self.tol, self.atol = tol, atol
        self.scale = mutation
        self.dither = None
        self.dither = [mutation[0], mutation[1]]
        self.cross_over_probability = recombination
        self.func = func
        self.args = args
        self.limits = np.array(bounds, dtype='float').T
        self.maxiter = maxiter
        self.maxfun = maxfun
        self.__scale_arg1 = 0.5 * (self.limits[0] + self.limits[1])
        self.__scale_arg2 = np.fabs(self.limits[0] - self.limits[1])
        self.parameter_count = np.size(self.limits, 1)
        self.random_number_generator = check_random_state(seed)
        self.num_population_members = max(5, popsize * self.parameter_count)
        self.population_shape = (self.num_population_members,
                                self.parameter_count)
        self._nfev = 0
        self.init_population_random()
        self.disp = disp

    def init_population_random(self):
        rng = self.random_number_generator
        self.population = rng.random_sample(self.population_shape)
        self.population_energies = (np.ones(self.num_population_members) *
                                    np.inf)
        self._nfev = 0

    @property
    def x(self):
        return self._scale_parameters(self.population[0])

    @property
    def convergence(self):
        return (np.std(self.population_energies) /
                np.abs(np.mean(self.population_energies) + _MACHEPS))

    def solve(self):
        nit, warning_flag = 0, False

        if np.all(np.isinf(self.population_energies)):
            self._calculate_population_energies()

        for nit in range(1, self.maxiter + 1):

            try:
                next(self)
            except StopIteration:
                warning_flag = True
                break
            convergence = self.convergence
            if (self.callback and
                    self.callback(self._scale_parameters(self.population[0]),
                                  convergence=self.tol / convergence) is True):
                warning_flag = True
                break
            intol = (np.std(self.population_energies) <=
                    self.atol +
                    self.tol * np.abs(np.mean(self.population_energies)))
            if warning_flag or intol:
                break
        else:
            warning_flag = True
        DE_result = OptimizeResult(
            x=self.x,
            fun=self.population_energies[0],
            nfev=self._nfev,
            nit=nit,
            message="status_message",
            success=(warning_flag is not True))

        if self.polish:
            result = minimize(self.func,
                              np.copy(DE_result.x),
                              method='L-BFGS-B',
                              bounds=self.limits.T,
                              args=self.args)
            self._nfev += result.nfev
            DE_result.nfev = self._nfev
            if result.fun < DE_result.fun:
                DE_result.fun = result.fun
                DE_result.x = result.x
                DE_result.jac = result.jac
                self.population_energies[0] = result.fun
                self.population[0] = self._unscale_parameters(result.x)
        return DE_result

    def _calculate_population_energies(self):
        itersize = max(0, min(len(self.population), self.maxfun - self._nfev + 1))
        candidates = self.population[:itersize]
        parameters = np.array([self._scale_parameters(c) for c in candidates]) 
        energies = self.func(parameters, *self.args)
        self.population_energies = energies
        self._nfev += itersize
        minval = np.argmin(self.population_energies)
        lowest_energy = self.population_energies[minval]
        self.population_energies[minval] = self.population_energies[0]
        self.population_energies[0] = lowest_energy

        self.population[[0, minval], :] = self.population[[minval, 0], :]

    def __iter__(self):
        return self

    def __next__(self):
        if np.all(np.isinf(self.population_energies)):
            self._calculate_population_energies()

        if self.dither is not None:
            self.scale = (self.random_number_generator.rand()
                          * (self.dither[1] - self.dither[0]) + self.dither[0])

        itersize = max(0, min(self.num_population_members, self.maxfun - self._nfev + 1))
        trials = np.array([self._mutate(c) for c in range(itersize)])
        for trial in trials: self._ensure_constraint(trial)
        parameters = np.array([self._scale_parameters(trial) for trial in trials])
        energies = self.func(parameters, *self.args)
        self._nfev += itersize

        for candidate,(energy,trial) in enumerate(zip(energies, trials)):
            if energy < self.population_energies[candidate]:
                self.population[candidate] = trial
                self.population_energies[candidate] = energy

                if energy < self.population_energies[0]:
                    self.population_energies[0] = energy
                    self.population[0] = trial
        return self.x, self.population_energies[0]

    def next(self):
        return self.__next__()

    def _scale_parameters(self, trial):
        return self.__scale_arg1 + (trial - 0.5) * self.__scale_arg2

    def _unscale_parameters(self, parameters):
        return (parameters - self.__scale_arg1) / self.__scale_arg2 + 0.5

    def _ensure_constraint(self, trial):
        for index in np.where((trial < 0) | (trial > 1))[0]:
            trial[index] = self.random_number_generator.rand()

    def _mutate(self, candidate):
        trial = np.copy(self.population[candidate])
        rng = self.random_number_generator
        fill_point = rng.randint(0, self.parameter_count)
        bprime = self.mutation_func(self._select_samples(candidate, 5))
        crossovers = rng.rand(self.parameter_count)
        crossovers = crossovers < self.cross_over_probability
        crossovers[fill_point] = True
        trial = np.where(crossovers, bprime, trial)
        return trial

    def _best1(self, samples):
        r0, r1 = samples[:2]
        return (self.population[0] + self.scale *
                (self.population[r0] - self.population[r1]))

    def _rand1(self, samples):
        r0, r1, r2 = samples[:3]
        return (self.population[r0] + self.scale *
                (self.population[r1] - self.population[r2]))

    def _select_samples(self, candidate, number_samples):
        idxs = list(range(self.num_population_members))
        idxs.remove(candidate)
        self.random_number_generator.shuffle(idxs)
        idxs = idxs[:number_samples]
        return idxs
