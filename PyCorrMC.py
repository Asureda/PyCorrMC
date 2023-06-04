import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

class PyCorrMC:
    def function1(self, x):
        # Función 1 para jacknife
        return x

    def function2(self, x, y):
        # Función 2 para jacknife
        return x-y**2

    def jack_knife_1d(x, function):
        n = len(x)
        x_total = np.sum(x)
        jk_aver_x = (x_total - x) / np.float64(n - 1)
        jk_func = function(jk_aver_x)
        jk_func_aver = np.mean(jk_func)
        err = np.sqrt((n - 1) * np.mean((jk_func - jk_func_aver) ** 2))
        return jk_func_aver, err
    
    def jack_knife_2d(self,x,y,function):
        x_total=np.sum(x)
        y_total=np.sum(y)
        jk_aver_x=[]
        jk_aver_y=[]
        for i,j in zip(x,y):
            jk_aver_x.append((x_total-i)/np.float64(len(x)-1))
            jk_aver_y.append((y_total-j)/np.float64(len(y)-1))
        
        jk_func=[]
        jk_func=np.array(list(map(lambda x,y : function(x,y),jk_aver_x,jk_aver_y )))
        jk_func_aver=np.sum(jk_func)/np.float64(len(x))
        
        err=np.sqrt((np.float64(len(x)-1))*np.sum((jk_func-jk_func_aver)**2.)/np.float64(len(x)))
        return jk_func_aver,err

    def fitting(self, x, a, b, c):
        # Función de ajuste
        return -np.exp(-a * x) * b + c
    
    def binning_analysis(self, time_series, magnitude, max_size):
        # Calcula x como el logaritmo base 2 de la longitud de la serie de tiempo
        x = int(np.log2(len(time_series)))
        
        # Calcula n como 2^x si 2^x es menor o igual a max_size, de lo contrario n toma el valor de max_size
        n = 2 ** x if 2 ** x <= max_size else max_size
        
        # Genera una secuencia de tamaños de bins en múltiplos de 2, comenzando desde 2^0 hasta 2^(log2(n)-1)
        sizes = 2 ** np.arange(0, int(np.log2(n)), dtype=int)
        
        # Crea un array de ceros de tamaño igual a la cantidad de tamaños de bins
        sems = np.zeros(len(sizes), dtype=float)
        
        # Obtiene la cantidad total de muestras en la serie de tiempo
        n_samples = len(time_series)
        
        # Itera sobre los diferentes tamaños de bins
        for s in range(len(sizes)):
            # Calcula la cantidad de bins para el tamaño actual
            n_bins = n_samples // sizes[s]
            
            # Obtiene los promedios de los bins utilizando reshape para agrupar los datos
            bin_avgs = np.mean(time_series[:n_bins * sizes[s]].reshape((n_bins, -1)), axis=1)
            
            # Calcula el error estándar de los promedios de los bins
            sems[s] = np.std(bin_avgs, ddof=1.5) / np.sqrt(n_bins)
        
        # Obtiene los tamaños de bins correspondientes hasta max_size - 2
        sizes_subset = sizes[:max_size - 2]
        
        # Define la función fit_fn que se utilizará en el ajuste de curva
        def fit_fn(x, a, b, c):
            return -np.exp(-a * x) * b + c

        # Realiza el ajuste de curva utilizando curve_fit y obtiene los parámetros ajustados
        fit_params, _ = curve_fit(fit_fn, sizes_subset, sems[:max_size], (0.05, 1, 0.05))
        
        # Calcula los valores ajustados para todos los tamaños de bins utilizando los parámetros ajustados
        fit_sems = fit_fn(sizes, *fit_params)

        tau_int = 0.5*(sems[-1]/sems[0])        
        print(f"Final Standard Error of the Mean: {fit_params[2]:.8f}")
        print(f"Integrated autocorrelation time: {tau_int:.8f}")
        print(f"Exp autocorrelation time: {fit_params[1]:.8f}")
        N_eff = n_samples / (2 * tau_int)
        print(f"Original number of samples: {n_samples}")
        print(f"Effective number of samples: {N_eff:.1f}")
        print(f"Ratio: {N_eff / n_samples:.3f}\n")

        plt.figure(figsize=(10, 6))
        plt.plot(sizes, sems,'x',color='black',markersize=5,label='data')
        plt.plot(sizes, fit_sems,'-.',color='black',linewidth=0.5, label='fit')
        plt.title('Binning results')
        plt.xlabel('m')
        plt.ylabel('$\sigma_m$')
        plt.legend()
        plt.xscale("log")
        plt.show()


    def autocorrelation_analysis(self, data, C, window, N_MAX=1000):
        # initial processing
        data_size = len(data)
        avg = np.average(data)
        data_centered = data - avg

        # auto-covariance function
        autocov = np.empty(window)
        for j in range(window):
            autocov[j] = np.dot(data_centered[:data_size - j], data_centered[j:])
        autocov /= data_size

        # autocorrelation function
        acf = autocov / autocov[0]

        # integrate autocorrelation function
        j_max_v = np.arange(window)
        tau_int_v = np.zeros(window)
        for j_max in j_max_v:
            tau_int_v[j_max] = 0.5 + np.sum(acf[1:j_max + 1])

        # find j_max
        j_max = 0
        while j_max < C * tau_int_v[j_max]:
            j_max += 1

        # wrap it up
        tau_int = tau_int_v[j_max]
        N_eff = data_size / (2 * tau_int)
        sem = np.sqrt(autocov[0] / N_eff)

        # create ACF plot
        fig = plt.figure(figsize=(10, 6))
        plt.gca().axhline(0, color="gray",linewidth=1)
        plt.plot(acf)
        plt.xlabel("lag time $j$")
        plt.ylabel("$\hat{K}^{XX}_j$")
        plt.show()

        def exp_fnc(x, a, b):
            return a * np.exp(-x / b)
        
        j = np.arange(1, N_MAX)
        j_log = np.logspace(0, 3, 100)
        popt, pcov = curve_fit(exp_fnc, j, autocov[1:N_MAX], p0=[15, 10])

        fig = plt.figure(figsize=(10, 6))
        plt.plot(j, autocov[1:N_MAX], "x", label="numerical ACF")
        plt.plot(j, exp_fnc(j, popt[0], popt[1]), label="exponential fit")
        plt.xlim((1, N_MAX))
        plt.xscale("log")
        plt.xlabel("lag time $j$")
        plt.ylabel("$\hat{K}^{XX}_j$")
        plt.legend()
        plt.show()
        print(f"Exponential autocorrelation time: {popt[1]:.2f} sampling intervals")
        # create integrated ACF plot
        fig = plt.figure(figsize=(10, 6))
        plt.plot(j_max_v[1:], C * tau_int_v[1:])
        plt.ylim(plt.gca().get_ylim()) # explicitly keep the limits of the first plot
        plt.plot(j_max_v[1:], j_max_v[1:])
        plt.plot([j_max], [C * tau_int_v[j_max]], "ro")
        plt.xscale("log")
        plt.xlabel(r"sum length $j_\mathrm{max}$")
        plt.ylabel(r"$C \times \hat{\tau}_{X, \mathrm{int}}$")
        plt.title("")
        plt.show()

        # print out stuff
        print(f"Mean value: {avg:.4f}")
        print(f"Standard error of the mean: {sem:.4f}")
        print(f"Integrated autocorrelation time: {tau_int:.2f} time steps")
        print(f"Effective number of samples: {N_eff:.1f}")

        return sem
        
    def block_analyze(self,input_data, n_blocks=16):
        data = np.asarray(input_data)
        block = 0
        # this number of blocks is recommended by Janke as a reasonable compromise
        # between the conflicting requirements on block size and number of blocks
        block_size = int(data.shape[1] // n_blocks)
        print(f"block_size: {block_size}")
        # initialize the array of per-block averages
        block_average = np.zeros((n_blocks, data.shape[0]))
        # calculate averages per each block
        for block in range(n_blocks):
            block_average[block] = np.average(data[:, block * block_size: (block + 1) * block_size], axis=1)
        # calculate the average and average of the square
        av_data = np.average(data, axis=1)
        av2_data = np.average(data * data, axis=1)
        # calculate the variance of the block averages
        block_var = np.var(block_average, axis=0)
        # calculate standard error of the mean
        err_data = np.sqrt(block_var / (n_blocks - 1))
        # estimate autocorrelation time using the formula given by Janke
        # this assumes that the errors have been correctly estimated
        tau_data = np.zeros(av_data.shape)
        for val in range(av_data.shape[0]):
            if av_data[val] == 0:
                # unphysical value marks a failure to compute tau
                tau_data[val] = -1.0
            else:
                tau_data[val] = 0.5 * block_size * n_blocks / (n_blocks - 1) * block_var[val] \
                    / (av2_data[val] - av_data[val] * av_data[val])
        return av_data, err_data, tau_data, block_size
