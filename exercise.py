# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Analysis of Howell's data with pymc

# +
import numpy as np
import matplotlib.pyplot as plt # type: ignore

import pandas as pd             # type: ignore
# -

# Partial census data for !Kung San people (Africa), collected by Nancy Howell (~ 1960), csv from R. McElreath, "Statistical Rethinking", 2020.

# +
howell: pd.DataFrame

try:
    howell = pd.read_csv('https://raw.githubusercontent.com/rmcelreath/rethinking/master/data/Howell1.csv', sep=';', dtype={'male': bool})
except:
    howell = pd.read_csv('Howell1.csv', sep=';', dtype={'male': bool})
# -

# ## A normal model for the height
#
# We want to analyse the hypothesis that the height of adult people is normally distributed, therefore we design this statistical model ($h$ is the height), with an *a priori* normal distribution of the mean, and an *a priori* uniform distribution of the standard deviation.
#
#
# $ h \sim N(\mu, \sigma)$
#
# $ \mu \sim N(170, 20) $
#
# $ \sigma \sim U(0, 50) $
#
#

import pymc as pm   # type: ignore
import arviz as az  # type: ignore

# +
norm_height = pm.Model()

with norm_height:
    mu = pm.Normal('mu_h', 170, 20)
    sigma = pm.Uniform('sigma_h', 0, 50)
    h = pm.Normal('height', mu, sigma)
# -

# ### Exercise 1
#
# The model can be used to draw random samples. In other words, if you assume the variables `mu`, `sigma`, and `h` are distributed as stated in your statistical model, you can generate synthetic (fake) data which comply with your *a priori* (i.e., before having seen any data) hypotheses.
#
# Plot the *a priori* densities of the three random variables of the model. You can sample random values with the function `pm.draw`. For example `pm.draw(mu, draws=1000)` samples 1000 values from the *a priori* distribution of `mu`.
#

fig, ax = plt.subplots(ncols=3, figsize=(15,5))
ax[0].hist(pm.draw(sigma, draws=10000), bins='auto', density=True)
ax[0].set_title(str(sigma))
ax[1].hist(pm.draw(mu, draws=10000), bins='auto', density=True)
ax[1].set_title(str(mu))
ax[2].hist(pm.draw(h, draws=10000), bins='auto', density=True)
_ = ax[2].set_title(str(h))

# ### Exercise 2
#
# Consider only adult ($\geq 18$) males. Redefine the model above, with the same *a priori* assumptions, but making the height `h` an **observed** variable, using Howell's data about adult males as observations.

adult_males = howell.query('male & age >= 18')

# +
norm_height_am = pm.Model()

with norm_height_am:
    mu = pm.Normal('mu_h', 170, 20)
    sigma = pm.Uniform('sigma_h', 0, 50)
    h = pm.Normal('height', mu, sigma, observed=adult_males['height'])
# -

# ### Exercise 3
#
# Sample values from the posterior, by using `pm.sample()`. Remember to execute this within the context of the model, by using a `with` statement. By default, `pm.sample()` returns an `InferenceData` object which packages all the data about the sampling. One can summarize the *posterior* values with `az.summary`. To play further with the *posterior* distributions is useful to use `az.extract` to get an object that can be mostly used as a pandas `DataFrame` (but in fact is another type: `xarray.Dataset`).

with norm_height_am:
    idata = pm.sample(chains=4, progressbar=False)

az.summary(idata)

post = az.extract(idata, combined=True).to_pandas()

# ### Exercise 4
#
# Plot together the density of the posterior `mu_h` and the density of the prior `mu_h`.
#

sim_mu = pm.draw(mu, draws=4000)

fig, ax = plt.subplots()
ax.hist(post['mu_h'], bins='auto', density=True, label='Posterior mu_h')
ax.hist(sim_mu, bins='auto', density=True, label='Prior mu_h')
ax.set_xlim((150, 170))
_ = fig.legend()

# ### Exercise 5
#
# Plot the posterior densities by using `az.plot_posterior`.

with norm_height_am:
    pm.plot_posterior(idata)


# ### Exercise 6
#
# The sampling produced 4000 different values for `mu_h` and 4000 different values for `sigma_h`.
# Plot together all the posterior *height* densities, by using all the sampled values for `mu_h` and `sigma_h` (Use the `gaussian` function below. You will get many lines, $4000\times4000$! Use a gray color and a linewidth of 0.1 and possibly use one sample every 100 to reduce computing time). Add to the plot (in red) the posterior height density computed by using the mean for the posterior `mu` and `sigma`. Add to the plot (in dashed blue) the prior height density computed by using the mean for the prior `mu` and `sigma` (used the values computed by solving the previous exercise).
#

def gaussian(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    return (1/(2*np.pi*sigma**2)**.5)*np.exp(-(x - mu)**2/(2*sigma**2))


prior_mu = pm.draw(mu, draws=10000).mean()
prior_sigma = pm.draw(sigma, draws=10000).mean()

fig, ax = plt.subplots()
x = np.linspace(100, 200, 1000)
for m in range(0, len(post['mu_h']), 100):
    for s in range(0, len(post['sigma_h']), 100):
        ax.plot(x, gaussian(x, post['mu_h'].iloc[m],
                            post['sigma_h'].iloc[s]),
                color='gray', linewidth=.1)
ax.plot(x, gaussian(x, post['mu_h'].mean(),
                    post['sigma_h'].mean()), color='red', 
       label='Posterior mean')
ax.plot(x, gaussian(x, prior_mu,
                       prior_sigma), color='blue', 
        linestyle='dashed', label='Prior')
ax.set_title('Posterior height distribution')
_ = ax.legend()


# ## A linear regression model
#
# We want to analyze the relationship between height and weight in adult males. We consider the following model, where $h$ is the height, $w$ is the weight, $\bar w$ is the mean weight.
#
# $ h \sim N(\mu, \sigma)$
#
# $ \mu = \alpha + \beta*(w - \bar w) $
#
# $ \alpha = N(178, 20) $
#
# $ \beta = N(0, 10) $
#
# $ \sigma \sim U(0, 50) $
#

# ### Exercise 8
#
# Define the model `linear_regression` as a `pm.Model()`. Use Howell's data as observations.

# +
linear_regression = pm.Model()

with linear_regression:
    sigma = pm.Uniform('sigma_h', 0, 50)
    alpha = pm.Normal('alpha', 178, 20)
    beta = pm.Normal('beta', 0, 10)
    mu = alpha + beta*(adult_males['weight'] - adult_males['weight'].mean())
    h = pm.Normal('height', mu, sigma, observed=adult_males['height'])
# -

# ### Exercise 9
#
# Sample the model and plot the posterior densities.
#

# +
with linear_regression:
    idata_regression = pm.sample(chains=4, progressbar=False)

r_post = az.extract(idata_regression, combined=True).to_pandas()
# -

with linear_regression:
    pm.plot_posterior(idata_regression)

# ### Exercise 10
#
# Plot a scatter plot of heights and the deviations of the weights from the mean. Add to the plot the regression line  using as the parameters the mean of the sampled posterior values.
#

# +
d_weight = adult_males['weight'] - adult_males['weight'].mean()

x = np.linspace(d_weight.min(), d_weight.max(), 100)


fig, ax = plt.subplots()
ax.scatter(d_weight, adult_males['height'])
ax.set_ylabel('height (cm)')
ax.set_xlabel('deviations of the weights from the mean (kg)')
_ = ax.plot(x, r_post['alpha'].mean() +
            r_post['beta'].mean()*x,
            color='red')
