import numpy as np
from numpy.typing import NDArray
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller
import statsmodels.stats.api as sms
from scipy.stats import ranksums, shapiro, pearsonr, bootstrap
import pywt


def bootstrap_test(df: pd.DataFrame, alpha=0.05):
    """If the confidence interval does not contain zero, we can reject the null hypothesis that there is no difference in mean returns between the two portfolios at 5% level. This implies that LOHT portfolio has a higher mean return than HOLT portfolio."""
    alpha = alpha  # significance level
    B = 1000  # number of bootstrap samples
    seed = 42  # random seed

    # Extract the returns of portfolio LOHT and HOLT as numpy arrays
    x1 = df["LOHT"].to_numpy()
    x2 = df["HOLT"].to_numpy()

    # Define a function to compute the difference in mean returns between two portfolios
    def diff(x1, x2, axis=None):
        return np.mean(x1, axis=axis) - np.mean(x2, axis=axis)

    # Perform the bootstrap test using the bootstrap function of the scipy.stats package
    test_result = bootstrap(
        (x1, x2),
        statistic=diff,
        confidence_level=1 - alpha,
        n_resamples=B,
        random_state=seed,
        method="percentile",
    )

    # Print the test result
    # print(test_result)
    return test_result.confidence_interval


def freq_domain_test(df: pd.DataFrame, alpha=0.05) -> tuple[float, NDArray[np.float64]]:
    # Parameters
    n = len(df)  # number of observations
    w = "cmor1.5-1.0"  # wavelet name
    scales = np.arange(1, n + 1)  # wavelet scales
    alpha = alpha  # significance level
    B = 1000  # number of permutations

    # Function to estimate evolutionary spectra using wavelet transform
    def evo_spec(x, w, scales):
        # Compute the continuous wavelet transform
        cwt, freqs = pywt.cwt(x, scales, w)
        # Compute the power spectrum
        power = np.abs(cwt) ** 2
        # Normalize by frequency
        power /= freqs[:, None]  # type: ignore
        # Return the evolutionary spectrum
        return power

    # Function to compute the correlation between two evolutionary spectra
    def evo_corr(p1, p2):
        # Reshape the spectra into vectors
        v1 = p1.ravel()
        v2 = p2.ravel()
        # Compute the Pearson correlation coefficient
        r, _ = pearsonr(v1, v2)  # type: ignore
        # Return the correlation
        return r

    # Function to perform a randomization test based on correlation
    def rand_test(p1, p2, B, alpha) -> tuple[float, NDArray[np.float64]]:
        # Compute the observed correlation
        r_obs = evo_corr(p1, p2)
        # Initialize an array to store the permuted correlations
        r_perm = np.zeros(B)
        # Loop over the number of permutations
        for b in range(B):
            # Shuffle one of the spectra along time axis
            p2_perm = np.random.permutation(p2.T).T
            # Compute the permuted correlation
            r_perm[b] = evo_corr(p1, p2_perm)
        # Compute the p-value as the proportion of permuted correlations more extreme than the observed one
        p_value = np.mean(np.abs(r_perm) >= np.abs(r_obs))
        # Compare the p-value with the significance level and make a decision
        if p_value < alpha:
            print(f"Reject null hypothesis: p-value = {p_value:.3f}")
        else:
            print(f"Fail to reject null hypothesis: p-value = {p_value:.3f}")
        # Return the p-value and the permuted correlations
        return p_value, r_perm

    # Apply the functions to the two portfolios
    # Extract the returns of portfolio LOHT and HOLT
    x1 = df["LOHT"].to_numpy()
    x2 = df["HOLT"].to_numpy()
    # Estimate the evolutionary spectra for each portfolio
    p1 = evo_spec(x1, w, scales)
    p2 = evo_spec(x2, w, scales)
    # Perform the randomization test based on correlation
    p_value, r_perm = rand_test(p1, p2, B, alpha)
    return p_value, r_perm


def shapiro_test(df: pd.DataFrame) -> tuple[float, float]:
    """Test for normality using the Shapiro-Wilk test

    Returns Shapiro-Wilk test p-values for (LOHT, HOLT). If the p-value
    is less than the significance level (usually 0.05), we conclude
    that it is unlikely that the data came from a normal distribution.
    """
    _, holt_pval = shapiro(df["HOLT"])
    _, loht_pval = shapiro(df["LOHT"])
    return holt_pval, loht_pval


def twosample_ttest(df: pd.DataFrame) -> tuple[float, float]:
    """Performs a two-sample t-test to compare the means of the
    returns of the two portfolios. A two-sample t-test assumes
    that the returns are normally distributed and independent.
    The null hypothesis is that there is no difference between
    the means of the two portfolios, and the alternative hypothesis
    is that there is a difference.

    Returns (t_statistic, p-value). A positive test statistic
    indicates that the first sample has a higher mean than the second
    sample, and vice versa. If **p-value** is less than 0.05, we can
    conclude that there is a statistically significant difference between
    the means of the two portfolios.
    """
    tsr = sms.CompareMeans(sms.DescrStatsW(df["LOHT"]), sms.DescrStatsW(df["HOLT"]))
    t_stat, p_value, _ = tsr.ttest_ind(usevar="unequal")
    return t_stat, float(p_value)


def wilcoxon_rs_test(df: pd.DataFrame) -> tuple[float, float]:
    """Wilcoxon rank-sum test, also known as the Mann-Whitney U test,
    which tests the null hypothesis that two independent samples are
    drawn from the same distribution. The alternative hypothesis is
    that one of the distributions is stochastically greater than the other.

    Return (test statistic, test p-value). If p-value < 0.05 then we can conclude that
    there is evidence that LOHT portfolio has higher returns than
    HOLT portfolio. Otherwise we can't conclude that.
    """
    # Extract returns of portfolio LOHT and HOLT
    returns_LOHT = df["LOHT"]
    returns_HOLT = df["HOLT"]

    # Perform Wilcoxon rank-sum test with alternative='greater'
    test_result = ranksums(returns_LOHT, returns_HOLT, alternative="two-sided")

    # Print test result
    return test_result


def plot_acf_cmp(df: pd.DataFrame) -> None:
    # Check for independence using the autocorrelation function (ACF)
    fig, axs = plt.subplots(2, figsize=(7, 5), sharex=True)
    plot_acf(df["LOHT"], ax=axs[0])
    plot_acf(df["HOLT"], ax=axs[1])
    axs[0].set_title("Autocorrelation Plot for LOHT")
    axs[1].set_title("Autocorrelation Plot for HOLT")
    # Set custom x-axis labels using the dates from the DataFrame index
    axs[1].set_xlabel("Year")
    axs[1].set_xticks(range(len(df.index)))
    axs[1].set_xticklabels(df.index)

    # Rotate the x-axis tick labels for better readability
    for ax in axs:
        plt.sca(ax)
        plt.xticks(rotation=45)
    plt.show()


def ljungbox_test(df: pd.DataFrame, poutput=False) -> tuple[float, float]:
    """Test for the presence of autocorrelation.

    Returns Ljung-Box test p-values for (LOHT, HOLT). If p-values < 0.5, then there is autocorrelation
    """
    loht_lb = acorr_ljungbox(df["LOHT"], lags=10)
    holt_lb = acorr_ljungbox(df["HOLT"], lags=10)
    loht_lb_pvalues = min(loht_lb.lb_pvalue.values)
    holt_lb_pvalues = min(holt_lb.lb_pvalue.values)

    if poutput:
        print(f"Ljung-Box test p-value for LOHT: {round(loht_lb_pvalues, 2)}")
        print(f"Ljung-Box test p-value for HOLT: {round(holt_lb_pvalues, 2)}")
    return loht_lb_pvalues, holt_lb_pvalues


def adf_test(df: pd.DataFrame, poutput=False) -> tuple[float, float]:
    """Check for identical distribution using the augmented Dickey-Fuller (ADF) test.

    Returns ADF test p-values for (LOHT, HOLT). If the ADF test p-values are less
    than 0.05, it suggests that the returns for each portfolio are stationary and
    identically distributed."""
    df = df.diff().dropna()
    adf_loht = float(adfuller(df["LOHT"])[1])
    adf_holt = float(adfuller(df["HOLT"])[1])

    if poutput:
        # print(f'ADF test statistic for LOHT: {adf_loht[0]}')
        print(f"ADF test p-value for LOHT: {round(adf_loht, 4)}")
        # print(f'ADF test statistic for HOLT: {adf_holt[0]}')
        print(f"ADF test p-value for HOLT: {round(adf_holt, 4)}")

    return adf_loht, adf_holt
