"""
Compute SNR and time-frequency properties of gravitational wave signals.

This module provides functionality to calculate the optimal signal-to-noise
ratio (SNR) and various time-frequency measures of gravitational wave waveforms
in specified noise backgrounds. It includes both noise-dependent and
noise-independent signal characterization measures.

Functions
---------
xoptimalsnr : function
    Compute SNR and time-frequency measures of a waveform in noise background

MATLAB Documentation
--------------------
XOPTIMALSNR - Compute SNR and other properties of a signal in noise.

XOPTIMALSNR - Compute the SNR and time-frequency measures of a waveform
in a specified noise background.  Simpler noise-independent measures of
the wave amplitude are also provided.

  [SNR, h_rss, h_peak, Fchar, bw, Tchar, dur, Tchar_w, dur_w, tfbox, hw] = ...
      xoptimalsnr(h,t0,fs,S,F0,dF,Fmin,Fmax,tfFrac)

  h      Array.  Waveform timeseries data.  Each column holds the
         timeseries for one of the GW polarizations.  (There may be any
         number of polarisations, for the non-GR case.  The order of
         polarizations does not matter.)  The timeseries duration should
         be a power of 2.
  t0     Scalar.  Time at which the first waveform data point h(1,:) is
         sampled.
  fs     Scalar.  Sampling rate (Hz) of waveform data.  This should be a
         power of 2.
  S      Vector (optional).  Noise background one-sided POWER (not
         amplitude) spectrum.  If supplied, it must cover at least the
         range [Fmin, Fmax], and must be sampled in ascending
         order of frequency.  For noise-independent signal measures use
         S=f0=df=[].  In this case the constant spectrum S(f)=2 will be
         used (corresponding to a two-sided noise spectrum of unity).
  F0     Scalar or vector (optional).  Scalar: Frequency at which S(1) is
         sampled.  Vector: frequencies at which S is sampled.
  dF     Scalar (optional).  Frequency spacing of the noise spectrum S.
         If F0 is a vector then dF = [] must be used.
  Fmin   Scalar (optional).  Minimum frequency (Hz) to include in
         frequency-domain calculations.  Defaults to 0 if [] or no noise
         spectrum specified, or minimum frequency of noise spectrum.
  Fmax   Scalar (optional).  Maximum frequency (Hz) to include in
         frequency-domain calculations. Defaults to Nyquist if [] or no noise
         spectrum specified, or maximum frequency of noise spectrum.
  tfFrac Optional scalar in interval (0,1). Fraction of noise-weighted signal
         energy (SNR^2), measured separately in time and frequency domains, to
         be used in computing tfbox output.

The SNR and h_rss measures are restricted to the frequency interval
[Fmin, Fmax], but the time-domain measures use the entire timeseries.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(filename="xpypeline.log", encoding="utf-8", level=logging.DEBUG)

def xoptimalsnr(
    h: np.ndarray,
    t0: float,
    fs: float,
    S: np.ndarray | None = None,
    F0: float | np.ndarray | None = None,
    dF: float | None = None,
    Fmin: float | None = None,
    Fmax: float | None = None,
    tfFrac: float = 0.99,
) -> tuple[
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    list[float] | None,
    np.ndarray,
]:
    """
    Compute optimal SNR and time-frequency properties of gravitational wave signals.

    This function calculates the optimal signal-to-noise ratio and various
    time-frequency measures of a gravitational wave waveform in a specified
    noise background. It provides both noise-dependent and noise-independent
    signal characterization measures.

    Parameters
    ----------
    h : ndarray
        Waveform timeseries data with shape (n_samples, n_polarizations).
        Each column contains the timeseries for one GW polarization. Any
        number of polarizations is supported for non-GR cases. The order
        of polarizations does not matter. Timeseries duration should
        preferably be a power of 2 for optimal FFT performance.
    t0 : float
        Time (in seconds) at which the first waveform data point h[0,:]
        is sampled. This sets the time reference for the waveform.
    fs : float
        Sampling rate in Hz of the waveform data. Should preferably be
        a power of 2 for optimal FFT performance.
    S : ndarray, optional
        One-sided noise power spectral density (PSD) background, not amplitude.
        Must cover at least the frequency range [Fmin, Fmax] and be sampled
        in ascending order of frequency. If None (default), noise-independent
        signal measures are computed using a flat spectrum S(f)=2.
    F0 : float or ndarray, optional
        Frequency specification for the noise PSD. If scalar: frequency at
        which S[0] is sampled. If array: frequencies at which S is sampled.
        Required if S is provided.
    dF : float, optional
        Frequency spacing of the noise spectrum S when F0 is scalar.
        Must be None if F0 is an array. Required if S and F0 are provided
        and F0 is scalar.
    Fmin : float, optional
        Minimum frequency in Hz to include in frequency-domain calculations.
        Defaults to 0 if no noise spectrum is specified, or to the minimum
        frequency of the noise spectrum.
    Fmax : float, optional
        Maximum frequency in Hz to include in frequency-domain calculations.
        Defaults to Nyquist frequency if no noise spectrum is specified,
        or to the maximum frequency of the noise spectrum.
    tfFrac : float, optional
        Fraction of noise-weighted signal energy (between 0 and 1) to be
        contained in the time-frequency bounding box. Default is 0.99 (99%).

    Returns
    -------
    SNR : float
        Optimal signal-to-noise ratio in the given noise background,
        defined as SNR = sqrt(2 * ∫[Fmin to Fmax] df * p_FD / S)
        where p_FD is the frequency-domain power distribution.
    h_rss : float
        Root-sum-square amplitude of the waveform in Hz^(-1/2),
        defined as h_rss = sqrt(∫[Fmin to Fmax] df * p_FD).
    h_peak : float
        Maximum instantaneous RMS amplitude over all polarizations,
        defined as h_peak = sqrt(max(p_TD)) where p_TD is the
        time-domain power distribution.
    Fchar : float
        Characteristic frequency in Hz of the signal in the given noise,
        defined as the noise-weighted centroid of the frequency distribution:
        Fchar = ∫(f * p_FD / S) df / ∫(p_FD / S) df
    bw : float
        Effective bandwidth in Hz of the signal in the given noise,
        defined as the noise-weighted RMS width around Fchar:
        bw = sqrt(∫((f - Fchar)² * p_FD / S) df / ∫(p_FD / S) df)
    Tchar : float
        Characteristic time in seconds WITHOUT noise weighting,
        defined as the centroid of the time-domain power distribution:
        Tchar = ∫(t * p_TD) dt / ∫p_TD dt
    dur : float
        Duration in seconds WITHOUT noise weighting,
        defined as the RMS width around Tchar:
        dur = sqrt(∫((t - Tchar)² * p_TD) dt / ∫p_TD dt)
    Tchar_w : float
        Characteristic time in seconds WITH noise weighting,
        computed using the noise-weighted waveform h_weighted.
    dur_w : float
        Duration in seconds WITH noise weighting,
        computed using the noise-weighted waveform h_weighted.
    tfbox : list of float or None
        Time-frequency bounding box containing tfFrac fraction of the
        noise-weighted signal energy. Returns None if tfFrac <= 0.
        Contains [t_start, f_start, duration, bandwidth] where:
        - t_start: start time in seconds
        - f_start: start frequency in Hz
        - duration: time duration in seconds
        - bandwidth: frequency bandwidth in Hz
    h_weighted : ndarray
        Noise-weighted time-domain waveform with same shape as input h.
        Defined as h_weighted = IFFT(FFT(h) / sqrt(S)).
        If no noise spectrum is provided, this equals h up to normalization.

    Raises
    ------
    ValueError
        If Fmin >= Fmax or if input parameters are inconsistent.

    Notes
    -----
    The computations use energy distributions defined as:
    - p_TD = |h[:,0]|² + |h[:,1]|² + ... (time domain)
    - p_FD = 2 * (|FFT(h[:,0])|² + |FFT(h[:,1])|² + ...) for f >= 0

    No windowing is applied to FFTs since windowing is not appropriate
    for transient signals where statistical properties change over time.

    The noise power spectrum S is interpolated to match the frequencies
    of the FFT grid when necessary.

    SNR and h_rss measures are restricted to [Fmin, Fmax], but time-domain
    measures use the entire timeseries.

    If the timeseries length is odd, the last sample is dropped to ensure
    even length for proper FFT handling.

    Examples
    --------
    Compute SNR for a chirp waveform in Advanced LIGO noise:

    >>> import numpy as np
    >>> # Generate a simple chirp
    >>> fs = 4096
    >>> duration = 1.0
    >>> t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    >>> f0, f1 = 35, 300  # Start and end frequencies
    >>> chirp_rate = (f1 - f0) / duration
    >>> phase = 2 * np.pi * (f0 * t + 0.5 * chirp_rate * t**2)
    >>> h_plus = 1e-21 * np.sin(phase)
    >>> h_cross = 1e-21 * np.cos(phase)
    >>> h = np.column_stack([h_plus, h_cross])
    >>>
    >>> # Use flat noise for simplicity
    >>> snr, h_rss, h_peak, fchar, bw, tchar, dur, tchar_w, dur_w, tfbox, hw = (
    ...     xoptimalsnr(h, 0.0, fs, Fmin=20, Fmax=2000)
    ... )
    >>> print(f"SNR: {snr:.2f}, Characteristic frequency: {fchar:.1f} Hz")

    Analyze a sine-Gaussian burst:

    >>> # Generate sine-Gaussian
    >>> t = np.linspace(-0.1, 0.1, 1024, endpoint=False)
    >>> f0 = 100  # Central frequency
    >>> tau = 0.01  # Time constant
    >>> sg = np.exp(-t**2 / (2*tau**2)) * np.sin(2*np.pi*f0*t)
    >>> h_sg = np.column_stack([sg, np.zeros_like(sg)])
    >>>
    >>> # Analyze without noise weighting
    >>> results = xoptimalsnr(h_sg, t[0], 1/(t[1]-t[0]))
    >>> snr, h_rss, h_peak, fchar, bw, tchar, dur = results[:7]
    >>> print(f"Peak amplitude: {h_peak:.2e}")
    >>> print(f"Central frequency: {fchar:.1f} Hz")
    >>> print(f"Bandwidth: {bw:.1f} Hz")
    """

    # ---- Validate input arguments
    if h.shape[0] % 2 != 0:
        h = h[:-1, :]  # Drop the last sample if odd number of points
    N = h.shape[0]  # Number of data points
    T = N / fs  # Duration of time series

    if not np.log2(T).is_integer():
        logger.warning("Time series duration is not a power of 2.")

    t = t0 + np.arange(N) / fs

    # If no noise spectrum is specified, create a dummy one
    noise_independent = (
        S is None
        or F0 is None
        or (isinstance(S, np.ndarray) and S.size == 0)
        or (isinstance(F0, np.ndarray) and F0.size == 0)
    )
    if noise_independent:
        dF = 1 / T
        F0 = 0
        S = 2 * np.ones(N // 2 + 1)
        F = F0 + np.arange(len(S)) * dF
    elif isinstance(F0, (float, int)) or (isinstance(F0, np.ndarray) and F0.ndim == 0):
        F = F0 + np.arange(len(S)) * dF
    else:
        F = F0

    # Define Fmin and Fmax
    if Fmax is None:
        Fmax = F[-1] if not noise_independent else fs / 2
    if Fmin is None:
        Fmin = F[0] if not noise_independent else 0

    if Fmin >= Fmax:
        raise ValueError("Frequency limits must satisfy Fmin < Fmax.")

    # ---- Time-domain properties (NO noise weighting)
    p_TD = np.sum(h**2, axis=1)  # Energy distribution in time
    h_peak = np.sqrt(np.max(p_TD))  # Peak amplitude
    Tchar = np.sum(t * p_TD) / np.sum(p_TD)  # Characteristic time
    dur = np.sqrt(np.sum((t - Tchar) ** 2 * p_TD) / np.sum(p_TD))  # Duration

    # ---- Time-domain properties (WITH noise weighting)
    if noise_independent:
        Tchar_w, dur_w, h_weighted = Tchar, dur, h
    else:
        hf_orig = np.fft.fft(h, axis=0) / fs
        f_orig = np.fft.fftfreq(N, d=1 / fs)
        f_orig = f_orig[: N // 2 + 1]
        one_over_S = np.zeros_like(f_orig)
        idx = np.logical_and(f_orig >= F[0], f_orig <= F[-1])
        one_over_S[idx] = np.interp(f_orig[idx], F, 1 / S)
        # Apply weighting only to positive frequencies, then reconstruct full spectrum
        hf_pos = hf_orig[: N // 2 + 1]
        hf_pos_weighted = hf_pos * np.sqrt(one_over_S)[:, None]
        # Reconstruct full spectrum with Hermitian symmetry
        hf_weighted = np.zeros_like(hf_orig)
        hf_weighted[: N // 2 + 1] = hf_pos_weighted
        if N % 2 == 0:
            # Even length - exclude Nyquist frequency from conjugate
            hf_weighted[N // 2 + 1 :] = np.conj(hf_pos_weighted[-2:0:-1])
        else:
            # Odd length
            hf_weighted[N // 2 + 1 :] = np.conj(hf_pos_weighted[-1:0:-1])
        h_weighted = np.fft.ifft(hf_weighted, axis=0).real
        p_TD_weighted = np.sum(h_weighted**2, axis=1)
        Tchar_w = np.sum(t * p_TD_weighted) / np.sum(p_TD_weighted)
        dur_w = np.sqrt(
            np.sum((t - Tchar_w) ** 2 * p_TD_weighted) / np.sum(p_TD_weighted),
        )

    # ---- Frequency-domain calculations
    hf = np.fft.fft(h, axis=0) / fs
    f = np.fft.fftfreq(N, d=1 / fs)
    idx = np.logical_and(f >= Fmin, f <= Fmax)
    f = f[idx]
    hf = hf[idx, :]
    p_FD = 2 * np.sum(np.abs(hf) ** 2, axis=1)

    S = np.interp(f, F, S)
    SNR2f = 2 * p_FD / S
    # Use actual FFT frequency resolution for SNR calculation
    dF_fft = 1 / T  # Frequency resolution of the FFT
    SNR = np.sqrt(dF_fft * np.sum(SNR2f))  # Signal-to-noise ratio
    Fchar = np.sum(f * SNR2f) / np.sum(SNR2f)  # Characteristic frequency
    bw = np.sqrt(np.sum((f - Fchar) ** 2 * SNR2f) / np.sum(SNR2f))  # Bandwidth
    h_rss = np.sqrt(dF_fft * np.sum(p_FD))  # Root-sum-square amplitude

    # ---- Bounding box
    if tfFrac > 0:
        prc_start = (1 - tfFrac) / 2
        prc_end = (1 + tfFrac) / 2

        cht2 = np.cumsum(p_TD) / np.sum(p_TD)
        tfbox_t_start = np.interp(prc_start, cht2, t)
        tfbox_t_end = np.interp(prc_end, cht2, t)
        tfbox_dur = tfbox_t_end - tfbox_t_start

        chf2 = np.cumsum(SNR2f) / np.sum(SNR2f)
        tfbox_f_start = np.interp(prc_start, chf2, f)
        tfbox_f_end = np.interp(prc_end, chf2, f)
        tfbox_bw = tfbox_f_end - tfbox_f_start

        tfbox = [tfbox_t_start, tfbox_f_start, tfbox_dur, tfbox_bw]
    else:
        tfbox = None

    return (
        SNR,
        h_rss,
        h_peak,
        Fchar,
        bw,
        Tchar,
        dur,
        Tchar_w,
        dur_w,
        tfbox,
        h_weighted,
    )
