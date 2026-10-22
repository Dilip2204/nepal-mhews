"""USGS Tectonic Event & ObsPy High-Frequency Rock-Ice Detachment Detector.

Implements Dynamic Precursor Trigger 4 from AGENTS.md:
  4. Seismic Trigger: M4.0+ tectonic event or high-frequency environmental seismic
     signature indicating rock-ice detachment.
"""

from typing import Dict, List, Sequence, Tuple
import numpy as np
from app.core.config import settings


def _compute_band_energy_ratio_and_kurtosis(
    wf: np.ndarray,
    sampling_rate_hz: float,
    high_freq_band_hz: Tuple[float, float],
) -> Tuple[float, float]:
    """Compute 2-15 Hz spectral energy ratio and envelope kurtosis (SciPy or NumPy FFT)."""
    try:
        from scipy import signal, stats

        freqs, psd = signal.welch(wf, fs=sampling_rate_hz, nperseg=min(256, wf.size))
        trapz_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))
        total_power = float(trapz_fn(psd, freqs))
        band_mask = (freqs >= high_freq_band_hz[0]) & (freqs <= high_freq_band_hz[1])
        band_power = float(trapz_fn(psd[band_mask], freqs[band_mask]))
        hf_ratio = band_power / max(total_power, 1e-12)

        envelope = np.abs(signal.hilbert(wf))
        env_kurtosis = float(stats.kurtosis(envelope, fisher=False))
        return hf_ratio, env_kurtosis
    except Exception:
        # Pure NumPy FFT periodogram & analytic Hilbert transform fallback
        freqs = np.fft.rfftfreq(wf.size, d=1.0 / sampling_rate_hz)
        psd = (np.abs(np.fft.rfft(wf)) ** 2) / wf.size
        total_power = float(np.sum(psd))
        band_mask = (freqs >= high_freq_band_hz[0]) & (freqs <= high_freq_band_hz[1])
        band_power = float(np.sum(psd[band_mask]))
        hf_ratio = band_power / max(total_power, 1e-12)

        # Analytic signal via FFT for envelope kurtosis
        fft_vals = np.fft.fft(wf)
        h = np.zeros(wf.size)
        if wf.size % 2 == 0:
            h[0] = h[wf.size // 2] = 1
            h[1 : wf.size // 2] = 2
        else:
            h[0] = 1
            h[1 : (wf.size + 1) // 2] = 2
        envelope = np.abs(np.fft.ifft(fft_vals * h))
        env_mean = float(np.mean(envelope))
        env_std = max(float(np.std(envelope)), 1e-12)
        env_kurtosis = float(np.mean(((envelope - env_mean) / env_std) ** 4))
        return hf_ratio, env_kurtosis


def detect_rock_ice_avalanche_signature(
    waveform: Sequence[float],
    sampling_rate_hz: float = 100.0,
    high_freq_band_hz: Tuple[float, float] = (2.0, 15.0),
) -> Dict[str, object]:
    """Analyze seismic waveform for rock-ice detachment / mass-flow signature.

    Tectonic earthquakes exhibit impulsive P/S arrivals (high kurtosis > 10) and lower
    dominant frequencies, whereas rock-ice avalanches produce emergent, spindle-shaped
    envelopes with sustained energy in the 2-15 Hz band and lower kurtosis (typically < 8).

    Args:
        waveform: 1D seismic velocity/acceleration time series (counts or m/s).
        sampling_rate_hz: Instrument sampling rate in Hz (default 100 Hz).
        high_freq_band_hz: Bandpass window (low_hz, high_hz) for mass-wasting energy.

    Returns:
        Dictionary of spectral and envelope diagnostics + `is_rock_ice_detachment`.
    """
    wf = np.asarray(waveform, dtype=np.float64)
    if wf.size < 64:
        raise ValueError("Waveform must contain at least 64 samples.")

    wf = wf - np.mean(wf)
    max_amp = float(np.max(np.abs(wf)))
    if max_amp < 1e-12:
        return {
            "high_freq_energy_ratio": 0.0,
            "kurtosis": 0.0,
            "sta_lta_max": 0.0,
            "is_rock_ice_detachment": False,
        }

    hf_ratio, env_kurtosis = _compute_band_energy_ratio_and_kurtosis(
        wf, sampling_rate_hz, high_freq_band_hz
    )

    # Classic STA/LTA ratio on squared signal
    sq = wf * wf
    n_sta = max(int(0.5 * sampling_rate_hz), 4)
    n_lta = max(int(5.0 * sampling_rate_hz), n_sta * 2)
    if wf.size >= n_lta:
        sta = np.convolve(sq, np.ones(n_sta) / n_sta, mode="valid")
        lta = np.convolve(sq, np.ones(n_lta) / n_lta, mode="valid")
        min_len = min(sta.size, lta.size)
        sta_lta = sta[-min_len:] / np.maximum(lta[-min_len:], 1e-12)
        sta_lta_max = float(np.max(sta_lta))
    else:
        sta_lta_max = float(np.max(sq) / max(np.mean(sq), 1e-12))

    is_detachment = bool(
        hf_ratio >= 0.45 and env_kurtosis < 9.5 and sta_lta_max >= 1.8
    )

    return {
        "high_freq_energy_ratio": round(hf_ratio, 4),
        "kurtosis": round(env_kurtosis, 3),
        "sta_lta_max": round(sta_lta_max, 3),
        "is_rock_ice_detachment": is_detachment,
    }


def evaluate_seismic_trigger(
    events: List[Dict[str, object]],
    waveform: Sequence[float] = (),
    sampling_rate_hz: float = 100.0,
    min_magnitude: float = settings.seismic_min_magnitude,
) -> Dict[str, object]:
    """Evaluate combined tectonic (M4.0+) and environmental rock-ice seismic triggers."""
    max_mag = 0.0
    qualifying_events: List[Dict[str, object]] = []
    for ev in events:
        mag = float(ev.get("magnitude", 0.0))
        max_mag = max(max_mag, mag)
        if mag >= min_magnitude:
            qualifying_events.append(ev)

    tectonic_trigger = len(qualifying_events) > 0
    estimated_kh = (
        round(min(0.35, 0.04 * (10 ** (0.25 * (max_mag - 4.0)))), 3)
        if tectonic_trigger
        else 0.0
    )

    waveform_result = {
        "high_freq_energy_ratio": 0.0,
        "kurtosis": 0.0,
        "sta_lta_max": 0.0,
        "is_rock_ice_detachment": False,
    }
    if len(waveform) >= 64:
        waveform_result = detect_rock_ice_avalanche_signature(
            waveform=waveform, sampling_rate_hz=sampling_rate_hz
        )

    trigger_active = bool(
        tectonic_trigger or waveform_result["is_rock_ice_detachment"]
    )
    return {
        "max_magnitude": round(max_mag, 2),
        "tectonic_trigger_active": tectonic_trigger,
        "qualifying_event_count": len(qualifying_events),
        "estimated_seismic_kh": estimated_kh,
        "waveform_diagnostics": waveform_result,
        "trigger_active": trigger_active,
    }
