# import jax.numpy as jnp
import numpy as np
from sklearn.preprocessing import StandardScaler
from autoeis.utils import generate_circuit_fn

avail_methods = [
    "UW",
    "X2",
    "WX2",
    "PW",
    "B",
    "log-B",
    "log-BW",
    "chi-squared",
    "normalized-chi-squared",
    "chi-phase",
    "chi-magnitude",
]


class EISObjective:
    def __init__(self, circuit, freq, Z, method="UW"):
        msg = (
            f"Invalid method: {method}. Use 'chi-squared', 'nyquist', 'bode', or 'magnitude'."
        )
        assert method in avail_methods, msg
        self.method = method

        assert len(freq) == len(Z), "Length of frequency and impedance data must match."

        self.fn = generate_circuit_fn(circuit, jit=True)
        self.freq = freq
        self.Z = Z
        self.mag_gt = np.abs(Z)
        self.phase_gt = np.angle(Z)

        # Normalize impedance data
        self.scaler_real = StandardScaler().fit(Z.real.reshape(-1, 1))
        self.scaler_imag = StandardScaler().fit(Z.imag.reshape(-1, 1))

    def obj_UW(self, p):
        """Computes ECM error based on the Nyquist plot."""
        Z_pred = self.fn(self.freq, p)
        res = np.hstack((Z_pred.real - self.Z.real, Z_pred.imag - self.Z.imag))
        return res

    def obj_X2(self, p):
        """Computes ECM error based on residual-based χ2."""
        Z_pred = self.fn(self.freq, p)
        residual_real = Z_pred.real - self.Z.real
        residual_imag = Z_pred.imag - self.Z.imag
        weight = 1 / np.sqrt(self.Z.real**2 + self.Z.imag**2)
        res = np.hstack((residual_real * weight, residual_imag * weight))
        return res

    def obj_WX2(self, p, weights):
        """Computes ECM error based on residual-based χ2."""
        Z_pred = self.fn(self.freq, p)
        residual_real = Z_pred.real - self.Z.real
        residual_imag = Z_pred.imag - self.Z.imag
        w = weights / np.sqrt(self.Z.real**2 + self.Z.imag**2)
        res = np.hstack((residual_real * w, residual_imag * w))
        return res

    def obj_PW(self, p):
        """Computes ECM error based on residual-based χ2."""
        Z_pred = self.fn(self.freq, p)
        residual_real = Z_pred.real - self.Z.real
        residual_imag = Z_pred.imag - self.Z.imag
        weight_real = 1 / self.Z.real
        weight_imag = 1 / self.Z.imag
        res = np.hstack((residual_real * weight_real, residual_imag * weight_imag))
        return res

    def obj_B(self, p):
        """Computes ECM error based on the Bode plot."""
        Z_pred = self.fn(self.freq, p)
        mag = np.abs(Z_pred)
        phase = np.angle(Z_pred)
        res = np.hstack((mag - self.mag_gt, phase - self.phase_gt))
        # res = np.hstack((mag - self.mag_gt, phase - self.phase_gt))
        return res

    def obj_log_B(self, p):
        """Computes ECM error based on the Bode plot."""
        Z_pred = self.fn(self.freq, p)
        mag = np.abs(Z_pred)
        phase = np.angle(Z_pred)
        res = np.hstack((np.log10(mag) - np.log10(self.mag_gt), phase - self.phase_gt))
        # res = np.hstack((mag - self.mag_gt, phase - self.phase_gt))
        return res

    def obj_log_BW(self, p):
        """Computes ECM error based on the Bode plot."""
        Z_pred = self.fn(self.freq, p)
        mag = np.abs(Z_pred)
        phase = np.angle(Z_pred)
        res = np.hstack(
            (
                np.log10(mag / self.mag_gt) / np.log10(self.mag_gt),
                (phase - self.phase_gt) / self.phase_gt,
            )
        )
        # res = np.hstack((mag - self.mag_gt, phase - self.phase_gt))
        return res

    def obj_chi_squared(self, p):
        """Computes ECM error based on residual-based χ2."""
        Z_pred = self.fn(self.freq, p)
        residual = (Z_pred.real - self.Z.real) ** 2 + (Z_pred.imag - self.Z.imag) ** 2
        weight = 1 / (self.Z.real**2 + self.Z.imag**2)
        return residual * weight

    def obj_normalized_chi_squared(self, p):
        """Computes ECM error based on normalized residual-based χ2."""
        Z_data_real = self.scaler_real.transform(self.Z.real.reshape(-1, 1)).flatten()
        Z_data_imag = self.scaler_imag.transform(self.Z.imag.reshape(-1, 1)).flatten()

        Z_pred = self.fn(self.freq, p)
        Z_pred_real = self.scaler_real.transform(Z_pred.real.reshape(-1, 1)).flatten()
        Z_pred_imag = self.scaler_imag.transform(Z_pred.imag.reshape(-1, 1)).flatten()

        return np.append(Z_pred_real - Z_data_real, Z_pred_imag - Z_data_imag) ** 2

    def obj_phase_chi(self, p):
        """Computes ECM error based on the Bode plot."""
        Z_pred = self.fn(self.freq, p)

        residual = (Z_pred.real - self.Z.real) ** 2 + (Z_pred.imag - self.Z.imag) ** 2
        weight = 1 / (self.Z.real**2 + self.Z.imag**2)

        phase = np.angle(Z_pred)
        res = np.hstack(((phase - self.phase_gt) / self.phase_gt, residual * weight))
        return res

    def obj_mag(self, p):
        """Computes ECM error based on the magnitude of impedance deviation."""
        Z_pred = self.fn(self.freq, p)
        res = np.abs(Z - Z_pred)
        return res

    def __call__(self, p, *args, **kwargs):
        if self.method == "UW":
            return self.obj_UW(p, *args, **kwargs)
        elif self.method == "X2":
            return self.obj_X2(p, *args, **kwargs)
        elif self.method == "WX2":
            return self.obj_WX2(p, *args, **kwargs)
        elif self.method == "PW":
            return self.obj_PW(p, *args, **kwargs)
        elif self.method == "B":
            return self.obj_B(p, *args, **kwargs)
        elif self.method == "log-B":
            return self.obj_log_B(p, *args, **kwargs)
        elif self.method == "log-BW":
            return self.obj_log_BW(p, *args, **kwargs)
        elif self.method == "chi-squared":
            return self.obj_chi_squared(p, *args, **kwargs)
        elif self.method == "normalized-chi-squared":
            return self.obj_normalized_chi_squared(p, *args, **kwargs)
        elif self.method == "chi-phase":
            return self.obj_phase_chi(p, *args, **kwargs)
        elif self.method == "chi-magnitude":
            return self.obj_mag(p, *args, **kwargs)


class EISObjectiveLog(EISObjective):
    def __init__(self, circuit, freq, Z, method="UW"):
        super().__init__(circuit, freq, Z, method)

    def __call__(self, logp, *args, **kwargs):
        p = np.exp(logp)
        return super().__call__(p, *args, **kwargs)
