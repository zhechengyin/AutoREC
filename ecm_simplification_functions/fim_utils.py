import numpy as np
import numdifftools as nd
import autoeis as ae


class DefaultTransform:
    def __init__(self):
        pass

    def transform(self, params):
        return params

    def inverse_transform(self, transformed_params):
        return transformed_params

    def __call__(self, x):
        return self.transform(x)


class Transform:
    def __init__(self, param_names):
        self.param_names = param_names

    def transform(self, params):
        """Transform the parameters from the original space to the transformed space."""
        transformed_params = []
        for ii, name in enumerate(self.param_names):
            if name.startswith("P") and name.endswith("n"):
                # Pn parameter
                transformed_params = np.append(transformed_params, params[ii])
            else:
                # Other parameters
                transformed_params = np.append(transformed_params, np.log(params[ii]))
        return transformed_params

    def inverse_transform(self, transformed_params):
        """Transform the parameters from the transformed space back to original space."""
        params = []
        for ii, name in enumerate(self.param_names):
            if name.startswith("P") and name.endswith("n"):
                # Pn parameter
                params = np.append(params, transformed_params[ii])
            else:
                # Other parameters
                params = np.append(params, np.exp(transformed_params[ii]))
        return params

    def __call__(self, x):
        return self.transform(x)


class FIM_nd:
    """Abstract base class for the FIM modules."""

    def __init__(self, model, transform=None, **kwargs):
        """Instantiate FIM class"""
        self.model = model
        self.jac_func = nd.Jacobian(self._model_wrapper, **kwargs)

        # Get the parameter transformation
        if transform is None:
            self.transform = DefaultTransform()
        else:
            self.transform = transform

    def Jacobian(self, x, *args, **kwargs):
        """Compute the Jacobian of the model, evaluated at parameter ``x``.
        Parameter ``x`` should be written in the parameterization that the model
        uses.
        """
        params = self.transform(x)
        return self.jac_func(params, *args, **kwargs)

    def FIM(self, x, *args, **kwargs):
        """Compute the FIM."""
        Jac = self.Jacobian(x, *args, **kwargs)
        return Jac.T @ Jac

    def __call__(self, x, *args, **kwargs):
        return self.FIM(x, *args, **kwargs)

    def _model_wrapper(self, x, *args, **kwargs):
        """This is the function that we feed into the function that computes the
        Jacobian.
        """
        # Transform the parameters
        params_orig = self.transform.inverse_transform(x)
        # Evaluate model
        return self.model(params_orig, *args, **kwargs)


def normalize_values(values, reference):
    min_val = np.min(reference)
    max_val = np.max(reference)
    return (values - min_val) / (max_val - min_val)


def eis_wrapper_fn(circuit, freq, Z, mode="nyquist", normalize=False):
    circuit_fn = ae.utils.generate_circuit_fn(circuit)

    def eval_fn(params):
        Zpreds = circuit_fn(freq, params)
        if mode == "nyquist":
            Zre = Zpreds.real
            Zim = Zpreds.imag
            if normalize:
                Zre = normalize_values(Zre, Z.real)
                Zim = -normalize_values(-Zim, -Z.imag)
            return np.append(Zre, Zim)
        elif mode == "bode":
            mag = np.abs(Zpreds)
            phi = np.angle(Zpreds)
            if normalize:
                mag = normalize_values(mag, np.abs(Z))
                phi = normalize_values(phi, np.angle(Z))
            return np.append(np.log10(mag), phi)

    return eval_fn


def compute_fim(circuit, freq, Z, params_array, mode="bode", verbose=False):
    eval_fn = eis_wrapper_fn(circuit, freq, Z, mode)
    transform = Transform(ae.parser.get_parameter_labels(circuit))
    fim_fn = FIM_nd(eval_fn, transform)
    fim = fim_fn(params_array)
    eigvals, eigvecs = np.linalg.eigh(fim / np.linalg.norm(fim))  # Normalize FIM
    eigvals /= max(eigvals)  # Normalize eigenvalues to [0, 1]
    # eigvals in an ascending order
    if verbose:
        print("Eigenvalues of FIM (normalized):", eigvals)
    return fim, eigvals, eigvecs
