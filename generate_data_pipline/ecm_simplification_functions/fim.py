import numpy as np
import numdifftools as nd


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
