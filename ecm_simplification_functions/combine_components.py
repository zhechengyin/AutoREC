"""Collection of functions to combine resistors, capacitors, and inductors in series and
parallel configurations.
"""

import numpy as np


def direct_sum(*values):
    return np.sum(values)


def reciprocal_sum(*values):
    return 1.0 / np.sum(1.0 / np.array(values))


def combine_R(*values, connection):
    if connection == "series":
        return direct_sum(*values)
    elif connection == "parallel":
        return reciprocal_sum(*values)


def combine_C(*values, connection):
    if connection == "series":
        return reciprocal_sum(*values)
    elif connection == "parallel":
        return direct_sum(*values)


def combine_L(*values, connection):
    return combine_R(*values, connection=connection)


def combine_components(*values, component_type, connection):
    if component_type == "R":
        return combine_R(*values, connection=connection)
    elif component_type == "C":
        return combine_C(*values, connection=connection)
    elif component_type == "L":
        return combine_L(*values, connection=connection)
