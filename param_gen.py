import numpy as np
import matplotlib.pyplot as plt
from impedance.models.circuits import CustomCircuit
import warnings
import itertools # generate all possible combinations of elements in lists
import re
from tqdm import tqdm
import random
import pickle
import os


random.seed(42)
np.random.seed(42)

# Suppress the specific warning from impedance.py
warnings.filterwarnings("ignore", category=UserWarning, 
                        message="Simulating circuit based on initial parameters")

# Added by Yonatan
# Some control variables
file_path = "data/EIS_test.pkl"
n_combinations = 100
# Frequency
frequencies = None  # Use default frequencies
# frequencies = np.logspace(4, -2, 80)

# first we define function to randomly choose 100 EIS initial values

def generate_random_combinations(arrays_in_order, n_combinations=n_combinations):  # The n_combinations indicates how many EIS data to be generated per circuit
    # Determine the lengths of the input arrays
    lengths = [len(arr) for arr in arrays_in_order]
    
    # Calculate the total number of combinations
    total_combinations = np.prod(lengths)

    # Check if the requested number of combinations is larger than the total number of combinations
    if n_combinations > total_combinations:
        raise ValueError(f"Cannot generate more than {total_combinations} unique combinations.")
    
    # Create a set to store the combinations
    combinations = set()

    # Generate random combinations
    while len(combinations) < n_combinations:
        # Choose a random index for each array in arrays_in_order
        indices = [random.randrange(len(arr)) for arr in arrays_in_order]
        
        # Create a combination from these indices
        combination = tuple(arr[index] for arr, index in zip(arrays_in_order, indices))
        
        # Add this combination to the set
        combinations.add(combination)

    # Convert the set back to a list
    combinations = list(map(list, combinations))

    return combinations

# first we define function to randomly choose 100 EIS initial values

def generate_random_combinations(arrays_in_order, n_combinations=n_combinations):  # The n_combinations indicates how many EIS data to be generated per circuit
    # Determine the lengths of the input arrays
    lengths = [len(arr) for arr in arrays_in_order]
    
    # Calculate the total number of combinations
    total_combinations = np.prod(lengths)

    # Check if the requested number of combinations is larger than the total number of combinations
    if n_combinations > total_combinations:
        raise ValueError(f"Cannot generate more than {total_combinations} unique combinations.")
    
    # Create a set to store the combinations
    combinations = set()

    # Generate random combinations
    while len(combinations) < n_combinations:
        # Choose a random index for each array in arrays_in_order
        indices = [random.randrange(len(arr)) for arr in arrays_in_order]
        
        # Create a combination from these indices
        combination = tuple(arr[index] for arr, index in zip(arrays_in_order, indices))
        
        # Add this combination to the set
        combinations.add(combination)

    # Convert the set back to a list
    combinations = list(map(list, combinations))

    return combinations

def GenerateImpedance(R=(0, 0, 0, 0), C=(0, 0, 0, 0), CPE=(0, 0, 0, 0), CPE_alpha=0, L=(0, 0, 0, 0), Wo_0=0, Wo_1=0, circuit=None, frequencies=None):
    
    R0,R1,R2,R3=R
    C0,C1,C2,C3=C
    CPE0,CPE1,CPE2,CPE3=CPE
    L0,L1,L2,L3=L
    
    # Map element names to the corresponding numpy arrays
    
    elements_dict = {
        'R0': R0,
        'R1': R1,
        'R2': R2,
        'R3': R3,
        
        'C0': C0,
        'C1': C1,
        'C2': C2,
        'C3': C3,
        
        'Wo1': Wo_0,
        'Wo_1': Wo_1,
        
        'CPE0': CPE0,
        'CPE1': CPE1,
        'CPE2': CPE2,
        'CPE3': CPE3,
        
        'CPE_alpha0' : CPE_alpha,
        'CPE_alpha1' : CPE_alpha,
        'CPE_alpha2' : CPE_alpha,
        'CPE_alpha3' : CPE_alpha,
        
        'L0': L0,
        'L1': L1,
        'L2': L2,
        'L3': L3,
    }

    # Split the input string into element names using regex
    element_names = re.findall(r'(?:Wo1|[RCL]\d|CPE\d)', circuit)

    # Adding Wo_1 just after Wo1
    element_names = [name if name != 'Wo1' else ['Wo1', 'Wo_1'] for name in element_names]
    element_names = [item for sublist in element_names for item in (sublist if isinstance(sublist, list) else [sublist])]
    
     # Adding CPE_alpha0 just after CPE0
    element_names = [name if name != 'CPE0' else ['CPE0', 'CPE_alpha0'] for name in element_names]
    element_names = [item for sublist in element_names for item in (sublist if isinstance(sublist, list) else [sublist])]
    
    # Adding CPE_alpha1 just after CPE1
    element_names = [name if name != 'CPE1' else ['CPE1', 'CPE_alpha1'] for name in element_names]
    element_names = [item for sublist in element_names for item in (sublist if isinstance(sublist, list) else [sublist])]
    
    # Adding CPE_alpha2 just after CPE2
    element_names = [name if name != 'CPE2' else ['CPE2', 'CPE_alpha2'] for name in element_names]
    element_names = [item for sublist in element_names for item in (sublist if isinstance(sublist, list) else [sublist])]
    
    # Adding CPE_alpha2 just after CPE3
    element_names = [name if name != 'CPE3' else ['CPE3', 'CPE_alpha3'] for name in element_names]
    element_names = [item for sublist in element_names for item in (sublist if isinstance(sublist, list) else [sublist])]
    
    print(element_names)
    
    # Get the numpy arrays in the order specified by the input string
    arrays_in_order = [elements_dict[name] for name in element_names]
    

    # Generate the combinations
    combinations = generate_random_combinations(arrays_in_order)
    print(combinations)

    impedances = []
    
    for i in tqdm(range(len(combinations))):
        
        impedanceObject = CustomCircuit(circuit=circuit, initial_guess=combinations[i])
        
        # Generate the impedance of the circuit
        
        Z = impedanceObject.predict(frequencies)
        impedances.append(Z)
    
    return impedances

# function to generate points per decade
# start higher than end
# start 2 means 100 etc

def pointsPerDecade(start,end,points): 
    
    # Total number of points
    total_points = int((start - end + 1) * points)
    
    # Generate the frequency range
    output = np.logspace(start, end, num=total_points)

    return output

if frequencies is None:
    start_frequency = 5  # Corresponds to 10^5 Hz
    end_frequency = -2  # Corresponds to 10^-2 Hz

    # Number of points per decade
    points_per_decade = 10

    frequencies = pointsPerDecade(start_frequency,end_frequency,points_per_decade)

# function to normalize data

def process_impedances_norm(impedance_values):
    modulus_blocking = []
    phase_blocking = []

    for imp in impedance_values:
        real = imp.real
        imag = imp.imag

        modulus_val = np.sqrt(real ** 2 + imag ** 2)

        modulus_val = modulus_val - modulus_val[0]
        modulus_val = modulus_val / max(modulus_val)
        #modulus_val = modulus_val / modulus_val[-1]

        modulus_blocking.append(modulus_val)

    for imp in impedance_values:
        real = imp.real
        imag = imp.imag

        phase = np.arctan2(imag, real) * (180 / np.pi)  # convert radians to degrees

        phase_blocking.append(phase)

    real_blocking = []
    imag_blocking = []

    for modulus, phase in zip(modulus_blocking, phase_blocking):
        # convert phase to radians
        phase_rad = np.radians(phase)

        real_val = modulus * np.cos(phase_rad)
        imag_val = modulus * np.sin(phase_rad)

        real_blocking.append(real_val)
        imag_blocking.append(imag_val)

    impedances_blocking_normalized = []

    for real, imag in zip(real_blocking, imag_blocking):
        impedances_blocking_normalized.append(np.array(real) + 1j * np.array(imag))

    return impedances_blocking_normalized

def plot_impedances(impedance_values, frequencies, circuit):
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))
    
    plt.suptitle(f'Circuit: {circuit}', fontsize=20, fontweight='bold')  # set the title for the entire figure

    # Nyquist plot
    for impedance in impedance_values:
        # separate the real and imaginary parts
        real = impedance.real
        imag = impedance.imag
        axs[0].plot(real, -imag, linewidth=4.0)  # plot on the same figure
    axs[0].set_xlabel('Real Impedance / Unitless', fontsize=15, fontweight='bold')
    axs[0].set_ylabel('Imaginary Impedance / Unitless', fontsize=15, fontweight='bold')
    # axs[0].set_xlim(-0.45, 1.05)
    # axs[0].set_ylim(-0.45, 1.05)
    axs[0].set_title('Nyquist plot', fontsize=15, fontweight='bold')
    axs[0].tick_params(axis='both', which='major', labelsize=15)
    axs[0].grid(True)

    # Bode plot
    for impedance in impedance_values:
        # separate the real and imaginary parts
        real = impedance.real
        imag = impedance.imag

        # calculate the phase and modulus
        phase = np.arctan2(imag, real) * (180 / np.pi)  # phase in degrees
        modulus = np.sqrt(real**2 + imag**2)  # modulus

        # plot modulus
        axs[1].semilogx(frequencies, modulus, linewidth=4.0)  
        axs[1].set_xlabel('Frequency / Hz', fontsize=15, fontweight='bold')
        axs[1].set_ylabel('Impedance Modulus / Unitless', fontsize=15, fontweight='bold')
        axs[1].set_title('Bode plot - Impedance Modulus', fontsize=15, fontweight='bold')
        axs[1].tick_params(axis='both', which='major', labelsize=15)
        axs[1].grid(True)

        # plot phase
        axs[2].semilogx(frequencies, phase, linewidth=4.0)  
        axs[2].set_xlabel('Frequency / Hz', fontsize=15, fontweight='bold')
        axs[2].set_ylabel('Phase (Degrees)', fontsize=15, fontweight='bold')
        axs[2].set_title('Bode plot - Phase', fontsize=15, fontweight='bold')
        axs[2].tick_params(axis='both', which='major', labelsize=15)
        axs[2].grid(True)

    plt.tight_layout()
    # plt.savefig(str(circuit)+'.png', dpi=100)
    plt.show()
