"""
Simple Self-Organizing Map

Parameters:
    data: numpy array of arrays, where rows = data points and columns = features (default = iris dataset)
        "iris" = sci-kit learn Iris dataset
        "cancer" = sci-kit learn Cancer dataset
    map_size: integer, side length of square map (default = calculated as 5*sqrt(# of samples in data))
    init_learning_rate: float, initial learning rate (default = 1.0)
    n_iterations: integer, number of iterations (default = 300)
    neighbourhood_function: string (default = "ricker")
        "step" = step function (neighbouring neurons coded as "1", all others "0")
        "gaussian" = gaussian function
        "ricker" = ricker wavelet function
    test_map: boolean (default = TRUE)
        TRUE = map will be tested using datapoints with known classes (supports up to 10 classes)
        FALSE = map will not be tested
    test_classes: numpy array of length = #datapoints in data; values should correspond to known classes of data (default = Iris dataset)
        Must be included when test_map = True
    num_tests: integer, number of datapoints to use during testing (default = 30)
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris, load_breast_cancer


def som(data="iris", map_size=0, init_learning_rate=1.0, n_iterations=300, neighbourhood_type="ricker", test_map=True, test_classes=0, num_tests=30):
    # main function
    try:
        x = get_data(data, test_classes)[0]
        y = get_data(data,test_classes)[1]
        global g_map
        if map_size == 0:
            map_size = int(np.round(np.sqrt(5*(np.sqrt(x.shape[0])))))
        g_map = map_size
        init_radius = max(map_size, map_size) / 2
        time_constant = n_iterations / np.log(init_radius)
        n_features = x.shape[1]
        weights = np.random.rand(map_size, map_size, n_features)
        results = run_som(x, weights, init_learning_rate, n_iterations, neighbourhood_type, time_constant, init_radius)
        visualize_som_weights(results)
        if test_map:
            run_test(x,y,results,num_tests)
        else:
            plt.show()
    except Exception as err:
        print ("Error:",err)
        raise
    except:
        print ("Error:")
        print (*check_params(x, map_size, init_learning_rate, n_iterations, neighbourhood_type, test_map,y,num_tests),sep='\n')
    return

def get_data(data, classes):
    if data == "iris":
        dataset = load_iris()
        x = dataset.data
        y = dataset.target
    if data == "cancer":
        dataset = load_breast_cancer()
        x = dataset.data
        y = dataset.target
    return x, y


def check_params(data, map_size, init_learning_rate, n_iterations, neighbourhood_type, test_map, test_classes, num_tests):
    # checks for parameter errors
    errors = []
    valid_data = ["iris","cancer"]
    if not isinstance(data, np.ndarray):
        if data not in valid_data:
            errors.append("Data must be a numpy array or a string reading one of {}".format(valid_data))
    if not isinstance(map_size, int):
        errors.append("Map size must be a positive integer")
    if not isinstance(init_learning_rate, (float, int)):
        errors.append("Initial learning rate must be a float")
    if not isinstance(n_iterations, int):
        errors.append("Number of iterations must be a positive integer")
    valid_inputs = ["step", "gaussian", "ricker"]
    if neighbourhood_type not in valid_inputs:
        errors.append("Neighbourhood type must be one of {}".format(valid_inputs))
    if not isinstance(test_map, bool):
        errors.append("Test map must be a boolean")
    if not isinstance(test_classes, np.ndarray):
        errors.append("Test classes must be a numpy array")
    elif test_classes.shape[0] != data.shape[0]:
        errors.append("Test classes and data must have the same number of rows")
    if not isinstance(num_tests, int):
        errors.append("Number of tests must be a positive integer")
    if not errors:
        errors.append("Error unknown")
    return errors


def calc_distance(v1, v2):
    # calculates Euclidean distance between two vectors
    return np.linalg.norm(v1 - v2)

def find_bmu(input_vector, weights):
    # finds best matching unit for an input vector
    bmu_idx = None
    min_dist = np.inf
    for i in range(g_map):
        for j in range(g_map):
            w = weights[i, j]
            dist = calc_distance(input_vector, w)
            if dist < min_dist:
                min_dist = dist
                bmu_idx = (i, j)
    return bmu_idx

def neighbourhood(dist_to_bmu, radius, type):
    # determines where a neuron is within a neighbourhood function
    if type == "step":
        if dist_to_bmu < radius:
            return 1
        else:
            return 0
    elif type == "gaussian":
        return np.exp(-dist_to_bmu**2 / (2 * radius**2))
    elif type == "ricker":
        excit = np.exp(-dist_to_bmu**2 / (2 * radius**2))
        inhib = np.exp(-dist_to_bmu**2 / (radius**2))
        return excit - inhib

def update_weights(input_vector, bmu_idx, iteration, weights, learning_rate, time_constant, neighbourhood_type, init_radius):
    # updates neuron weights
    radius = init_radius * np.exp(-iteration / time_constant)
    for i in range(g_map):
        for j in range(g_map):
            w = weights[i, j]
            dist_to_bmu = calc_distance(np.array([i, j]), np.array(bmu_idx))
            if dist_to_bmu < radius:
                influence = neighbourhood(dist_to_bmu, radius, neighbourhood_type)
                weights[i, j] += influence * learning_rate * (input_vector - w)
    return weights

def visualize_som_weights(weights):
    plt.figure()
    sns.heatmap(weights[:, :, 0])
    #plt.show()

def run_som(data, weights, init_learning_rate, n_iterations, neighbourhood_type, time_constant, init_radius):
    # loops through all iterations
    learning_rate = init_learning_rate
    for iteration in range(n_iterations):
        input_vector = data[np.random.randint(0, data.shape[0])]
        bmu_idx = find_bmu(input_vector, weights)
        weights = update_weights(input_vector, bmu_idx, iteration, weights, learning_rate, time_constant, neighbourhood_type, init_radius)
        learning_rate = init_learning_rate * np.exp(-iteration / n_iterations)
    return weights

def infer_som(input_vector, weights):
    # finds best matching unit for a known datapoint
    inferred_index = find_bmu(input_vector, weights)
    return inferred_index

def test_som(data,classes,weights):
    # tests and plots known datapoints
    colours = ["blue","yellow","green","red","purple","brown","orange","white","black","pink"]
    test_index = np.random.randint(0, data.shape[0])
    test_data = data[test_index]
    test_location = infer_som(test_data, weights)
    correct_class = classes[test_index]
    plt.scatter(test_location[0], test_location[1], color=colours[correct_class])

def run_test(data,classes,weights,num_tests):
    # runs test using known datapoints
    for i in range(num_tests):
        test_som(data,classes,weights)
    plt.show()

som()



