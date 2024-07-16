import scipy.io

# Load the .mat files
train_data = scipy.io.loadmat('few_shot_data/Amazon_clothing_train.mat')
test_data = scipy.io.loadmat('few_shot_data/Amazon_clothing_test.mat')
network_data = scipy.io.loadmat('few_shot_data/Amazon_clothing_network.mat')

# Display the keys in each .mat file
print(train_data.keys())
print(test_data.keys())
print(network_data.keys())
