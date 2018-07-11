import mnist_loader
from network import Network

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

net = Network(
    [784, 30, 10],
    random_init=False
)

net.SGD(training_data=training_data, epochs=30, mini_batch_size=10, eta=3.0, test_data=test_data,
        random_shuffle=True)
