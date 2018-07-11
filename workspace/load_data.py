import pickle

with open("/Users/zy/Documents/workspace/neural-networks-and-deep-learning/data/mnist.pkl") as fid:
    data = pickle.load(fid)

print(data[0][0].shape)
print(data[0][1].shape)