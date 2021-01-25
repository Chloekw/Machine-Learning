import torch
import torch.nn as nn
from sklearn.datasets import load_digits
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pylab as pl

class OneLayerNet(nn.Module):
  def __init__(self):
    super(OneLayerNet, self).__init__()
    self.fc = nn.Linear(2, 1, bias = True)

  def forward(self, x):
    x = self.fc(x)

    return F.relu(torch.sign(x))

class TwoLayerNet(nn.Module):
  def __init__(self):
    super(TwoLayerNet, self).__init__()
    self.fc1 = nn.Linear(2, 2, bias = True)
    self.fc2 = nn.Linear(2, 1, bias=True)

  def forward(self, x):
    return self.fc2(F.relu(self.fc1(x)))

class ThreeLayerNet(nn.Module):
  def __init__(self):
    super(ThreeLayerNet, self).__init__()
    self.fc1 = nn.Linear(64, 64, bias=True)
    self.fc2 = nn.Linear(64, 32, bias=True)
    self.fc3 = nn.Linear(32, 1, bias=True)

  def forward(self, x):
    return self.fc3(F.relu(self.fc2(F.relu(self.fc1(x)))))

class ConvNet(nn.Module):
  def __init__(self):
    super(ConvNet, self).__init__()
    self.conv1 = nn.Conv2d(1, 8, 3)
    self.conv3 = nn.Conv2d(8, 4, 3)
    self.fc = nn.Linear(4 , 1, bias=True)
    self.maxpool = nn.MaxPool2d(kernel_size = 2)
    self.identity = nn.Identity()

  def forward(self, x):
    x = F.relu(self.conv1(x))
    x = self.maxpool(x)
    x = F.relu(self.conv3(x)).view(-1,4)
    x = self.fc(x)
    return x

# For the extra credit, I design a neural network with explicitly described as digits classification.
# The first layer is a fully-connected linear layer with 64 inputs and 64 outputs with activation function relu
# The second layer is a linear layer with 64 inputs and 10 outputs with activation function log softmax with dim=1, to distinguish digits 0-9 (so it is 10 classes)
# The third layer is a linear layer with 10 inputs and 2 outputs with activation function relu, to implicitly decribed two output classes
# The forth layer is just a linear layer with 2 inputs and map to 1 output with identity activation function

class ExtraCreditNet(nn.Module):
  def __init__(self):
    super(ExtraCreditNet, self).__init__()
    self.fc = nn.Linear(64, 64, bias=True)
    self.fc1 = nn.Linear(64, 10, bias=True)
    self.fc2 = nn.Linear(10, 2, bias=True)
    self.fc3 = nn.Linear(2, 1, bias=True)

  def forward(self, x):
    x = F.relu(self.fc(x))
    x = F.log_softmax(self.fc1(x), dim=1)
    x = F.relu(self.fc2(x))
    return self.fc3(x)




def XOR_data():
  X = torch.tensor([[-1., -1.], [1., -1.], [-1., 1.], [1., 1.]])
  Y = (torch.prod(X, dim=1) < 0.).float()
  return X, Y.view(-1,1)

def digits_data():
  digits, labels = load_digits(return_X_y=True)
  # pl.gray()
  # pl.matshow(digits)
  # pl.show()
  digits = torch.tensor(digits.reshape([-1, 8, 8]), dtype=torch.float)
  labels = torch.tensor(labels.reshape([-1, 1]) % 2 == 1, dtype=torch.float)
  test_digits = digits[:180,:,:]
  test_labels = labels[:180]
  digits = digits[180:,:,:]
  labels = labels[180:]
  return digits, labels, test_digits, test_labels

def gradient_descent(net, X, Y, num_iterations, eta):
  objective_fn = nn.BCEWithLogitsLoss()
  with torch.no_grad():
    objective_values = [ objective_fn(net(X), Y).item() ]
    error_rates = [ error_rate(net(X), Y).item() ]
  for _ in range(num_iterations):
    pred = net(X)
    loss = objective_fn(pred, Y)
    objective_values.append(loss.item())
    error_rates.append(error_rate(pred, Y).item())
    net.zero_grad()
    loss.backward()
    with torch.no_grad():
      for p in net.parameters():
        p -= eta * p.grad
  return objective_values, error_rates

def error_rate(Yhat, Y):
  return ((torch.sign(Yhat) > 0).float() != Y).float().mean()

if __name__ == '__main__':
  XOR_X, XOR_Y = XOR_data()
  digits, labels, test_digits, test_labels = digits_data()
  # print(digits.size())

  num_iterations = 25
  eta = 1.0
  x_axis = [0]+[i + 1 for i in range(num_iterations)]

#################### 1-layer ####################
  torch.manual_seed(0)
  net1 = OneLayerNet()
  loss, err = gradient_descent(net1, XOR_X, XOR_Y, num_iterations, eta)

  print("initial objective value: ", loss[0])
  print("initilal error rate: ", err[0])
  print("final objective value: ", loss[-1])
  print("final error rate: ", err[-1])

  plt.plot(x_axis, loss)
  plt.ylabel('value of objective function')
  plt.xlabel('iteration')
  plt.legend(['one-layer NN Objective function'])
  plt.show()

  plt.plot(x_axis, err)
  plt.ylabel('value of training error rate')
  plt.xlabel('iteration')
  plt.legend(['one-layer NN Training Error Rate'])
  plt.show()

# ########################### 2-layer #######################################
  torch.manual_seed(0)
  net2 = TwoLayerNet()
  loss2, err2 = gradient_descent(net2, XOR_X, XOR_Y, num_iterations, eta)


  print("initial objective value: ", loss2[0])
  print("initilal error rate: ", err2[0])
  print("final objective value: ", loss2[-1])
  print("final error rate: ", err2[-1])

  plt.plot(x_axis, loss2)
  plt.ylabel('value of objective function')
  plt.xlabel('iteration')
  plt.legend(['two-layer NN Objective function'])
  plt.show()

  plt.plot(x_axis, err2)
  plt.ylabel('value of training error rate')
  plt.xlabel('iteration')
  plt.legend(['two-layer NN Training Error Rate'])
  plt.show()

  num_iterations = 500
  eta = 0.1
  x_axis = [0] + [i for i in range(num_iterations)]

########################### 3-layer ##########################
  torch.manual_seed(0)
  net3 = ThreeLayerNet()
  # XXX train three-layer net on digits data
  X = digits.view(-1, 64)


  loss3, err3 = gradient_descent(net3, X, labels, num_iterations, eta)

  print("initial objective value: ", loss3[0])
  print("initilal error rate: ", err3[0])
  print("final objective value: ", loss3[-1])
  print("final error rate: ", err3[-1])

  plt.plot(x_axis, loss3)
  plt.ylabel('value of objective function')
  plt.xlabel('iteration')
  plt.legend(['three-layer NN Objective function'])
  plt.show()

  plt.plot(x_axis, err3)
  plt.ylabel('value of training error rate')
  plt.xlabel('iteration')
  plt.legend(['three-layer NN Training Error Rate'])
  plt.show()

  # print(test_digits.view(-1,64).size(1))
  print("ThreeLayerNet: Test error rate: {0}".format(error_rate(net3(test_digits.view(-1,64)),test_labels)))

################ conv ##################
  torch.manual_seed(0)
  net4 = ConvNet()
  # XXX train conv net on digits data
  X = digits.unsqueeze(1)

  loss4, err4 = gradient_descent(net4, X, labels, num_iterations, eta)
  print("initial objective value: ", loss4[0])
  print("initilal error rate: ", err4[0])
  print("final objective value: ", loss4[-1])
  print("final error rate: ", err4[-1])

  plt.plot(x_axis, loss4)
  plt.ylabel('value of objective function')
  plt.xlabel('iteration')
  plt.legend(['Convolutional NN Objective function'])
  plt.show()

  plt.plot(x_axis, err4)
  plt.ylabel('value of training error rate')
  plt.xlabel('iteration')
  plt.legend(['Convolutional NN Training Error Rate'])
  plt.show()
  print("ConvNet: Test error rate: {0}".format(error_rate(net4(test_digits.unsqueeze(1)),test_labels)))

########## extra credit ##########
  torch.manual_seed(0)
  net5 = ExtraCreditNet()
  # XXX train conv net on digits data
  X = digits.view(-1, 64)

  loss5, err5 = gradient_descent(net5, X, labels, num_iterations, eta)
  print("initial objective value: ", loss5[0])
  print("initilal error rate: ", err5[0])
  print("final objective value: ", loss5[-1])
  print("final error rate: ", err5[-1])

  plt.plot(x_axis, loss5)
  plt.ylabel('value of objective function')
  plt.xlabel('iteration')
  plt.legend(['(extra credit) NN Objective function'])
  plt.show()

  plt.plot(x_axis, err5)
  plt.ylabel('value of training error rate')
  plt.xlabel('iteration')
  plt.legend(['(extra credit) NN Training Error Rate'])
  plt.show()
  print("(extra credit): Test error rate: {0}".format(error_rate(net5(test_digits.view(-1, 64)), test_labels)))
