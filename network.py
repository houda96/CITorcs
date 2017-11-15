# Code in file nn/two_layer_net_nn.py
import torch
from torch.autograd import Variable


class Network:
    def __init__(self, data, labels, learning_rate=1e-5, H=100, n_iterations=500):

        # N is batch size; D_in is input dimension;
        # H is hidden dimension; D_out is output dimension.
        # N, D_in, H, D_out = 64, 1000, 100, 10
        # self.data = data
        # self.labels = labels
        self.N = data.shape[0]
        self.D_in = data.shape[1]
        self.H = H #?
        self.D_out = labels.shape[1]
        self.n_iterations = n_iterations
        self.loss_saved = []
        # Create random Tensors to hold inputs and outputs, and wrap them in Variables.
        # x = Variable(torch.randn(N, D_in))
        # y = Variable(torch.randn(N, D_out), requires_grad=False)

        self.x = Variable(torch.FloatTensor(data))
        self.y = Variable(torch.FloatTensor(labels), requires_grad=False)


        # Use the nn package to define our model as a sequence of layers. nn.Sequential
        # is a Module which contains other Modules, and applies them in sequence to
        # produce its output. Each Linear Module computes output from input using a
        # linear function, and holds internal Variables for its weight and bias.
        self.model = torch.nn.Sequential(
                  torch.nn.Linear(self.D_in, self.H),
                  torch.nn.ReLU(),
                  torch.nn.Linear(self.H, self.D_out),
                )

        # The nn package also contains definitions of popular loss functions; in this
        # case we will use Mean Squared Error (MSE) as our loss function.
        self.loss_fn = torch.nn.MSELoss(size_average=False)
        self.learning_rate = learning_rate

    def train(self):
        for t in range(self.n_iterations):
          # Forward pass: compute predicted y by passing x to the model. Module objects
          # override the __call__ operator so you can call them like functions. When
          # doing so you pass a Variable of input data to the Module and it produces
          # a Variable of output data.
          y_pred = self.model(self.x)
          print(y_pred)

          # Compute and print loss. We pass Variables containing the predicted and true
          # values of y, and the loss function returns a Variable containing the loss.
          loss = self.loss_fn(y_pred, self.y)
          print(t, loss.data[0])
          self.loss_saved.append(loss.data[0])

          # Zero the gradients before running the backward pass.
          self.model.zero_grad()

          # Backward pass: compute gradient of the loss with respect to all the learnable
          # parameters of the model. Internally, the parameters of each Module are stored
          # in Variables with requires_grad=True, so this call will compute gradients for
          # all learnable parameters in the model.
          loss.backward()

          # Update the weights using gradient descent. Each parameter is a Variable, so
          # we can access its data and gradients like we did before.
          for param in self.model.parameters():
            param.data -= self.learning_rate * param.grad.data


    def forward(self, datapoint):
        print(datapoint)
        x = Variable(torch.Tensor(datapoint))
        return self.model(x)
