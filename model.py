import torch.nn as nn
import torch.nn.functional as F
import torchbnn as bnn


class PrefNet(nn.Module):
    def __init__(self, n_input):
        super(PrefNet, self).__init__()
        self.fc1 = bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=n_input, out_features=10)
        self.fc2 = bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=10, out_features=10)
        self.fc3 = bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=10, out_features=1)

    def forward_once(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)

        return x

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2


class PrefNet_MC_dropout(nn.Module):
    def __init__(self, n_input):
        super(PrefNet_MC_dropout, self).__init__()
        self.fc1 = nn.Linear(n_input, 100)
        self.fc2 = nn.Linear(100, 30)
        self.fc3 = nn.Linear(30, 1)
        self.dropout = nn.Dropout(0.5)

    def forward_once(self, x):
        x = self.fc1(x)
        x = self.dropout(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)

        return x

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2
