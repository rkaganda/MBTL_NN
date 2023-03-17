import torch
import torch.nn as nn


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)


class Model(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, num_layers):
        super(Model, self).__init__()

        # init params
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # layers
        # rn
        self.rnn = nn.GRU(input_size, hidden_dim, num_layers, batch_first=True)
        # fc layer
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, x):
        batch_size = x.size(0)

        # init for first input
        hidden = self.init_hidden(batch_size)

        # pass input and hidden state to run
        out, hidden = self.rnn(x, hidden)

        # reshape for fc
        out = out.contiguous().view(-1, self.hidden_dim)
        out = self.fc(out)

        return out, hidden

    def init_hidden(self, batch_size):
        # create zero for first pass hidden
        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)

        return hidden


def setup_model(input_size, actions_size, learning_rate):
    model = Model(
        input_size=input_size,
        output_size=actions_size,
        hidden_dim=input_size,
        num_layers=1
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    return model, optimizer






