import torch
import torch.nn as nn


class FCMul(nn.Module):
    def __init__(self, n_pts, pts_dim, hidden_dim=256):
        super(FCMul, self).__init__()

        self.n_pts = n_pts
        self.pts_dim = pts_dim
        self.N = hidden_dim

        a = nn.ReLU()  # nn.Tanh()
        tanh = nn.Tanh()

        self.dlo = nn.Sequential(
            nn.Flatten(-2),
            nn.Linear(self.n_pts * self.pts_dim, self.N),
            a,
            nn.Linear(self.N, self.N),
            a,
            nn.Linear(self.N, self.N),
            a,
        )

        self.act = nn.Sequential(
            nn.Linear(4, self.N),
            a,
            nn.Linear(self.N, self.N),
            a,
        )

        self.state_action = nn.Sequential(
            nn.Linear(2 * self.N, self.N),
            a,
        )

        self.pred = nn.ModuleList(
            [
                nn.Linear(self.N, self.N),
                a,
                nn.Linear(self.N, self.N),
                a,
                nn.Linear(self.N, self.n_pts * self.pts_dim),
                nn.Unflatten(-1, (self.n_pts, self.pts_dim)),
            ]
        )

    def forward(self, dlo, action):
        x_s = self.dlo(dlo)
        x_a = self.act(action)

        x = torch.concat([x_s, x_a], dim=-1)

        x = self.state_action(x)

        for l in self.pred:
            x = l(x)

        x += dlo

        return x


class EarlyStopping:
    def __init__(self, patience=50, min_epochs=100):
        self.patience = patience
        self.min_epochs = min_epochs

        self.no_improve = 0
        self.min_loss = 100000

    def stop(self, loss):
        if loss < self.min_loss:
            self.no_improve = 0
            self.min_loss = loss
        else:
            self.no_improve += 1

        if self.no_improve >= self.patience and self.min_epochs <= self.min_epochs:
            return True
        else:
            return False
        

if __name__ == "__main__":
    ### TEST ###

    model = FCMul(n_pts=15, pts_dim=2, hidden_dim=256)
    dlo = torch.randn(10, 15, 2)  # (batch_size, n_pts, pts_dim)
    action = torch.randn(10, 4)  # (batch_size, action_dim
    output = model(dlo, action)
    print(output.shape)  # Should be (10, 15, 2)