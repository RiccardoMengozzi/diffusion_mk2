import os, torch, json
import numpy as np
from tqdm import tqdm

from diffusion_mk2.dataset.shape_prediction_dataset import DloDataset
from diffusion_mk2.model.simple_fc.fc_mul import FCMul  


np.set_printoptions(precision=4,    # number of decimal places
                    suppress=True,  # suppress scientific notation
                    linewidth=100,  # characters per line
                    threshold=1000) # controls summarization of large arrays


PROJECT_DIR = os.path.dirname(__file__)
LOG_INTERVAL = 50
CHECKPOINT_INTERVAL = 30000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = dict(
    batch_size=256,
    epochs=300000,
    lr=5e-4,
    hidden_dim=256,
    dim_points=2,
    num_points=15,
    dataset_path=os.path.join(PROJECT_DIR, "zarr_data", "shape_prediction.zarr.zip"),
)


###################################
print(f"Using device: {DEVICE}")
print("*" * 20)
for k, v in config.items():
    print(f"\t{k}: {v}")
print("*" * 20)
###################################

# DATASETS
train_data = DloDataset(config["dataset_path"], num_points=config["num_points"])

train_loader = torch.utils.data.DataLoader(train_data, batch_size=config["batch_size"], shuffle=True, num_workers=0)


print("Train set size: {}".format(len(train_data)))
print("")

# MODEL
model = FCMul(n_pts=config["num_points"], pts_dim=config["dim_points"], hidden_dim=config["hidden_dim"])
model = model.to(DEVICE)
loss_fcn = lambda x, y: torch.mean(torch.linalg.norm(x - y, axis=-1))
optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])


best_loss = np.inf
global_step = 0
with tqdm(range(config["epochs"]), desc="Epoch") as epoch_bar:
    for epoch in epoch_bar:
        train_epoch_loss = 0.0
        val_epoch_loss = 0.0

        ##############################
        # TRAIN
        model.train()
        for i, data in enumerate(train_loader):
            optimizer.zero_grad()

            dlo_0, dlo_1, action = data
            dlo_0, dlo_1, action = dlo_0.to(DEVICE), dlo_1.to(DEVICE), action.to(DEVICE)

            pred = model(dlo_0, action)

            loss = loss_fcn(pred, dlo_1)
            loss.backward()
            optimizer.step()
            train_epoch_loss += loss.item()
            global_step += 1

        train_epoch_loss /= len(train_loader)
        

        if epoch % CHECKPOINT_INTERVAL == 0:
            # Save model checkpoint
            checkpoint_path = os.path.join(PROJECT_DIR, "checkpoints", "shape_prediction", f"chkp_{epoch}.pt")
            state = dict(config)
            state["model"] = model.state_dict()
            torch.save(state, checkpoint_path)
            print(f"Saved checkpoint at {checkpoint_path}")

        epoch_bar.set_postfix(train_loss=train_epoch_loss)
        

## save
model_path = os.path.join(PROJECT_DIR, "checkpoints", "shape_prediction", "final_model.pt")
state = dict(config)
state["model"] = model.state_dict()
torch.save(state, model_path)
print(f"Final model saved at {model_path}")