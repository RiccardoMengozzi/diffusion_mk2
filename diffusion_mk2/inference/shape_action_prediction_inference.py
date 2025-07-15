import torch
import json
import numpy as np
import matplotlib.pyplot as plt
from diffusion_mk2.model.simple_fc.fc_mul import EarlyStopping, FCMul
from diffusion_mk2.dataset.shape_prediction_dataset import DloDataset
from diffusion_mk2.model import normalization_pca


np.set_printoptions(
    precision=4,  # number of decimal places
    suppress=True,  # suppress scientific notation
    linewidth=100,  # characters per line
    threshold=1000,
)  # controls summarization of large arrays


INITIAL_SHAPE = [
    [0.5001327395439148, 0.09760397672653198, 0.7030474543571472],
    [0.5001180768013, 0.08557221293449402, 0.7035804986953735],
    [0.49994075298309326, 0.07116676867008209, 0.703355610370636],
    [0.49985554814338684, 0.057220686227083206, 0.7035009860992432],
    [0.49996402859687805, 0.043037816882133484, 0.7036293148994446],
    [0.5001211762428284, 0.028735987842082977, 0.7034039497375488],
    [0.5001206398010254, 0.01479171309620142, 0.703572690486908],
    [0.5001211762428284, -0.00010601416579447687, 0.7034040093421936],
    [0.5001206994056702, -0.014050287194550037, 0.7035725712776184],
    [0.49994251132011414, -0.028471291065216064, 0.70335453748703],
    [0.49985554814338684, -0.04241557419300079, 0.7035011053085327],
    [0.4999639689922333, -0.056598369032144547, 0.7036292552947998],
    [0.5001189708709717, -0.07089325785636902, 0.7034013867378235],
    [0.5001285672187805, -0.08477339893579483, 0.7035502195358276],
    [0.5000271201133728, -0.097350113093853, 0.7038936018943787],
]

INITIAL_SHAPE2 = [
    [0.5098176598548889, 0.09619959443807602, 0.7029239535331726],
    [0.5087174773216248, 0.08420448750257492, 0.7034635543823242],
    [0.5071823596954346, 0.06986713409423828, 0.7032378315925598],
    [0.5056365132331848, 0.0560012124478817, 0.7033860683441162],
    [0.5039926171302795, 0.04189841449260712, 0.7035178542137146],
    [0.5019183158874512, 0.027723772451281548, 0.7032956480979919],
    [0.49916666746139526, 0.01401852909475565, 0.7034683227539062],
    [0.4956073760986328, -0.0005029179737903178, 0.7032980918884277],
    [0.49251899123191833, -0.014148346148431301, 0.7034692764282227],
    [0.4902229309082031, -0.02841532789170742, 0.703260064125061],
    [0.4898168742656708, -0.04237056151032448, 0.7034040689468384],
    [0.4920097291469574, -0.05634232610464096, 0.7039608955383301],
    [0.4977140724658966, -0.06941630691289902, 0.7037822604179382],
    [0.5048942565917969, -0.08131605386734009, 0.7038280367851257],
    [0.511346161365509, -0.09216979891061783, 0.7041739821434021],
]

TARGET_SHAPE = [
    [0.5017937421798706, 0.06600702553987503, 0.703045129776001],
    [0.5006880164146423, 0.0540233850479126, 0.7035818696022034],
    [0.4991505444049835, 0.039694078266620636, 0.7033557891845703],
    [0.49759531021118164, 0.02583102509379387, 0.7035031318664551],
    [0.49592894315719604, 0.011737468652427197, 0.7036347985267639],
    [0.49384239315986633, -0.002425392623990774, 0.7034091353416443],
    [0.49117371439933777, -0.01613902859389782, 0.703580915927887],
    [0.48770397901535034, -0.030669523403048515, 0.7034105658531189],
    [0.48380863666534424, -0.04410296306014061, 0.7035818099975586],
    [0.47880783677101135, -0.05766928195953369, 0.7033718228340149],
    [0.4736013412475586, -0.07063820213079453, 0.7035200595855713],
    [0.4694138467311859, -0.08416691422462463, 0.7040837407112122],
    [0.4676882028579712, -0.09834947437047958, 0.7038724422454834],
    [0.46772992610931396, -0.11220502853393555, 0.7035718560218811],
    [0.46852532029151917, -0.1247471496462822, 0.703890860080719],
]

TARGET_SHAPE2 = [
    [0.5097389221191406, 0.08994009345769882, 0.7029209733009338],
    [0.508328378200531, 0.07797549664974213, 0.7034686803817749],
    [0.506314754486084, 0.06370138376951218, 0.7032462954521179],
    [0.504052996635437, 0.04993632063269615, 0.7033881545066833],
    [0.5010848641395569, 0.036062393337488174, 0.7035217881202698],
    [0.49696195125579834, 0.022336773574352264, 0.7033019661903381],
    [0.49192124605178833, 0.009278922341763973, 0.7034813761711121],
    [0.48613595962524414, -0.004534755367785692, 0.7033276557922363],
    [0.4808559715747833, -0.017511596903204918, 0.7036956548690796],
    [0.4762074649333954, -0.031188488006591797, 0.7038674354553223],
    [0.4740716814994812, -0.04494089633226395, 0.7035360932350159],
    [0.4771314859390259, -0.058660779148340225, 0.7039633393287659],
    [0.4860174059867859, -0.06975298374891281, 0.7037873864173889],
    [0.4969460368156433, -0.07834946364164352, 0.7038314342498779],
    [0.5071049332618713, -0.08586764335632324, 0.7041720747947693],
]


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return x


class ActionFinderGradient:
    def __init__(
        self, checkpoint_path, device="cpu", lr=1e-3, num_steps=500, verbose=False
    ):
        self.device = device
        self.lr = lr
        self.num_steps = num_steps
        self.verbose = verbose

        # STATE
        state = torch.load(checkpoint_path, map_location=torch.device(self.device))
        self.num_nodes = state["num_points"]
        self.dim_points = state["dim_points"]
        self.hidden_dim = state["hidden_dim"]

        # MODEL
        self.model = FCMul(
            n_pts=self.num_nodes, pts_dim=self.dim_points, hidden_dim=self.hidden_dim
        )
        self.model.load_state_dict(state["model"])
        self.model.eval()

        self.loss_fcn = self.loss_fcn_sum

        print("*" * 50)
        print("Action Finder")
        print(f"num_steps: {self.num_steps}")
        print(f"lr: {self.lr}")
        print("*" * 50)

    def loss_fcn_mean(self, x, y):
        return torch.sum(torch.linalg.norm(x - y, axis=-1), axis=-1)

    def loss_fcn_max(self, x, y):
        return torch.max(torch.linalg.norm(x - y, axis=-1), axis=-1)[0]

    def loss_fcn_sum(self, x, y):
        return torch.sum(torch.linalg.norm(x - y, axis=-1), axis=-1)

    def sample_init_action_given_idx(self, dlo_0, dlo_1, idx):
        """
        Sample a displacement that moves the node towards the target
        """
        np.random.seed()  # to reeseed allowing different initial displacements for every process

        # DISPLACEMENT
        node_pos = dlo_0[idx]
        node_target = dlo_1[idx]
        direction = node_target - node_pos

        f = np.random.normal(0.5, 0.2, size=2)
        disp = f * direction[:2]

        max = np.max(np.abs(disp))
        if max > 0.1:
            disp = disp / max * 0.1

        # THETA
        # dir_init = dlo_0[idx + 1] - dlo_0[idx]
        # dir_init = dir_init / np.linalg.norm(dir_init)
        # dir_target = dlo_1[idx + 1] - dlo_1[idx]
        # dir_target = dir_target / np.linalg.norm(dir_target)
        # angle_target = np.arctan2(dir_target[1], dir_target[0])
        # angle_init = np.arctan2(dir_init[1], dir_init[0])

        # theta = np.random.normal(0.0, 0.2)
        theta = 0.0
        return disp, theta

    def run_from_file(self, file_path):
        dlo_0, dlo_1, action_gt, _ = self.sample_obj.load_sample(file_path)
        return self.run(dlo_0, dlo_1, action_gt=action_gt)

    def run(self, dlo_0, dlo_1, action_gt=None, idx=None):
        if idx is None and action_gt is None:
            raise ValueError("Either idx or action_gt must be provided")

        if action_gt is None:
            action_gt = [idx, 0, 0, 0]

        action_idx = np.array(action_gt[0])
        action_pos = np.array(action_gt[1:3])
        action_theta = np.array(action_gt[3])
        cs0, csR = normalization_pca.compute_normalize_factors(dlo_0)
        dlo_0_n = normalization_pca.normalize_pca(dlo_0, cs0, csR)
        dlo_1_n = normalization_pca.normalize_pca(dlo_1, cs0, csR)
        action_idx_n = normalization_pca.normalize_min_max(
            action_idx, 0, self.num_nodes - 1
        )
        action_pos_n = normalization_pca.normalize_pca(
            action_pos, cs0, csR, rotation_only=True
        )
        action_theta_n = action_theta

        action_gt_n = np.array(
            [action_idx_n, action_pos_n[0], action_pos_n[1], action_theta_n]
        )

        # to tensor
        dlo_0_tn = torch.from_numpy(dlo_0_n.copy()).float().unsqueeze_(0)
        dlo_1_tn = torch.from_numpy(dlo_1_n.copy()).float().unsqueeze_(0)
        action_gt_tn = torch.from_numpy(action_gt_n.copy()).float().unsqueeze_(0)

        # init action
        idx = int(action_gt[0])
        disp, theta = self.sample_init_action_given_idx(dlo_0, dlo_1, idx)

        action_init = np.array([idx, disp[0], disp[1], theta])
        disp_n = normalization_pca.normalize_pca(disp, cs0, csR, rotation_only=True)
        action_init_n = np.array([action_idx_n, disp_n[0], disp_n[1], action_theta_n])

        action_init_tn = torch.from_numpy(action_init_n).float().unsqueeze_(0)

        # ******************************************************************************************

        # find action
        opt_log = self.find_action(dlo_0_tn, dlo_1_tn, action_init_tn)

        # ******************************************************************************************

        # best action
        best_loss_idx = np.argmin([opt_log[k]["loss"] for k in opt_log.keys()])
        best_action_n = opt_log[best_loss_idx]["action"]
        best_action_tn = torch.from_numpy(best_action_n).float().unsqueeze_(0)

        best_action_idx = best_action_n[0]
        best_action_pos = best_action_n[1:3]
        best_action_theta = best_action_n[3]

        best_action_idx_dn = normalization_pca.denormalize_min_max(
            best_action_idx, 0, self.num_nodes - 1
        )
        best_action_pos_dn = normalization_pca.denormalize_pca(
            best_action_pos, cs0, csR, rotation_only=True
        )
        best_action_theta_dn = best_action_theta

        best_action = np.array(
            [
                best_action_idx_dn,
                best_action_pos_dn[0],
                best_action_pos_dn[1],
                best_action_theta_dn,
            ]
        )

        print("best_action_n", best_action_n)
        print("best_action", best_action)

        print("action_init_n", action_init_n)
        print("action_init", action_init)

        print("action_gt_n", action_gt_n)
        print("action_gt", action_gt)

        # predictions
        pred = self.model(dlo_0_tn, best_action_tn)
        pred_init = self.model(dlo_0_tn, action_init_tn)
        pred_gt = self.model(dlo_0_tn, action_gt_tn)

        # denormalize everything
        pred = normalization_pca.denormalize_pca(to_numpy(pred.squeeze()), cs0, csR)
        pred_init = normalization_pca.denormalize_pca(
            to_numpy(pred_init.squeeze()), cs0, csR
        )
        pred_gt = normalization_pca.denormalize_pca(
            to_numpy(pred_gt.squeeze()), cs0, csR
        )

        # loss
        loss_action_gt = self.loss_fcn(
            torch.from_numpy(pred_gt.copy()), torch.from_numpy(dlo_1.copy())
        ).item()
        loss_action_init = self.loss_fcn(
            torch.from_numpy(pred_init.copy()), torch.from_numpy(dlo_1.copy())
        ).item()
        loss_pred = self.loss_fcn(
            torch.from_numpy(pred.copy()), torch.from_numpy(dlo_1.copy())
        ).item()

        output_log = {
            "dlo_0": dlo_0,
            "dlo_1": dlo_1,
            "pred": pred,
            "pred_init": pred_init,
            "pred_gt": pred_gt,
            "opt_log": opt_log,
            "best_action": best_action,
            "best_action_normalized": best_action_n,
            "init_action": action_init,
            "init_action_normalized": action_init_n,
            "gt_action": action_gt,
            "gt_action_normalized": action_gt_n,
            "loss_action_gt": loss_action_gt,
            "loss_action_init": loss_action_init,
            "loss_pred": loss_pred,
        }

        return output_log

    def run_batch(self, dlo_0, dlo_1, indices):
        cs0, csR = normalization_pca.compute_normalize_factors(dlo_0)
        dlo_0_n = normalization_pca.normalize_pca(dlo_0, cs0, csR)
        dlo_1_n = normalization_pca.normalize_pca(dlo_1, cs0, csR)

        # to tensor
        dlo_0_tn = torch.from_numpy(dlo_0_n.copy()).float().unsqueeze_(0)
        dlo_1_tn = torch.from_numpy(dlo_1_n.copy()).float().unsqueeze_(0)

        # init action

        actions_init = []
        for idx in indices:
            disp, theta = self.sample_init_action_given_idx(dlo_0, dlo_1, idx)
            action_init = np.array([idx, disp[0], disp[1], theta])

            action_init_idx = action_init[0]
            action_init_pos = action_init[1:3]
            action_init_theta = action_init[3]

            action_init_idx_n = normalization_pca.normalize_min_max(
                action_init_idx, 0, self.num_nodes - 1
            )
            action_init_pos_n = normalization_pca.normalize_pca(
                action_init_pos, cs0, csR, rotation_only=True
            )
            action_init_theta_n = action_init_theta

            action_init_n = np.array(
                [
                    action_init_idx_n,
                    action_init_pos_n[0],
                    action_init_pos_n[1],
                    action_init_theta_n,
                ]
            )
            actions_init.append(action_init_n)

        actions_init = np.array(actions_init)
        action_init_tn = torch.from_numpy(actions_init).float()

        # ******************************************************************************************
        dlo_0_tn_batch = dlo_0_tn.tile(len(indices), 1, 1)
        dlo_1_tn_batch = dlo_1_tn.tile(len(indices), 1, 1)

        # find action
        opt_log = self.find_action_batch(
            dlo_0_tn_batch, dlo_1_tn_batch, action_init_tn
        )

        # ******************************************************************************************
        losses = np.array([opt_log[k]["losses"] for k in opt_log.keys()])
        output_batch_log = {}
        for i in range(len(indices)):
            opt_log_losses = losses[i, :]
            opt_log_action = np.array([opt_log[k]["action"][i] for k in opt_log.keys()])

            # best action
            best_loss_idx = np.argmin(opt_log_losses)
            best_loss = opt_log_losses[best_loss_idx]

            best_action_n = opt_log[best_loss_idx]["action"][i]
            
            best_action_idx = best_action_n[0]
            best_action_pos = best_action_n[1:3]
            best_action_theta = best_action_n[3]

            best_action_idx_dn = normalization_pca.denormalize_min_max(
                best_action_idx, 0, self.num_nodes - 1
            )
            best_action_pos_dn = normalization_pca.denormalize_pca(
                best_action_pos, cs0, csR, rotation_only=True
            )
            best_action_theta_dn = best_action_theta

            best_action = np.array(
                [
                    best_action_idx_dn,
                    best_action_pos_dn[0],
                    best_action_pos_dn[1],
                    best_action_theta_dn,
                ]
            )

            print("best action {i} - {action}".format(i=i, action=best_action))
            best_action_tn = torch.from_numpy(best_action_n).float()

            # predictions
            pred = self.model(
                dlo_0_tn_batch[i].unsqueeze(0),
                best_action_tn.unsqueeze(0),
            )
            pred_init = self.model(
                dlo_0_tn_batch[i].unsqueeze(0),
                action_init_tn[i].unsqueeze(0),
            )

            # denormalize everything
            pred = normalization_pca.denormalize_pca(
                to_numpy(pred.squeeze()), cs0, csR
            )
            pred_init = normalization_pca.denormalize_pca(
                to_numpy(pred_init.squeeze()), cs0, csR
            )

            # loss
            loss_action_init = self.loss_fcn(
                torch.from_numpy(pred_init.copy()), torch.from_numpy(dlo_1.copy())
            ).item()
            loss_pred = self.loss_fcn(
                torch.from_numpy(pred.copy()), torch.from_numpy(dlo_1.copy())
            ).item()
            loss_init_pred = self.loss_fcn(
                torch.from_numpy(dlo_0.copy()), torch.from_numpy(pred.copy())
            ).item()

            output_batch_log[indices[i]] = {
                "dlo_0": dlo_0,
                "dlo_1": dlo_1,
                "pred": pred,
                "pred_init": pred_init,
                "opt_log": {"loss": opt_log_losses, "action": opt_log_action},
                "best_action": best_action,
                "best_action_normalized": best_action_n,
                "init_action": action_init,
                "init_action_normalized": action_init_n,
                "loss_action_init": loss_action_init,
                "loss_pred": loss_pred,
                "loss_init_pred": loss_init_pred,
                "best_loss": best_loss,
            }

        return output_batch_log

    def find_action(self, dlo_0, dlo_1, action, print_every=10):
        idx = action[:, 0]
        trainable_params = torch.nn.Parameter(action[:, 1:].clone(), requires_grad=True)
        optimizer = torch.optim.Adam([trainable_params], lr=self.lr)

        patience_scheduler = 20
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=20, factor=0.1
        )

        tanh = torch.nn.Tanh()

        opt_log_dict = {}
        early_stopping = EarlyStopping(
            patience=patience_scheduler * 3, min_epochs=self.num_steps // 5
        )
        for step in range(self.num_steps):
            optimizer.zero_grad()

            # trainable_params2 = tanh(trainable_params)
            trainable_params2 = trainable_params

            action_tn = torch.cat([idx.unsqueeze(0), trainable_params2], dim=1)
            pred = self.model(dlo_0, action_tn)

            loss = self.loss_fcn(pred, dlo_1)

            action_save = action_tn.clone().squeeze().detach().numpy()
            opt_log_dict[step] = {
                "loss": loss.item(),
                "action": action_save,
            }

            if self.verbose:
                if step % print_every == 0:
                    print(f"step: {step}, loss: {loss.item()}, action: {action_save}")

            # backward
            loss.backward()
            optimizer.step()

            scheduler.step(loss.item())

            # EARLY STOPPING
            if early_stopping.stop(loss.item()):
                print("Early Stopping!")
                break

        return opt_log_dict

    def find_action_batch(self, dlos_0, dlos_1, actions, print_every=10):
        indices = actions[:, 0].unsqueeze(1)

        trainable_params = torch.nn.Parameter(
            actions[:, 1:].clone(), requires_grad=True
        )
        optimizer = torch.optim.Adam([trainable_params], lr=self.lr)

        # patience_scheduler = 20
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.1)
        tanh = torch.nn.Tanh()

        opt_log_dict = {}
        # early_stopping = EarlyStopping(patience=patience_scheduler * 3, min_epochs=self.num_steps // 5)
        for step in range(self.num_steps):
            optimizer.zero_grad()

            actions_tn = torch.cat([indices, tanh(trainable_params)], dim=1)

            pred = self.model(dlos_0, actions_tn)

            loss = self.loss_fcn(pred, dlos_1)

            opt_log_dict[step] = {
                "losses": to_numpy(loss),
                "action": actions_tn.clone().squeeze().detach().numpy(),
            }

            if step % print_every == 0 and self.verbose:
                print(f"step: {step}, loss: {np.mean(to_numpy(loss))}")

            # backward
            loss.backward(gradient=torch.ones_like(loss))
            optimizer.step()

            # mean_loss = torch.mean(loss)
            # scheduler.step(mean_loss.item())

            # EARLY STOPPING
            # if early_stopping.stop(mean_loss.item()):
            #    print("Early Stopping!")
            #    break

        return opt_log_dict

    def plot_log(self, log_dict):
        opt_log = log_dict["opt_log"]
        pred = log_dict["pred"]
        # pred_init = log_dict["pred_init"]
        # pred_gt = log_dict["pred_gt"]
        dlo_0 = log_dict["dlo_0"]
        dlo_1 = log_dict["dlo_1"]

        # loss_gt = log_dict["loss_action_gt"]
        # loss_init = log_dict["loss_action_init"]
        # gt_action_normalized = log_dict["gt_action_normalized"]
        # init_action_normalized = log_dict["init_action_normalized"]
        best_action = log_dict["best_action"]
        true_action = log_dict["true_action"]

        x_axis = np.array(list(opt_log.keys()))

        # LOG DICT
        loss_list = np.array([opt_log[k]["loss"] for k in opt_log.keys()])
        action_x_list = np.array([opt_log[k]["action"][1] for k in opt_log.keys()])
        action_y_list = np.array([opt_log[k]["action"][2] for k in opt_log.keys()])
        action_theta_list = np.array([opt_log[k]["action"][3] for k in opt_log.keys()])

        # best loss
        best_loss_idx = np.argmin(loss_list)
        best_x = action_x_list[best_loss_idx]
        best_y = action_y_list[best_loss_idx]
        best_theta = action_theta_list[best_loss_idx]

        # ACTION GT
        # gt_x_list = gt_action_normalized[1] * np.ones_like(x_axis)
        # gt_y_list = gt_action_normalized[2] * np.ones_like(x_axis)
        # gt_theta_list = gt_action_normalized[3] * np.ones_like(x_axis)
        # init_x_list = init_action_normalized[1] * np.ones_like(x_axis)
        # init_y_list = init_action_normalized[2] * np.ones_like(x_axis)
        # init_theta_list = init_action_normalized[3] * np.ones_like(x_axis)

        # EDGE ACTION
        # idx = int(best_action[0])
        # target_pos = self.sample_obj.compute_edge_target_position(
        #     dlo_0, idx, best_action[1], best_action[2], best_action[3]
        # )
        # target_pos = np.array(target_pos)

        fig = plt.figure(figsize=(10, 6))
        plt.plot(dlo_0[:, 0], dlo_0[:, 1], "o-", label="dlo_0")
        plt.plot(dlo_1[:, 0], dlo_1[:, 1], "v-", label="dlo_1", zorder=100)
        # plt.plot(pred_gt[:, 0], pred_gt[:, 1], "o-", label="pred_gt")
        # plt.plot(pred_init[:, 0], pred_init[:, 1], "o-", label="pred_init")
        plt.plot(pred[:, 0], pred[:, 1], "o-", label="pred")
        # plt.plot(
        #     target_pos[:, 0], target_pos[:, 1], "o-", label="target_pos", color="cyan"
        # )
        # Plot the best action as an arrow
        idx = int(best_action[0])
        start_pos = dlo_0[idx, :2]
        end_pos = start_pos + best_action[1:3]
        plt.arrow(
            start_pos[0],
            start_pos[1],
            best_action[1],
            best_action[2],
            head_width=0.005,
            head_length=0.01,
            fc="red",
            ec="red",
            length_includes_head=True,
            label="best_action",
        )
        plt.scatter(
            end_pos[0], end_pos[1], marker="*", s=120, color="red", label="action_end"
        )

        # Plot the true action as an arrow
        idx = int(true_action[0])
        true_start_pos = dlo_0[idx, :2]
        true_end_pos = true_start_pos + true_action[1:3]
        plt.arrow(
            true_start_pos[0],
            true_start_pos[1],
            true_action[1],
            true_action[2],
            head_width=0.005,
            head_length=0.01,
            fc="green",
            ec="green",
            length_includes_head=True,
            label="true_action",
        )
        plt.scatter(
            true_end_pos[0], true_end_pos[1], marker="*", s=120, color="green", label="true_action_end"
        )


        plt.scatter(dlo_0[0, 0], dlo_0[0, 1], marker="X", s=100)
        plt.scatter(dlo_1[0, 0], dlo_1[0, 1], marker="X", s=100)
        plt.scatter(pred[0, 0], pred[0, 1], marker="X", s=100)

        plt.axis("equal")
        plt.legend()

        fig, axs = plt.subplots(1, 4, figsize=(15, 6))
        axs[0].plot(x_axis, action_x_list, label=f"trained ({best_x:.2f})")
        # axs[0].plot(x_axis, gt_x_list, label="gt")
        # axs[0].plot(x_axis, init_x_list, label="init")
        axs[0].legend()
        axs[0].set_ylim([-1, 1])
        axs[0].set_title("disp_x")

        axs[1].plot(x_axis, action_y_list, label=f"trained ({best_y:.2f})")
        # axs[1].plot(x_axis, gt_y_list, label="gt")
        # axs[1].plot(x_axis, init_y_list, label="init")
        axs[1].legend()
        axs[1].set_ylim([-1, 1])
        axs[1].set_title("disp_y")

        axs[2].plot(x_axis, action_theta_list, label=f"trained ({best_theta:.2f})")
        # axs[2].plot(x_axis, gt_theta_list, label="gt")
        # axs[2].plot(x_axis, init_theta_list, label="init")
        axs[2].legend()
        axs[2].set_ylim([-1, 1])
        axs[2].set_title("theta")

        axs[3].plot(x_axis, loss_list)
        # axs[3].plot(x_axis, loss_gt * np.ones_like(x_axis), label="gt")
        # axs[3].plot(x_axis, loss_init * np.ones_like(x_axis), label="init")
        axs[3].set_title("loss")
        axs[3].legend()

        plt.tight_layout()
        plt.show()

def extract_shapes(dataset_path):
    init_shapes = []
    final_shapes = []
    actions = []
    with open(dataset_path, 'r') as f:
        is_first_obs_of_episode = True
        for line in f:
            try:
                data = json.loads(line.strip())
                if data.get("type") == "data":
                    if is_first_obs_of_episode:
                        init_shapes.append(np.array(data["obs_dlo"]))
                        final_shapes.append(np.array(data["obs_target"]))
                        actions.append(np.array(data["action_from_grasp_to_release"]))
                        is_first_obs_of_episode = False
                    else:
                        continue

                elif data.get("type") == "episode_end":
                    is_first_obs_of_episode = True

            except json.JSONDecodeError as e:
                print(f"Warning: Skipping invalid JSON line: {e}")
                continue


    # Convert to numpy arrays
    init_shapes = np.array(init_shapes, dtype=np.float32)
    final_shapes = np.array(final_shapes, dtype=np.float32)
    actions = np.array(actions, dtype=np.float32)

    return {
        "init": init_shapes,
        "target": final_shapes,
        "action": actions,
    }


if __name__ == "__main__":
    dataset_path = "/home/mengo/Research/LLM_DOM/diffusion_mk2/json_data/test.jsonl"
    checkpoint_path = "/home/mengo/Research/LLM_DOM/diffusion_mk2/weights/chkp_90000.pt"
    action_finder = ActionFinderGradient(
        checkpoint_path=checkpoint_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
        lr=1e-3,
        num_steps=5000,
        verbose=True,
    )
    data = extract_shapes(dataset_path)
    for init, target, action in zip(data["init"], data["target"], data["action"]):

        log = action_finder.run(
            dlo_0=init[:, :2],
            dlo_1=target[:, :2],
            idx=action[0],
        )
        log["true_action"] = action

        # indices = list(range(len(INITIAL_SHAPE)))
        # log = action_finder.run_batch(
        #     dlo_0=np.array(INITIAL_SHAPE)[:, :2],
        #     dlo_1=np.array(TARGET_SHAPE)[:, :2],
        #     indices=indices,
        # )

        action_finder.plot_log(log)
