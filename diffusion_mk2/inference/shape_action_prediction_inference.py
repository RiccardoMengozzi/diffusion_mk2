import torch
import json
import numpy as np
import matplotlib.pyplot as plt
from diffusion_mk2.model.simple_fc.fc_mul import EarlyStopping, FCMul
from diffusion_mk2.dataset.shape_prediction_dataset import DloDataset
from diffusion_mk2.model import normalization_pca
from scipy.optimize import minimize, differential_evolution
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

np.set_printoptions(
    precision=4,
    suppress=True,
    linewidth=100,
    threshold=1000,
)

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

        # Load model state
        state = torch.load(checkpoint_path, map_location=torch.device(self.device), weights_only=False)
        self.num_nodes = state["num_points"]
        self.dim_points = state["dim_points"]
        self.hidden_dim = state["hidden_dim"]
        self.stats = state["stats"]

        # Initialize model
        self.model = FCMul(
            n_pts=self.num_nodes, pts_dim=self.dim_points, hidden_dim=self.hidden_dim
        )
        self.model.load_state_dict(state["model"])
        self.model.eval()

        # Set loss function
        self.loss_fcn = self.loss_fcn_sum
        
        # Optimization parameters
        self.num_restarts = 5  # Multiple random restarts
        self.use_ensemble = True  # Use ensemble of methods
        self.use_coarse_to_fine = True  # Coarse-to-fine optimization
        
        print("*" * 50)
        print("Improved Action Finder")
        print(f"num_steps: {self.num_steps}")
        print(f"lr: {self.lr}")
        print(f"num_restarts: {self.num_restarts}")
        print(f"use_ensemble: {self.use_ensemble}")
        print("*" * 50)

    def loss_fcn_mean(self, x, y):
        return torch.mean(torch.linalg.norm(x - y, axis=-1), axis=-1)

    def loss_fcn_max(self, x, y):
        return torch.max(torch.linalg.norm(x - y, axis=-1), axis=-1)[0]

    def loss_fcn_sum(self, x, y):
        return torch.sum(torch.linalg.norm(x - y, axis=-1), axis=-1)
    
    def loss_fcn_huber(self, x, y, delta=0.01):
        """Huber loss - less sensitive to outliers"""
        diff = torch.linalg.norm(x - y, axis=-1)
        mask = diff < delta
        loss = torch.where(mask, 0.5 * diff**2, delta * (diff - 0.5 * delta))
        return torch.sum(loss, axis=-1)
    
    def normalize_action(self, action, cs0, csR):
        action_idx = np.array(action[0])
        action_pos = np.array(action[1:3])
        action_theta = np.array(action[3])

        action_idx_n = normalization_pca.normalize_min_max(
            action_idx,
            self.stats["action"]["min"][0],
            self.stats["action"]["max"][0] 
        )
        action_pos_n = normalization_pca.normalize_pca(
            action_pos, cs0, csR, rotation_only=True
        )
        action_theta_n = normalization_pca.normalize_min_max(
            action_theta,
            self.stats["action"]["min"][-1],
            self.stats["action"]["max"][-1],
        )

        action_n = np.array(
            [action_idx_n, action_pos_n[0], action_pos_n[1], action_theta_n]
        )
        return action_n

    def denormalize_action(self, action_n, cs0, csR):
        action_n_idx = action_n[0]
        action_n_pos = action_n[1:3]
        action_n_theta = action_n[3]

        action_idx_dn = normalization_pca.denormalize_min_max(
            action_n_idx,
            self.stats["action"]["min"][0],
            self.stats["action"]["max"][0]
        )
        action_pos_dn = normalization_pca.denormalize_pca(
            action_n_pos, cs0, csR, rotation_only=True
        )
        action_theta_dn = normalization_pca.denormalize_min_max(
            action_n_theta,
            self.stats["action"]["min"][-1],
            self.stats["action"]["max"][-1],
        )

        action_dn = np.array([
            action_idx_dn,
            action_pos_dn[0],
            action_pos_dn[1],
            action_theta_dn,
        ])
        return action_dn

    def smart_action_initialization(self, dlo_0, dlo_1, idx, num_candidates=10):
        """Generate multiple smart initial action candidates"""
        candidates = []
        
        # Method 1: Direction-based initialization (original but improved)
        for _ in range(num_candidates // 3):
            node_pos = dlo_0[idx]
            node_target = dlo_1[idx]
            direction = node_target - node_pos
            
            # Add some noise and different scaling factors
            noise_scale = np.random.uniform(0.1, 0.3)
            direction_noise = np.random.normal(0, noise_scale, 2)
            scale_factor = np.random.uniform(0.3, 0.8)
            
            disp = scale_factor * direction[:2] + direction_noise
            
            # Clamp displacement
            max_disp = 0.15
            disp_norm = np.linalg.norm(disp)
            if disp_norm > max_disp:
                disp = disp / disp_norm * max_disp
            
            # Compute theta
            theta = self.compute_target_theta(dlo_0, dlo_1, idx)
            theta += np.random.normal(0, 0.1)  # Add some noise
            
            candidates.append([idx, disp[0], disp[1], theta])
        
        # Method 2: Neighbor-aware initialization
        for _ in range(num_candidates // 3):
            disp, theta = self.neighbor_aware_init(dlo_0, dlo_1, idx)
            candidates.append([idx, disp[0], disp[1], theta])
        
        # Method 3: Random exploration around target
        for _ in range(num_candidates - 2 * (num_candidates // 3)):
            node_target = dlo_1[idx]
            node_pos = dlo_0[idx]
            
            # Random displacement around the target direction
            target_disp = node_target[:2] - node_pos[:2]
            random_disp = np.random.normal(0, 0.05, 2)
            disp = target_disp * np.random.uniform(0.2, 0.9) + random_disp
            
            # Clamp
            max_disp = 0.12
            disp_norm = np.linalg.norm(disp)
            if disp_norm > max_disp:
                disp = disp / disp_norm * max_disp
            
            theta = np.random.uniform(-0.5, 0.5)
            candidates.append([idx, disp[0], disp[1], theta])
        
        return candidates

    def neighbor_aware_init(self, dlo_0, dlo_1, idx):
        """Initialize action considering neighboring nodes"""
        # Get neighboring nodes
        neighbors = []
        if idx > 0:
            neighbors.append(idx - 1)
        if idx < len(dlo_0) - 1:
            neighbors.append(idx + 1)
        
        # Compute average displacement of neighbors
        avg_neighbor_disp = np.zeros(2)
        for neighbor_idx in neighbors:
            neighbor_disp = dlo_1[neighbor_idx][:2] - dlo_0[neighbor_idx][:2]
            avg_neighbor_disp += neighbor_disp
        
        if len(neighbors) > 0:
            avg_neighbor_disp /= len(neighbors)
        
        # Current node displacement
        node_disp = dlo_1[idx][:2] - dlo_0[idx][:2]
        
        # Blend between node displacement and neighbor average
        blend_factor = 0.7
        disp = blend_factor * node_disp + (1 - blend_factor) * avg_neighbor_disp
        
        # Add some randomness
        disp += np.random.normal(0, 0.02, 2)
        
        # Clamp
        max_disp = 0.12
        disp_norm = np.linalg.norm(disp)
        if disp_norm > max_disp:
            disp = disp / disp_norm * max_disp
        
        theta = self.compute_target_theta(dlo_0, dlo_1, idx)
        theta += np.random.normal(0, 0.05)
        
        return disp, theta

    def compute_target_theta(self, dlo_0, dlo_1, idx):
        """Compute target theta based on local shape changes"""
        try:
            # Get adjacent node
            if idx == self.num_nodes - 1:
                idx2 = idx - 1
            else:
                idx2 = idx + 1
            
            # Initial and target directions
            dir_init = dlo_0[idx2] - dlo_0[idx]
            dir_target = dlo_1[idx2] - dlo_1[idx]
            
            # Normalize
            dir_init_norm = np.linalg.norm(dir_init)
            dir_target_norm = np.linalg.norm(dir_target)
            
            if dir_init_norm > 1e-6 and dir_target_norm > 1e-6:
                dir_init = dir_init / dir_init_norm
                dir_target = dir_target / dir_target_norm
                
                angle_init = np.arctan2(dir_init[1], dir_init[0])
                angle_target = np.arctan2(dir_target[1], dir_target[0])
                
                theta = angle_target - angle_init
                
                # Wrap to [-pi, pi]
                theta = np.arctan2(np.sin(theta), np.cos(theta))
            else:
                theta = 0.0
                
        except:
            theta = 0.0
            
        return theta

    def evaluate_action_batch(self, dlo_0_tn, dlo_1_tn, actions_tn):
        """Evaluate multiple actions in batch"""
        with torch.no_grad():
            preds = self.model(dlo_0_tn.repeat(len(actions_tn), 1, 1), actions_tn)
            losses = self.loss_fcn(preds, dlo_1_tn.repeat(len(actions_tn), 1, 1))
        return losses.cpu().numpy()

    def scipy_optimization(self, dlo_0_tn, dlo_1_tn, idx, bounds):
        """Use scipy optimization as an alternative method"""
        def objective(params):
            # params = [disp_x, disp_y, theta]
            action = torch.tensor([[idx, params[0], params[1], params[2]]], dtype=torch.float32)
            with torch.no_grad():
                pred = self.model(dlo_0_tn, action)
                loss = self.loss_fcn(pred, dlo_1_tn)
            return loss.item()
        
        # Set bounds for displacement and theta
        param_bounds = [
            (-0.15, 0.15),  # disp_x
            (-0.15, 0.15),  # disp_y  
            (-1.0, 1.0),    # theta
        ]
        
        best_result = None
        best_loss = float('inf')
        
        # Try multiple random initializations
        for _ in range(3):
            x0 = [
                np.random.uniform(-0.05, 0.05),
                np.random.uniform(-0.05, 0.05), 
                np.random.uniform(-0.2, 0.2)
            ]
            
            try:
                result = minimize(
                    objective, 
                    x0, 
                    method='L-BFGS-B',
                    bounds=param_bounds,
                    options={'maxiter': 100}
                )
                
                if result.fun < best_loss:
                    best_loss = result.fun
                    best_result = result
                    
            except:
                continue
        
        if best_result is not None:
            return np.array([idx, best_result.x[0], best_result.x[1], best_result.x[2]]), best_loss
        else:
            return None, float('inf')

    def find_action_ensemble(self, dlo_0, dlo_1, action_init):
        """Use ensemble of optimization methods"""
        idx = int(action_init[0])
        
        # Method 1: Improved gradient descent with multiple restarts
        gradient_results = []
        for restart in range(self.num_restarts):
            if restart == 0:
                init_action = action_init.copy()
            else:
                # Generate new initialization
                candidates = self.smart_action_initialization(dlo_0, dlo_1, idx, num_candidates=1)
                init_action = candidates[0]
            
            result = self.find_action_improved_gradient(dlo_0, dlo_1, init_action)
            gradient_results.append(result)
        
        # Find best gradient result
        best_gradient_idx = np.argmin([r['best_loss'] for r in gradient_results])
        best_gradient_result = gradient_results[best_gradient_idx]
        
        # Method 2: Scipy optimization (if enabled)
        scipy_result = None
        if self.use_ensemble:
            try:
                # Convert to normalized space for scipy
                cs0, csR = normalization_pca.compute_normalize_factors(dlo_0)
                dlo_0_n = normalization_pca.normalize_pca(dlo_0, cs0, csR)
                dlo_1_n = normalization_pca.normalize_pca(dlo_1, cs0, csR)
                
                dlo_0_tn = torch.from_numpy(dlo_0_n.copy()).float().unsqueeze_(0)
                dlo_1_tn = torch.from_numpy(dlo_1_n.copy()).float().unsqueeze_(0)
                
                scipy_action, scipy_loss = self.scipy_optimization(dlo_0_tn, dlo_1_tn, idx, None)
                if scipy_action is not None:
                    scipy_result = {
                        'best_action': scipy_action,
                        'best_loss': scipy_loss
                    }
            except:
                pass
        
        # Choose best result
        candidates = [best_gradient_result]
        if scipy_result is not None:
            candidates.append(scipy_result)
        
        best_candidate = min(candidates, key=lambda x: x['best_loss'])
        
        return {
            'opt_log': best_gradient_result['opt_log'],
            'best_action': best_candidate['best_action'],
            'best_loss': best_candidate['best_loss'],
            'method': 'gradient' if best_candidate == best_gradient_result else 'scipy'
        }

    def find_action_improved_gradient(self, dlo_0, dlo_1, action_init):
        """Improved gradient-based optimization"""
        cs0, csR = normalization_pca.compute_normalize_factors(dlo_0)
        dlo_0_n = normalization_pca.normalize_pca(dlo_0, cs0, csR)
        dlo_1_n = normalization_pca.normalize_pca(dlo_1, cs0, csR)
        
        dlo_0_tn = torch.from_numpy(dlo_0_n.copy()).float().unsqueeze_(0)
        dlo_1_tn = torch.from_numpy(dlo_1_n.copy()).float().unsqueeze_(0)
        
        action_init_n = self.normalize_action(action_init, cs0, csR)
        
        idx = torch.tensor([[action_init_n[0]]], dtype=torch.float32)
        trainable_params = torch.nn.Parameter(
            torch.tensor([action_init_n[1:]], dtype=torch.float32), 
            requires_grad=True
        )
        
        # Use different optimizers in sequence
        optimizers = [
            torch.optim.AdamW([trainable_params], lr=self.lr, weight_decay=1e-4),
            torch.optim.SGD([trainable_params], lr=self.lr * 0.1, momentum=0.9),
        ]
        
        # Schedulers
        schedulers = [
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizers[0], T_max=self.num_steps//2),
            torch.optim.lr_scheduler.ReduceLROnPlateau(optimizers[1], patience=50, factor=0.5)
        ]
        
        opt_log_dict = {}
        best_loss = float('inf')
        best_action = None
        patience_counter = 0
        patience_limit = 100
        
        # Phase 1: AdamW with cosine annealing
        current_optimizer = optimizers[0]
        current_scheduler = schedulers[0]
        
        for step in range(self.num_steps):
            # Switch optimizer halfway through
            if step == self.num_steps // 2:
                current_optimizer = optimizers[1]
                current_scheduler = schedulers[1]
            
            current_optimizer.zero_grad()
            
            # Apply constraints using tanh for bounded optimization
            constrained_params = torch.tanh(trainable_params) * 0.2  # Scale to reasonable range
            
            action_tn = torch.cat([idx, constrained_params], dim=1)
            pred = self.model(dlo_0_tn, action_tn)
            
            # Use Huber loss for robustness
            loss = self.loss_fcn_huber(pred, dlo_1_tn)
            
            # Add L2 regularization to prevent large actions
            reg_loss = 0.001 * torch.sum(constrained_params ** 2)
            total_loss = loss + reg_loss
            
            # Save action
            action_save = action_tn.clone().squeeze().detach().numpy()
            opt_log_dict[step] = {
                "loss": loss.item(),
                "action": action_save,
            }
            
            # Track best
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_action = action_save.copy()
                patience_counter = 0
            else:
                patience_counter += 1
            
            if self.verbose and step % 50 == 0:
                print(f"Step {step}, Loss: {loss.item():.6f}, Best: {best_loss:.6f}")
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_([trainable_params], max_norm=1.0)
            
            current_optimizer.step()
            
            # Update scheduler
            if step < self.num_steps // 2:
                current_scheduler.step()
            else:
                current_scheduler.step(loss.item())
            
            # Early stopping
            if patience_counter >= patience_limit:
                if self.verbose:
                    print(f"Early stopping at step {step}")
                break
        
        return {
            'opt_log': opt_log_dict,
            'best_action': best_action,
            'best_loss': best_loss
        }

    def run(self, dlo_0, dlo_1, action_gt=None, idx=None):
        """Main optimization routine"""
        if idx is None and action_gt is None:
            raise ValueError("Either idx or action_gt must be provided")

        if action_gt is None:
            action_gt = [idx, 0, 0, 0]

        cs0, csR = normalization_pca.compute_normalize_factors(dlo_0)
        dlo_0_n = normalization_pca.normalize_pca(dlo_0, cs0, csR)
        dlo_1_n = normalization_pca.normalize_pca(dlo_1, cs0, csR)
        action_gt_n = self.normalize_action(action_gt, cs0, csR)

        # Generate smart initializations
        idx = int(action_gt[0])
        init_candidates = self.smart_action_initialization(dlo_0, dlo_1, idx, num_candidates=8)
        
        # Evaluate initial candidates
        dlo_0_tn = torch.from_numpy(dlo_0_n.copy()).float().unsqueeze_(0)
        dlo_1_tn = torch.from_numpy(dlo_1_n.copy()).float().unsqueeze_(0)
        
        candidate_losses = []
        for candidate in init_candidates:
            candidate_n = self.normalize_action(candidate, cs0, csR)
            candidate_tn = torch.from_numpy(candidate_n).float().unsqueeze_(0)
            
            with torch.no_grad():
                pred = self.model(dlo_0_tn, candidate_tn)
                loss = self.loss_fcn(pred, dlo_1_tn)
                candidate_losses.append(loss.item())
        
        # Choose best initialization
        best_init_idx = np.argmin(candidate_losses)
        best_init_action = init_candidates[best_init_idx]
        
        print(f"Best initialization loss: {candidate_losses[best_init_idx]:.6f}")
        
        # Run ensemble optimization
        optimization_result = self.find_action_ensemble(dlo_0, dlo_1, best_init_action)
        
        # Extract results
        best_action_n = optimization_result['best_action']
        best_action = self.denormalize_action(best_action_n, cs0, csR)
        
        print(f"Optimization method: {optimization_result.get('method', 'gradient')}")
        print(f"Final loss: {optimization_result['best_loss']:.6f}")
        
        # Convert tensors for predictions
        best_action_tn = torch.from_numpy(best_action_n).float().unsqueeze_(0)
        init_action_n = self.normalize_action(best_init_action, cs0, csR)
        init_action_tn = torch.from_numpy(init_action_n).float().unsqueeze_(0)
        action_gt_tn = torch.from_numpy(action_gt_n).float().unsqueeze_(0)

        # Generate predictions
        with torch.no_grad():
            pred = self.model(dlo_0_tn, best_action_tn)
            pred_init = self.model(dlo_0_tn, init_action_tn)
            pred_gt = self.model(dlo_0_tn, action_gt_tn)

        # Denormalize predictions
        pred = normalization_pca.denormalize_pca(to_numpy(pred.squeeze()), cs0, csR)
        pred_init = normalization_pca.denormalize_pca(to_numpy(pred_init.squeeze()), cs0, csR)
        pred_gt = normalization_pca.denormalize_pca(to_numpy(pred_gt.squeeze()), cs0, csR)

        # Calculate final losses
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
            "opt_log": optimization_result['opt_log'],
            "best_action": best_action,
            "best_action_normalized": best_action_n,
            "init_action": best_init_action,
            "init_action_normalized": init_action_n,
            "gt_action": action_gt,
            "gt_action_normalized": action_gt_n,
            "loss_action_gt": loss_action_gt,
            "loss_action_init": loss_action_init,
            "loss_pred": loss_pred,
            "optimization_method": optimization_result.get('method', 'gradient'),
            "num_candidates_evaluated": len(init_candidates),
        }

        return output_log

    def plot_log(self, log_dict):
        """Enhanced plotting with more information"""
        opt_log = log_dict["opt_log"]
        pred = log_dict["pred"]
        pred_init = log_dict["pred_init"]
        pred_gt = log_dict["pred_gt"]
        dlo_0 = log_dict["dlo_0"]
        dlo_1 = log_dict["dlo_1"]

        best_action = log_dict["best_action"]
        true_action = log_dict["true_action"]
        init_action = log_dict["init_action"]

        x_axis = np.array(list(opt_log.keys()))
        loss_list = np.array([opt_log[k]["loss"] for k in opt_log.keys()])
        action_x_list = np.array([opt_log[k]["action"][1] for k in opt_log.keys()])
        action_y_list = np.array([opt_log[k]["action"][2] for k in opt_log.keys()])
        action_theta_list = np.array([opt_log[k]["action"][3] for k in opt_log.keys()])

        # Create figure with subplots
        fig = plt.figure(figsize=(16, 12))
        
        # Main shape plot
        ax1 = plt.subplot(2, 3, (1, 2))
        ax1.plot(dlo_0[:, 0], dlo_0[:, 1], "o-", label="initial", linewidth=2, markersize=4)
        ax1.plot(dlo_1[:, 0], dlo_1[:, 1], "s-", label="target", linewidth=2, markersize=4)
        ax1.plot(pred[:, 0], pred[:, 1], "^-", label="pred_best", linewidth=2, markersize=4)

        # Plot actions as arrows
        idx = int(best_action[0])
        start_pos = dlo_0[idx, :2]
        
        # Best action
        ax1.arrow(start_pos[0], start_pos[1], best_action[1], best_action[2],
                 head_width=0.005, head_length=0.01, fc="red", ec="red", 
                 length_includes_head=True, label="best_action", linewidth=2)
        
        # True action
        ax1.arrow(start_pos[0], start_pos[1], true_action[1], true_action[2],
                 head_width=0.005, head_length=0.01, fc="green", ec="green", 
                 length_includes_head=True, label="true_action", linewidth=2)
        

        ax1.scatter(dlo_0[0, 0], dlo_0[0, 1], marker="X", s=100, c="black")
        ax1.set_title(f"Shape Evolution (Method: {log_dict.get('optimization_method', 'gradient')})")
        ax1.axis("equal")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Loss evolution
        ax2 = plt.subplot(2, 3, 3)
        ax2.plot(x_axis, loss_list, 'b-', linewidth=2)
        ax2.set_title("Loss Evolution")
        ax2.set_xlabel("Step")
        ax2.set_ylabel("Loss")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Action component evolution
        ax3 = plt.subplot(2, 3, 4)
        ax3.plot(x_axis, action_x_list, 'r-', linewidth=2, label=f'disp_x (final: {best_action[1]:.3f})')
        ax3.axhline(y=true_action[1], color='g', linestyle='--', alpha=0.7, label='true_x')
        ax3.set_title("Displacement X Evolution")
        ax3.set_xlabel("Step")
        ax3.set_ylabel("Displacement X")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        ax4 = plt.subplot(2, 3, 5)
        ax4.plot(x_axis, action_y_list, 'm-', linewidth=2, label=f'disp_y (final: {best_action[2]:.3f})')
        ax4.axhline(y=true_action[2], color='g', linestyle='--', alpha=0.7, label='true_y')
        ax4.set_title("Displacement Y Evolution")
        ax4.set_xlabel("Step")
        ax4.set_ylabel("Displacement Y")
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        ax5 = plt.subplot(2, 3, 6)
        ax5.plot(x_axis, action_theta_list, 'c-', linewidth=2, label=f'theta (final: {best_action[3]:.3f})')
        ax5.axhline(y=true_action[3], color='g', linestyle='--', alpha=0.7, label='true_theta')
        ax5.set_title("Theta Evolution")
        ax5.set_xlabel("Step")
        ax5.set_ylabel("Theta (radians)")
        ax5.legend()
        ax5.grid(True, alpha=0.3)

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
    checkpoint_path = "/home/mengo/Research/LLM_DOM/diffusion_mk2/weights/chkp_50000.pt"

    action_finder = ActionFinderGradient(
        checkpoint_path=checkpoint_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
        lr=5e-3,
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



        action_finder.plot_log(log)