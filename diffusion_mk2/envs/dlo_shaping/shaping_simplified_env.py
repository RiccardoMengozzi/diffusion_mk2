import argparse
import os
import genesis as gs
import numpy as np




class ShapingSimplifiedEnv:
    def __init__(self):
        gs.init(
            backend=gs.cpu if self.cpu else gs.gpu,
            logging_level="error",
        )

        ########################## create a scene ##########################
        self.scene: gs.Scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=config["dt"],
                substeps=config["substeps"],
            ),
            viewer_options=gs.options.ViewerOptions(
                res=config["viewer_res"],
                camera_pos=config["viewer_camera_pos"],
                camera_lookat=config["viewer_camera_lookat"],
                camera_fov=config["viewer_camera_fov"],
                refresh_rate=config["viewer_refresh_rate"],
                max_FPS=config["viewer_max_fps"],
            ),
            vis_options=gs.options.VisOptions(
                visualize_mpm_boundary=True,
                show_world_frame=True,
            ),
            mpm_options=gs.options.MPMOptions(
                lower_bound=config["mpm_lower_bound"],
                upper_bound=config["mpm_upper_bound"],
                grid_density=config["mpm_grid_density"],
            ),
            show_FPS=config["show_FPS"],
            show_viewer=config["show_viewer"],
        )



        ########################## entities ##########################
        self.plane = self.scene.add_entity(
            gs.morphs.Plane(),
        )

        self.table = self.scene.add_entity(
            morph=gs.morphs.URDF(
                file=os.path.join(PROJECT_FOLDER, "models/SimpleTable/SimpleTable.urdf"),
                pos=self.table_position,
                euler=self.table_orientation,
                scale=self.table_scale,
                fixed=True,
            ),
            material=gs.materials.Rigid(),
            surface=gs.surfaces.Default(),
        )

        self.rope: MPMEntity = self.scene.add_entity(
            material=gs.materials.MPM.Elastic(
                E=self.dlo_E,  # Determines the squishiness of the rope (very low values act as a sponge)
                nu=self.dlo_nu,
                rho=self.dlo_rho,
                sampler=self.dlo_sampler,
            ),
            morph=gs.morphs.Cylinder(
                height=self.dlo_length,
                radius=self.dlo_radius,
                pos=self.dlo_position,
                euler=self.dlo_orientation,
            ),
            surface=gs.surfaces.Default(roughness=2, vis_mode="particle"),
        )
        self.franka: RigidEntity = self.scene.add_entity(
            gs.morphs.MJCF(
                file="xml/franka_emika_panda/panda.xml",
                pos=self.franka_position,
                euler=self.franka_orientation,
            ),
            material=gs.materials.Rigid(
                friction=self.ee_friction,
                needs_coup=self.ee_needs_coup,
                coup_friction=self.ee_coup_friction,
                sdf_cell_size=self.ee_sdf_cell_size,
                gravity_compensation=self.franka_gravity_compensation,
            ),
        )

        ########################## build ##########################
        self.scene.build()

        self.motors_dof = np.arange(7)
        self.fingers_dof = np.arange(7, 9)
        self.end_effector : RigidLink = self.franka.get_link("hand")

        # Optional: set control gains
        self.franka.set_dofs_kp(
            np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]),
        )
        self.franka.set_dofs_kv(
            np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]),
        )
        self.franka.set_dofs_force_range(
            np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
            np.array([87, 87, 87, 87, 12, 12, 12, 100, 100]),
        )

    def reset(self):
        pass

    def get_obs(self):
        pass

    def execute_trajectory(self):
        pass

    def run(self):
        pass


def main():
    parser = argparse.ArgumentParser(description="Teleop Push Data Generator")
    parser.add_argument("--cfg", type=str, default="diffusion_mk2/config/shaping_traj_prediction_env_config.yaml", help="Path to the configuration file")
    parser.add_argument("-v", "--vis", action="store_true")
    parser.add_argument("-g", "--gui", action="store_true", help="Enable GUI mode")
    parser.add_argument("-c", "--cpu", action="store_true", help="Run on CPU instead of GPU")
    parser.add_argument("-f", "--show_fps", action="store_true", help="Show FPS in the viewer")
    parser.add_argument("-e", "--n_episodes", type=int, default=1, help="Number of episodes to run")
    parser.add_argument("-a", "--n_actions", type=int, default=1, help="Number of actions per episode")
    args = parser.parse_args()

    env = ShapingSimplifiedEnv(args=args)
    env.run()


if __name__ == "__main__":
    main()
    













