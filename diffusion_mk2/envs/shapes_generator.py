import numpy as np
import genesis as gs
import os
import argparse
from genesis.engine.entities import RigidEntity, MPMEntity
from genesis.engine.entities.rigid_entity import RigidLink
import diffusion_mk2.utils.dlo_computations as dlo_utils


PROJECT_FOLDER = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NUMBER_OF_EPISODES = 3
NUMBER_OF_ACTIONS_PER_EPISODE = 8
ACTION_TIME = 2.0  # seconds (for 256 batch size)
VELOCITY = 0.05  # m/s
DT = 1e-2  # simulation time step
MPM_GRID_DENSITY = 256
SUBSTEPS = 40
TABLE_HEIGHT = 0.7005
HEIGHT_OFFSET = TABLE_HEIGHT
EE_OFFSET = 0.119
EE_QUAT_ROTATION = np.array([0, 0, -1, 0])
ROPE_LENGTH = 0.2
ROPE_RADIUS = 0.003
ROPE_BASE_POSITION = np.array([0.5, 0.0, HEIGHT_OFFSET + ROPE_RADIUS])
NUMBER_OF_PARTICLES = 15
PARTICLES_NUMBER_FOR_POS_SMOOTHING = 10


class ShapesGenerator:
    def __init__(self, vis=False, gui=False, cpu=False, show_fps=False):

        self.vis = vis
        self.gui = gui
        self.cpu = cpu
        self.show_fps = show_fps


        gs.init(
            backend=gs.cpu if self.cpu else gs.gpu,
            logging_level="error",
        )

        ########################## create a scene ##########################
        self.scene: gs.Scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=DT,
                substeps=SUBSTEPS,
            ),
            viewer_options=gs.options.ViewerOptions(
                res=(1080, 720),
                camera_pos=(0.5, 0.0, 1.4),
                camera_lookat=(0.5, 0.0, 0.0),
                camera_fov=80,
                refresh_rate=30,
                max_FPS=240,
            ),
            vis_options=gs.options.VisOptions(
                visualize_mpm_boundary=True,
                show_world_frame=True,
            ),
            mpm_options=gs.options.MPMOptions(
                lower_bound=(0.2, -0.3, HEIGHT_OFFSET - 0.05),
                upper_bound=(0.8, 0.3, HEIGHT_OFFSET + 0.1),
                grid_density=MPM_GRID_DENSITY,
            ),
            show_FPS=self.show_fps,
            show_viewer=self.vis,
        )

        self.cam = self.scene.add_camera(
            res=(1080, 720),
            pos=(0.5, -0.5, TABLE_HEIGHT + 0.6),
            lookat=(0.5, 0.0, TABLE_HEIGHT),
            fov=50,
            GUI=self.gui,
        )

        ########################## entities ##########################
        self.plane = self.scene.add_entity(
            gs.morphs.Plane(),
        )

        self.table = self.scene.add_entity(
            morph=gs.morphs.URDF(
                file=os.path.join(PROJECT_FOLDER, "models/SimpleTable/SimpleTable.urdf"),
                pos=(0.0, 0.0, 0.0),
                euler=(0, 0, 90),
                scale=1,
                fixed=True,
            ),
            material=gs.materials.Rigid(),
            surface=gs.surfaces.Default(),
        )

        self.rope: MPMEntity = self.scene.add_entity(
            material=gs.materials.MPM.Elastic(
                E=5e4,  # Determines the squishiness of the rope (very low values act as a sponge)
                nu=0.45,
                rho=2000,
                sampler="pbs",
            ),
            morph=gs.morphs.Cylinder(
                height=ROPE_LENGTH,
                radius=ROPE_RADIUS,
                pos=ROPE_BASE_POSITION,
                euler=(90, 0, 0),
            ),
            surface=gs.surfaces.Default(roughness=2, vis_mode="particle"),
        )
        self.franka: RigidEntity = self.scene.add_entity(
            gs.morphs.MJCF(
                file="xml/franka_emika_panda/panda.xml",
                pos=(0.0, 0.0, HEIGHT_OFFSET),
            ),
            material=gs.materials.Rigid(
                friction=2.0,
                needs_coup=True,
                coup_friction=2.0,
                sdf_cell_size=0.005,
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
        """Reset the environment for a new episode."""

        # Reset robot to initial position
        print("ciao")
        qpos = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.04, 0.04])
        self.franka.set_qpos(qpos)
        
        # Reset rope with random position
        rope_pos = ROPE_BASE_POSITION
        self.rope.set_position(rope_pos)

        
        # Step to initialize positions
        self.scene.step()


    def make_U_shape(self):
        target_pos, target_quat = dlo_utils.compute_pose_from_paticle_index(
            self.rope.get_particles(),
            particle_index=self.rope.get_particles().shape[0] // 2,
            ee_quat_offset=EE_QUAT_ROTATION,
            ee_offset=EE_OFFSET,
        )
        target_pos += np.array([-0.05, 0.0, 0.0])
        qpos = self.franka.inverse_kinematics(
            link=self.end_effector,
            pos=target_pos,
            quat=EE_QUAT_ROTATION,
        )
        qpos[-2:] = 0.0
        self.franka.set_qpos(qpos)
        self.scene.step()

        target_pos[0] += 0.15
        qpos = self.franka.inverse_kinematics(
            link=self.end_effector,
            pos=target_pos,
            quat=EE_QUAT_ROTATION,
        )
        qpos[-2:] = 0.0

        path = self.franka.plan_path(
            qpos_goal=qpos,
            num_waypoints=int(2.0 // DT),
        )

        for p in path:
            self.franka.control_dofs_position(p)
            self.scene.step()   
            if np.linalg.norm(self.end_effector.get_pos().cpu().numpy()[:2] - target_pos[:2]) < 0.01:  
                break

        print("U-shape state = ", dlo_utils.get_skeleton(
            self.rope.get_particles(),
            downsample_number=NUMBER_OF_PARTICLES,
            average_number=PARTICLES_NUMBER_FOR_POS_SMOOTHING,
            ))
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    parser.add_argument("-g", "--gui", action="store_true", default=False)
    parser.add_argument("-c", "--cpu", action="store_true", default=False)
    parser.add_argument("-f", "--show_fps", action="store_true", default=False)
    args = parser.parse_args()


    generator = ShapesGenerator(vis=args.vis, gui=args.gui, cpu=args.cpu, show_fps=args.show_fps)
    generator.reset()
    generator.make_U_shape()
