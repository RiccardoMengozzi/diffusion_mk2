from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class ViewerConfig:
    active: bool = True
    resolution: Tuple[int, int] = (1024, 768)
    position: List[float] = field(default_factory=lambda: [0, 0, 2])
    lookat: List[float] = field(default_factory=lambda: [0, 0, 0])
    fov: float = 60.0
    refresh_rate: int = 60
    max_fps: int = 60

@dataclass
class CameraConfig:
    gui: bool = False
    resolution: Tuple[int, int] = (1024, 768)
    position: List[float] = field(default_factory=lambda: [0.5, 0.0, 1.4])
    lookat: List[float] = field(default_factory=lambda: [0.5, 0.0, 0.0])
    fov: float = 80.0
    record: bool = False

@dataclass
class SimulationConfig:
    visualization: bool = True
    cpu: bool = False
    show_fps: bool = False
    dt: float = 0.01
    substeps: int = 1
    show_real_time_factor: bool = True
    viewer: ViewerConfig = field(default_factory=lambda: ViewerConfig)
    camera: CameraConfig = field(default_factory=lambda: CameraConfig)


@dataclass
class TableConfig:
    height: float = 0.8
    position: List[float] = field(default_factory=lambda: [0, 0, 0])
    orientation: List[float] = field(default_factory=lambda: [0, 0, 0])
    scale: float = 1.0

@dataclass
class DLOConfig:
    mpm_grid_density: int = 64
    mpm_lower_bound: List[float] = field(default_factory=lambda: [-1, -1, -1])
    mpm_upper_bound: List[float] = field(default_factory=lambda: [1, 1, 1])
    position: List[float] = field(default_factory=lambda: [0, 0, 1])
    orientation: List[float] = field(default_factory=lambda: [0, 0, 0])
    number_of_particles: int = 50
    particles_smoothing: int = 5
    length: float = 0.5
    radius: float = 0.01
    E: float = 1000.0  # Young's modulus
    nu: float = 0.3    # Poisson's ratio
    rho: float = 1000.0  # Density
    sampler: str = "regular"

@dataclass
class EndEffectorConfig:
    friction: float = 0.5
    needs_coup: bool = False
    coup_friction: float = 0.1
    sdf_cell_size: float = 0.001
    offset: float = 0.1
    z_lift: float = 0.05
    rot_offset: float = 0.0
    gripper_open_position: float = 0.04
    gripper_close_position: float = 0.0

@dataclass
class FrankaConfig:
    position: List[float] = field(default_factory=lambda: [0, 0, 0])
    orientation: List[float] = field(default_factory=lambda: [0, 0, 0])
    gravity_compensation: bool = True
    end_effector: EndEffectorConfig = field(default_factory=EndEffectorConfig)

@dataclass
class InferenceConfig:
    model_path: str = ""
    n_episodes: int = 1
    n_actions: int = 1
    plot: bool = False
    interactive: bool = False

@dataclass
class ShapingConfig:
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    viewer: ViewerConfig = field(default_factory=ViewerConfig)
    table: TableConfig = field(default_factory=TableConfig)
    dlo: DLOConfig = field(default_factory=DLOConfig)
    franka: FrankaConfig = field(default_factory=FrankaConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)

    @classmethod
    def from_yaml_and_args(cls, yaml_path: str, args):
        """Create config from YAML file and command line args"""
        from diffusion_mk2.utils.utils import load_yaml
        
        cfg_dict = load_yaml(yaml_path)
        config = cls.from_dict(cfg_dict)
        
        # Override with command line args
        config.apply_args_overrides(args)
        
        return config
    
    @classmethod
    def from_dict(cls, cfg_dict: dict):
        sim_dict = cfg_dict.get("simulation", {})
        sim_base = {k: v for k, v in sim_dict.items() if k not in ["camera", "viewer"]}
        viewer_dict = sim_dict.get("viewer", {})
        camera_dict = sim_dict.get("camera", {})

        return cls(
            simulation=SimulationConfig(
                **sim_base,
                viewer=ViewerConfig(**viewer_dict),
                camera=CameraConfig(**camera_dict)
            ),
            viewer=ViewerConfig(**viewer_dict),
            table=TableConfig(**cfg_dict.get("entities", {}).get("table", {})),
            dlo=DLOConfig(**cfg_dict.get("entities", {}).get("dlo", {})),
            franka=FrankaConfig(
                **{k: v for k, v in cfg_dict.get("entities", {}).get("franka", {}).items()
                if k != "end_effector"},
                end_effector=EndEffectorConfig(
                    **cfg_dict.get("entities", {}).get("franka", {}).get("end_effector", {})
                )
            ),
            inference=InferenceConfig(**cfg_dict.get("inference", {}))
        )
    
    def apply_args_overrides(self, args):
        """Apply command line argument overrides"""
        if args.vis is not None:
            self.simulation.visualization = args.vis
        if args.gui is not None:
            self.simulation.camera.gui = args.gui
        if args.cpu is not None:
            self.simulation.cpu = args.cpu
        if args.show_fps is not None:
            self.simulation.show_fps = args.show_fps
        if args.n_episodes is not None:
            self.inference.n_episodes = args.n_episodes
        if args.n_actions is not None:
            self.inference.n_actions = args.n_actions
        if args.record is not None:
            self.simulation.camera.record = args.record
        if args.plot is not None:
            self.inference.plot = args.plot
        if args.interactive is not None:
            self.inference.interactive = args.interactive