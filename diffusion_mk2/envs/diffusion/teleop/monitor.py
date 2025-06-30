from rich.text import Text
from rich.panel import Panel
from rich.layout import Layout
from rich.align import Align
from rich.box import ROUNDED
from genesis.engine.entities.rigid_entity import RigidLink


class Monitor():
    def __init__(self):
        self.status_message = None
        self.status_active = False
    
    def get_layout(self):
        """Define the layout for the Rich Live display."""
        layout = Layout(name="root")
        

        layout.split(
            Layout(name="header", size=3),
            Layout(name="status", size=3), # New layout for status
            Layout(name="main", ratio=1),
            Layout(name="footer", size=5)
        )
        layout["main"].split_row(
            Layout(name="controls"),
            Layout(name="info")
        )

        return layout

    def make_header_panel(self):
        return Panel(
            Align.center(
                Text(f"Teleoperation Data Generator", justify="center", style="bold yellow"),
                vertical="middle"
            ),
            title="[bold blue]DLO Pushin Teleop[/bold blue]",
            border_style="blue",
            box=ROUNDED
        )

    def make_controls_panel(self):
        controls_text = Text()
        controls_text.append("Movement Controls:\n", style="bold underline")
        controls_text.append("  W: Move End-Effector -X (forward)\n")
        controls_text.append("  S: Move End-Effector +X (backward)\n")
        controls_text.append("  A: Move End-Effector -Y (left)\n")
        controls_text.append("  D: Move End-Effector +Y (right)\n\n")
        controls_text.append("Episode Controls:\n", style="bold underline")
        controls_text.append("  SPACE: Save current episode data and start next episode\n")
        controls_text.append("  BACKSPACE: Skip current episode (data will NOT be saved)\n", style="red")

        return Panel(
            controls_text,
            title="[bold green]Controls[/bold green]",
            border_style="green",
            box=ROUNDED
        )

    def make_info_panel(self, 
                        current_episode, 
                        total_episodes, 
                        current_step, 
                        current_key_pressed, 
                        end_effector: RigidLink,
                        real_time_factor: float):
        info_text = Text()
        info_text.append("Real-time factor: ", style="none")
        info_text.append(f"{real_time_factor:.3f}x\n", style="bold cyan")
        info_text.append("Current Episode: ", style="none")
        info_text.append(f"{current_episode}/{total_episodes}\n", style="bold cyan")
        info_text.append("Current Step in Episode: ", style="none")
        info_text.append(f"{current_step}\n", style="bold magenta")
        info_text.append("Key Pressed: ", style="none")
        info_text.append(f"{current_key_pressed}\n", style="bold yellow")
        info_text.append("End-effector X: ", style="none")
        info_text.append(f"{end_effector.get_pos().cpu().numpy()[0]:.4f}\n", style="bold white")
        info_text.append("End-effector Y: ", style="none")
        info_text.append(f"{end_effector.get_pos().cpu().numpy()[1]:.4f}\n", style="bold white")


        return Panel(
            info_text,
            title="[bold blue]Information[/bold blue]",
            border_style="blue",
            box=ROUNDED
        )

    def make_footer_panel(self):
        return Panel(
            Align.center(
                Text("Press ESC or close the window to exit.", justify="center", style="dim white"),
                vertical="middle"
            ),
            border_style="white",
            box=ROUNDED
        )

    def make_status_panel(self, 
                          resetting : bool, 
                          saving : bool, 
                          current_episode: int = 0,
                          current_step: int = 0,
                          style: str = "bold yellow"):
        if resetting:
            message = "Resetting Episode..."
        elif saving:
            message = f"Saving Episode {current_episode} with {current_step} steps..."
        else:
            message = "Teleoperation in progress..."
        return Panel(
            Align.center(
                Text(message, justify="center", style=style),
                vertical="middle"
            ),
            title="[bold red]STATUS[/bold red]",
            border_style="red",
            box=ROUNDED
        )

