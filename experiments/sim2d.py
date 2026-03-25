"""
2D Pymunk Simulator for Phase 1 Validation

Generates bouncing ball trajectories with configurable physics:
  - Variable mass, friction, elasticity per object
  - Gravity, walls
  - Ground-truth (q, v) at every timestep
  - Contact events logged

This is the test bed for the hybrid belief simulator.
"""

import pymunk
import pymunk.autogeometry
import numpy as np
import torch
from dataclasses import dataclass, field


@dataclass
class ObjectConfig:
    """Configuration for a single simulated object."""
    radius: float = 15.0
    mass: float = 1.0
    elasticity: float = 0.8
    friction: float = 0.5
    position: tuple[float, float] = (250.0, 250.0)
    velocity: tuple[float, float] = (0.0, 0.0)


@dataclass
class SimConfig:
    """Configuration for the 2D simulation."""
    width: float = 500.0
    height: float = 500.0
    gravity: tuple[float, float] = (0.0, -300.0)
    dt: float = 1.0 / 60.0
    substeps: int = 10
    objects: list[ObjectConfig] = field(default_factory=list)


class Sim2D:
    """2D physics simulator using pymunk.

    Produces ground-truth trajectories for validating the
    hybrid belief simulator.
    """

    def __init__(self, config: SimConfig):
        self.config = config
        self.space = pymunk.Space()
        self.space.gravity = config.gravity

        # Track contact events
        self.contacts: list[dict] = []

        # Add walls
        self._add_walls()

        # Add objects
        self.bodies: list[pymunk.Body] = []
        self.shapes: list[pymunk.Shape] = []
        for obj_cfg in config.objects:
            body, shape = self._add_ball(obj_cfg)
            self.bodies.append(body)
            self.shapes.append(shape)

        # Contact handler
        self.space.on_collision(begin=self._on_contact_begin)

    def _add_walls(self):
        """Add static boundary walls."""
        w, h = self.config.width, self.config.height
        walls = [
            ((0, 0), (w, 0)),      # bottom
            ((0, 0), (0, h)),      # left
            ((w, 0), (w, h)),      # right
            ((0, h), (w, h)),      # top
        ]
        for a, b in walls:
            seg = pymunk.Segment(self.space.static_body, a, b, 2.0)
            seg.elasticity = 0.9
            seg.friction = 0.5
            self.space.add(seg)

    def _add_ball(self, cfg: ObjectConfig) -> tuple[pymunk.Body, pymunk.Shape]:
        """Add a ball to the simulation."""
        moment = pymunk.moment_for_circle(cfg.mass, 0, cfg.radius)
        body = pymunk.Body(cfg.mass, moment)
        body.position = cfg.position
        body.velocity = cfg.velocity
        shape = pymunk.Circle(body, cfg.radius)
        shape.elasticity = cfg.elasticity
        shape.friction = cfg.friction
        self.space.add(body, shape)
        return body, shape

    def _on_contact_begin(self, arbiter, space, data):
        """Log contact events."""
        shapes = arbiter.shapes
        # Find which objects are involved
        ids = []
        for s in shapes:
            if s in self.shapes:
                ids.append(self.shapes.index(s))
            else:
                ids.append(-1)  # wall

        self.contacts.append({
            "step": self._current_step,
            "objects": tuple(ids),
            "impulse": arbiter.total_impulse.length,
        })
        return True

    def get_state(self) -> dict[str, np.ndarray]:
        """Get current state of all objects."""
        K = len(self.bodies)
        positions = np.zeros((K, 2), dtype=np.float32)
        velocities = np.zeros((K, 2), dtype=np.float32)
        masses = np.zeros(K, dtype=np.float32)
        frictions = np.zeros(K, dtype=np.float32)
        elasticities = np.zeros(K, dtype=np.float32)

        for i, (body, shape) in enumerate(zip(self.bodies, self.shapes)):
            positions[i] = [body.position.x, body.position.y]
            velocities[i] = [body.velocity.x, body.velocity.y]
            masses[i] = body.mass
            frictions[i] = shape.friction
            elasticities[i] = shape.elasticity

        return {
            "positions": positions,
            "velocities": velocities,
            "masses": masses,
            "frictions": frictions,
            "elasticities": elasticities,
        }

    def step(self):
        """Advance simulation by one timestep."""
        dt_sub = self.config.dt / self.config.substeps
        for _ in range(self.config.substeps):
            self.space.step(dt_sub)

    def rollout(self, n_steps: int) -> dict[str, np.ndarray]:
        """Run simulation for n_steps and collect trajectory.

        Returns:
            dict with:
                positions: (T, K, 2)
                velocities: (T, K, 2)
                masses: (K,) — constant per object
                frictions: (K,) — constant per object
                elasticities: (K,) — constant per object
                contacts: list of contact events
        """
        self.contacts = []
        trajectory = {"positions": [], "velocities": []}

        for t in range(n_steps):
            self._current_step = t
            state = self.get_state()
            trajectory["positions"].append(state["positions"])
            trajectory["velocities"].append(state["velocities"])
            self.step()

        # Final state
        state = self.get_state()
        trajectory["positions"].append(state["positions"])
        trajectory["velocities"].append(state["velocities"])

        return {
            "positions": np.stack(trajectory["positions"]),  # (T+1, K, 2)
            "velocities": np.stack(trajectory["velocities"]),
            "masses": state["masses"],
            "frictions": state["frictions"],
            "elasticities": state["elasticities"],
            "contacts": self.contacts,
        }


def generate_dataset(
    n_trajectories: int = 1000,
    n_steps: int = 60,
    n_objects: int = 3,
    mass_range: tuple[float, float] = (0.5, 5.0),
    friction_range: tuple[float, float] = (0.1, 0.9),
    elasticity_range: tuple[float, float] = (0.3, 0.95),
    seed: int = 42,
) -> dict[str, torch.Tensor]:
    """Generate a dataset of 2D physics trajectories.

    Returns:
        dict with tensors:
            positions: (N, T+1, K, 2)
            velocities: (N, T+1, K, 2)
            masses: (N, K)
            frictions: (N, K)
            elasticities: (N, K)
            contact_matrices: (N, T, K, K) binary contact at each step
    """
    rng = np.random.RandomState(seed)

    all_pos, all_vel = [], []
    all_mass, all_fric, all_elast = [], [], []
    all_contacts = []

    for traj_idx in range(n_trajectories):
        objects = []
        for _ in range(n_objects):
            obj = ObjectConfig(
                radius=rng.uniform(10, 25),
                mass=rng.uniform(*mass_range),
                elasticity=rng.uniform(*elasticity_range),
                friction=rng.uniform(*friction_range),
                position=(rng.uniform(50, 450), rng.uniform(50, 450)),
                velocity=(rng.uniform(-200, 200), rng.uniform(-200, 200)),
            )
            objects.append(obj)

        config = SimConfig(objects=objects)
        sim = Sim2D(config)
        traj = sim.rollout(n_steps)

        all_pos.append(traj["positions"])
        all_vel.append(traj["velocities"])
        all_mass.append(traj["masses"])
        all_fric.append(traj["frictions"])
        all_elast.append(traj["elasticities"])

        # Build contact matrix per step
        contact_mat = np.zeros((n_steps, n_objects, n_objects), dtype=np.float32)
        for c in traj["contacts"]:
            t = c["step"]
            i, j = c["objects"]
            if i >= 0 and j >= 0 and t < n_steps:
                contact_mat[t, i, j] = 1.0
                contact_mat[t, j, i] = 1.0
        all_contacts.append(contact_mat)

        if (traj_idx + 1) % 200 == 0:
            print(f"  Generated {traj_idx + 1}/{n_trajectories}")

    # Normalize positions to [0, 1]
    positions = np.stack(all_pos) / 500.0
    velocities = np.stack(all_vel) / 500.0

    return {
        "positions": torch.from_numpy(positions),
        "velocities": torch.from_numpy(velocities),
        "masses": torch.from_numpy(np.stack(all_mass)),
        "frictions": torch.from_numpy(np.stack(all_fric)),
        "elasticities": torch.from_numpy(np.stack(all_elast)),
        "contact_matrices": torch.from_numpy(np.stack(all_contacts)),
    }


if __name__ == "__main__":
    print("Generating 2D physics dataset...")
    data = generate_dataset(n_trajectories=100, n_steps=60, n_objects=3)
    for k, v in data.items():
        print(f"  {k}: {v.shape} {v.dtype}")

    n_contacts = data["contact_matrices"].sum().item()
    print(f"\n  Total contact events: {int(n_contacts)}")
    print("Done!")
