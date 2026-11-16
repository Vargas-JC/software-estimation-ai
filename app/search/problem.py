from dataclasses import dataclass
from app.schemas import ProfileRequest, TaskRequest


@dataclass(frozen=True)
class AssignmentState:
    task_index: int
    assignments: tuple[int, ...]
    profile_hours: tuple[float, ...]
    accumulated_cost: float


class AssignmentProblem:
    """
    Formalizacion del espacio de busqueda para asignacion de tareas a perfiles:
    - Estado inicial: ninguna tarea asignada (task_index = 0).
    - Acciones: asignar la tarea actual a un perfil que no exceda max_hours_per_profile.
    - Test objetivo: task_index == len(tasks).
    - Costo de camino g(n): suma de (estimate_hours * hourly_rate) acumulada.
    """

    def __init__(
        self,
        tasks: list[TaskRequest],
        profiles: list[ProfileRequest],
        max_hours_per_profile: float | None = None,
    ):
        self.tasks = tasks
        self.profiles = profiles
        self.max_hours = max_hours_per_profile
        self.min_rate = min(p.hourly_rate for p in profiles)

    def initial_state(self) -> AssignmentState:
        return AssignmentState(
            task_index=0,
            assignments=(),
            profile_hours=tuple(0.0 for _ in self.profiles),
            accumulated_cost=0.0,
        )

    def is_goal(self, state: AssignmentState) -> bool:
        return state.task_index == len(self.tasks)

    def actions(self, state: AssignmentState) -> list[int]:
        if self.is_goal(state):
            return []
        hours = self.tasks[state.task_index].estimate_hours
        valid_profiles: list[int] = []
        for idx in range(len(self.profiles)):
            if self.max_hours is None or state.profile_hours[idx] + hours <= self.max_hours:
                valid_profiles.append(idx)
        return valid_profiles

    def transition(self, state: AssignmentState, profile_idx: int) -> AssignmentState:
        hours = self.tasks[state.task_index].estimate_hours
        step_cost = hours * self.profiles[profile_idx].hourly_rate
        new_hours = list(state.profile_hours)
        new_hours[profile_idx] += hours
        return AssignmentState(
            task_index=state.task_index + 1,
            assignments=state.assignments + (profile_idx,),
            profile_hours=tuple(new_hours),
            accumulated_cost=state.accumulated_cost + step_cost,
        )

    def heuristic(self, state: AssignmentState) -> float:
        # Heuristica admisible: horas restantes * menor tarifa por hora disponible.
        # Nunca sobreestima el costo real restante, garantizando optimalidad en A*.
        remaining_hours = sum(t.estimate_hours for t in self.tasks[state.task_index :])
        return remaining_hours * self.min_rate
