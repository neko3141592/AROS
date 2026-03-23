def calculate_next_timeout(per_try_timeout: float, total_max_timeout: float, current_total_used: float) -> float:

    remaining_budget = max(0, total_max_timeout - current_total_used)
    next_timeout = min(per_try_timeout, remaining_budget)

    return next_timeout
