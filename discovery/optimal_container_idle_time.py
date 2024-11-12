import random
import statistics

events = 100000
max_gap_between_events = 10 * 60
cold_start_time = statistics.mean([16, 5, 6, 6, 7, 6, 5, 34, 12, 5, 5])
gaps_between_events = [random.randint(1, max_gap_between_events) for _ in range(events)]
print(f"{max_gap_between_events = }")


def simulate(timeout=60):
    cpu_time = 0
    for gap in gaps_between_events:
        if gap > timeout:
            cpu_time += cold_start_time
        else:
            cpu_time += gap

        cpu_time += random.random()

    return int(cpu_time)


timeouts = list(range(1, 15 + 1))
cpu_times = [simulate(timeout) for timeout in timeouts]

for timeout, cpu_time in zip(timeouts, cpu_times):
    print(
        f"Timeout: {timeout} seconds, CPU time: {cpu_time / min(cpu_times):.4f} seconds"
    )

# Findings:
# The most optimal container idle time is close to the average cold start time.
