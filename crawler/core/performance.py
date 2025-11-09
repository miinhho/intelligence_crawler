"""Performance monitoring utilities using established libraries."""

import logging
import time
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable

import psutil

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """Monitor performance metrics during crawling"""

    def __init__(self):
        self.metrics: dict[str, Any] = {
            "timings": {},
            "counts": {},
            "cache_stats": {},
            "memory": {},
        }
        self.process = psutil.Process()
        self.start_time = time.time()

    @contextmanager
    def measure(self, operation: str):
        """Context manager to measure operation time"""
        start = time.time()
        start_memory = self.process.memory_info().rss / 1024 / 1024  # MB

        try:
            yield
        finally:
            elapsed = time.time() - start
            end_memory = self.process.memory_info().rss / 1024 / 1024  # MB

            if operation not in self.metrics["timings"]:
                self.metrics["timings"][operation] = []
            self.metrics["timings"][operation].append(elapsed)

            if operation not in self.metrics["memory"]:
                self.metrics["memory"][operation] = []
            self.metrics["memory"][operation].append(end_memory - start_memory)

            logger.debug(
                f"{operation}: {elapsed:.3f}s, memory: {end_memory - start_memory:.2f}MB"
            )

    def increment(self, counter: str, value: int = 1):
        """Increment a counter"""
        if counter not in self.metrics["counts"]:
            self.metrics["counts"][counter] = 0
        self.metrics["counts"][counter] += value

    def record_cache_stats(self, component: str, stats: dict):
        """Record cache statistics"""
        self.metrics["cache_stats"][component] = stats

    def get_summary(self) -> dict[str, Any]:
        """Get performance summary"""
        total_time = time.time() - self.start_time
        current_memory = self.process.memory_info().rss / 1024 / 1024  # MB

        summary = {
            "total_time": total_time,
            "current_memory_mb": current_memory,
            "cpu_percent": self.process.cpu_percent(interval=0.1),
            "timings": {},
            "counts": self.metrics["counts"],
            "cache_stats": self.metrics["cache_stats"],
        }

        # Aggregate timing stats
        for operation, times in self.metrics["timings"].items():
            if times:
                summary["timings"][operation] = {
                    "count": len(times),
                    "total": sum(times),
                    "mean": sum(times) / len(times),
                    "min": min(times),
                    "max": max(times),
                }

        # Aggregate memory stats
        memory_summary = {}
        for operation, deltas in self.metrics["memory"].items():
            if deltas:
                memory_summary[operation] = {
                    "mean_mb": sum(deltas) / len(deltas),
                    "max_mb": max(deltas),
                }
        summary["memory_usage"] = memory_summary

        return summary

    def print_summary(self):
        """Print formatted performance summary"""
        summary = self.get_summary()

        print("\n" + "=" * 60)
        print("PERFORMANCE SUMMARY")
        print("=" * 60)

        print(f"\nTotal Time: {summary['total_time']:.2f}s")
        print(f"Current Memory: {summary['current_memory_mb']:.2f} MB")
        print(f"CPU Usage: {summary['cpu_percent']:.1f}%")

        if summary["timings"]:
            print("\n--- Operation Timings ---")
            for op, stats in sorted(summary["timings"].items()):
                print(f"  {op}:")
                print(f"    Count: {stats['count']}")
                print(f"    Total: {stats['total']:.2f}s")
                print(f"    Mean: {stats['mean']:.3f}s")
                print(f"    Min/Max: {stats['min']:.3f}s / {stats['max']:.3f}s")

        if summary["counts"]:
            print("\n--- Counters ---")
            for name, value in sorted(summary["counts"].items()):
                print(f"  {name}: {value}")

        if summary["cache_stats"]:
            print("\n--- Cache Statistics ---")
            for component, stats in summary["cache_stats"].items():
                print(f"  {component}:")
                for key, value in stats.items():
                    print(f"    {key}: {value}")

        if summary.get("memory_usage"):
            print("\n--- Memory Usage by Operation ---")
            for op, stats in sorted(summary["memory_usage"].items()):
                print(
                    f"  {op}: mean={stats['mean_mb']:.2f}MB, max={stats['max_mb']:.2f}MB"
                )

        print("=" * 60 + "\n")


def timed(func: Callable) -> Callable:
    """Decorator to time function execution"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        logger.info(f"{func.__name__} took {elapsed:.3f}s")
        return result

    return wrapper


def async_timed(func: Callable) -> Callable:
    """Decorator to time async function execution"""

    @wraps(func)
    async def wrapper(*args, **kwargs):
        start = time.time()
        result = await func(*args, **kwargs)
        elapsed = time.time() - start
        logger.info(f"{func.__name__} took {elapsed:.3f}s")
        return result

    return wrapper
