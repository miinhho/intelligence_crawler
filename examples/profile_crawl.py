"""Example: Performance profiling during crawl"""

import asyncio

from crawler import IntelligentCrawler
from crawler.crawler_engine import CrawlConfig


async def main():
    # Enable profiling
    config = CrawlConfig(
        max_pages=10,
        max_depth=2,
        request_delay=0.5,
    )

    crawler = IntelligentCrawler(config, enable_profiling=True)

    results = await crawler.crawl(
        seed_urls=["https://example.com"],
        topic="technology",
        relevance_threshold=0.5,
    )

    # Performance summary is automatically printed
    # Also available in results["performance"]
    print(f"\nCrawled {len(results['pages'])} pages")
    print(f"Found {len(results['links'])} links")

    # Access specific metrics
    if "performance" in results:
        perf = results["performance"]
        print(f"\nTotal time: {perf['total_time']:.2f}s")
        print(f"Current memory: {perf['current_memory_mb']:.2f}MB")
        print(f"CPU usage: {perf['cpu_percent']:.1f}%")

        # Cache stats
        if perf.get("cache_stats"):
            print("\nCache stats:")
            for component, stats in perf["cache_stats"].items():
                print(f"  {component}: {stats}")


if __name__ == "__main__":
    asyncio.run(main())
