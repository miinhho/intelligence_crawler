import asyncio
import json
from datetime import datetime

from crawler import IntelligentCrawler
from crawler.crawler_engine import CrawlConfig


async def test_real_website():
    """실제 웹사이트 크롤링 테스트"""
    print("=" * 70)
    print("Real Website Crawl Test with Profiling")
    print("=" * 70)

    # 테스트 설정
    TEST_CONFIGS = [
        {
            "name": "Python Docs",
            "seed_urls": ["https://docs.python.org/3/tutorial/"],
            "topic": None,  # Auto-discover topic
            "max_pages": 10,
            "max_depth": 1,
        },
        {
            "name": "ArXiv ML",
            "seed_urls": ["https://arxiv.org/list/cs.LG/recent"],
            "topic": "machine learning deep learning research",
            "max_pages": 10,
            "max_depth": 1,
        },
    ]

    # 사용자에게 선택 요청
    print("\nAvailable test sites:")
    for i, config in enumerate(TEST_CONFIGS, 1):
        print(f"  {i}. {config['name']}")
        print(f"     URL: {config['seed_urls'][0]}")
        topic_str = config["topic"] if config["topic"] else "(auto-discover topic)"
        print(f"     Topic: {topic_str}")
        print(f"     Max pages: {config['max_pages']}")

    print("\n  0. Custom (enter your own)")

    try:
        choice = int(input("\nSelect test (0-2): "))
    except (ValueError, EOFError):
        choice = 1
        print(f"Using default: {TEST_CONFIGS[0]['name']}")

    # 설정 선택
    if choice == 0:
        test_config = {
            "name": "Custom",
            "seed_urls": [input("Enter seed URL: ")],
            "topic": input("Enter topic (leave empty to auto-discover): ") or None,
            "max_pages": int(input("Max pages (default 10): ") or "10"),
            "max_depth": int(input("Max depth (default 2): ") or "2"),
        }
    elif 1 <= choice <= len(TEST_CONFIGS):
        test_config = TEST_CONFIGS[choice - 1]
    else:
        print("Invalid choice, using default")
        test_config = TEST_CONFIGS[0]

    print(f"\n{'=' * 70}")
    print(f"Testing: {test_config['name']}")
    print(f"{'=' * 70}\n")

    # 크롤러 설정
    config = CrawlConfig(
        max_pages=test_config["max_pages"],
        max_depth=test_config["max_depth"],
        request_delay=1.0,  # 실제 사이트이므로 예의상 1초
        max_concurrent_requests=3,  # 서버 부담을 줄이기 위해 3개로 제한
        respect_robots_txt=True,
    )

    print("[1/4] Initializing crawler with profiling...")
    crawler = IntelligentCrawler(config, enable_profiling=True)
    print("  ✓ Crawler initialized\n")

    print("[2/4] Starting crawl...")
    print(f"  Seed URL: {test_config['seed_urls'][0]}")
    topic_str = (
        test_config["topic"] if test_config["topic"] else "(auto-discover topic)"
    )
    print(f"  Topic: {topic_str}")
    print(f"  Max pages: {config.max_pages}")
    print(f"  Max depth: {config.max_depth}")
    print(f"  Request delay: {config.request_delay}s")
    print()

    start_time = datetime.now()

    try:
        results = await crawler.crawl(
            seed_urls=test_config["seed_urls"],
            topic=test_config["topic"],
            relevance_threshold=0.7,
        )

        end_time = datetime.now()
        elapsed = (end_time - start_time).total_seconds()

        print(f"\n[3/4] Crawl completed in {elapsed:.1f}s!")

        # Show auto-discovered topic if applicable
        if test_config["topic"] is None and results.get("auto_discovered_topic"):
            print(f"\n  🔍 Auto-discovered topic: {results['auto_discovered_topic']}")

        # 결과 분석
        stats = results["statistics"]
        print("\n  📊 Crawl Statistics:")
        print(f"    Pages crawled: {stats['num_pages']}")
        print(f"    Links found: {stats['num_links']}")
        print(f"    Internal links: {stats.get('internal_links', 0)}")
        print(f"    External links: {stats.get('external_links', 0)}")
        print(f"    Avg relevance: {stats.get('avg_topic_relevance', 0):.3f}")
        print(f"    Graph density: {stats.get('density', 0):.3f}")

        # 성능 메트릭
        if "performance" in results:
            print("\n  ⚡ Performance Metrics:")
            perf = results["performance"]
            print(f"    Total time: {perf['total_time']:.2f}s")
            print(f"    Memory usage: {perf['current_memory_mb']:.2f}MB")
            print(f"    CPU usage: {perf['cpu_percent']:.1f}%")

            # 작업별 시간
            if perf.get("timings"):
                print("\n  ⏱️  Operation Timings:")
                for op, timing in perf["timings"].items():
                    print(
                        f"    {op}: {timing['mean']:.3f}s avg "
                        f"(total: {timing['total']:.2f}s, count: {timing['count']})"
                    )

            # 캐시 통계
            if perf.get("cache_stats"):
                print("\n  💾 Cache Statistics:")
                for component, cache_stats in perf["cache_stats"].items():
                    print(f"    {component}:")
                    for key, value in cache_stats.items():
                        print(f"      {key}: {value}")

        # 상위 관련 페이지
        print("\n  📄 Top Pages by Relevance:")
        sorted_pages = sorted(
            results["pages"], key=lambda x: x["topic_relevance"], reverse=True
        )
        for i, page in enumerate(sorted_pages[:5], 1):
            relevance_bar = "█" * int(page["topic_relevance"] * 20)
            print(f"    {i}. [{page['topic_relevance']:.3f}] {relevance_bar}")
            print(f"       {page['title'][:60]}")
            print(f"       Depth: {page['depth']}, PageRank: {page['pagerank']:.4f}")

        # 결과 저장
        print("\n[4/4] Saving results...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = (
            f"crawl_results_{test_config['name'].replace(' ', '_')}_{timestamp}.json"
        )

        output_data = {
            "test_name": test_config["name"],
            "timestamp": datetime.now().isoformat(),
            "config": {
                "seed_urls": test_config["seed_urls"],
                "topic": test_config["topic"],
                "max_pages": config.max_pages,
                "max_depth": config.max_depth,
            },
            "statistics": stats,
            "pages": results["pages"],
            "links": results["links"][
                :100
            ],  # 링크가 너무 많을 수 있으므로 상위 100개만
        }

        if "performance" in results:
            output_data["performance"] = results["performance"]

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        print(f"  💾 Results saved to: {filename}")

        # 요약
        print(f"\n{'=' * 70}")
        print("✓ Test completed successfully!")
        print(f"{'=' * 70}")
        print(f"\nTotal pages: {stats['num_pages']}/{config.max_pages}")
        print(f"Avg relevance: {stats.get('avg_topic_relevance', 0):.3f}")
        print(f"Time: {elapsed:.1f}s")

        if "performance" in results:
            perf = results["performance"]
            print(f"Memory: {perf['current_memory_mb']:.2f}MB")
            print(f"CPU: {perf['cpu_percent']:.1f}%")

    except KeyboardInterrupt:
        print("\n\n⚠️  Crawl interrupted by user")
    except Exception as e:
        print(f"\n✗ Error during crawl: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_real_website())
