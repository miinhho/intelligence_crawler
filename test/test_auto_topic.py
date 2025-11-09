import asyncio
import json
import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from test.test_mock_values import run_mock_server
from crawler import IntelligentCrawler
from crawler.crawler_engine import CrawlConfig


async def test_auto_topic_discovery():
    """모의 서버로 자동 주제 탐색 테스트 (topic 없음)"""

    print("\n" + "=" * 70)
    print("Mock Server Crawl Test - AUTO TOPIC DISCOVERY")
    print("=" * 70)

    # 모의 서버 시작
    print("\n[1/5] Starting mock server...")
    server_runner = await run_mock_server(8889)

    try:
        # 설정
        config = CrawlConfig(
            max_pages=8,  # 더 많은 페이지로 주제 탐색
            max_depth=2,
            request_delay=0.1,
            respect_robots_txt=True,
        )

        # 컴포넌트 초기화
        print("\n[2/5] Initializing components...")
        crawler = IntelligentCrawler(config, enable_profiling=True)
        print("  ✓ All components initialized")

        # 크롤링 시작 (topic 없음)
        seed_url = "http://localhost:8889/"

        print("\n[3/5] Starting crawl with auto topic discovery...")
        print(f"  Seed URL: {seed_url}")
        print("  Topic: (auto-discover)")
        print(f"  Max pages: {config.max_pages}")
        print(f"  Max depth: {config.max_depth}")
        print()

        results = await crawler.crawl(
            seed_urls=[seed_url],
            topic=None,  # No topic - auto-discover
        )

        print("\n[4/5] Crawl completed!")

        # 자동 탐색된 주제 표시
        if results.get("auto_discovered_topic"):
            print("\n  🔍 Auto-discovered topic:")
            print(f"     {results['auto_discovered_topic']}")
        else:
            print("\n  ⚠ Topic was not discovered (not enough pages)")

        # 결과 분석
        stats = results["statistics"]
        print("\n  📊 Statistics:")
        print(f"    Pages crawled: {stats['num_pages']}")
        print(f"    Links found: {stats['num_links']}")
        print(f"    Internal links: {stats.get('internal_links', 0)}")
        print(f"    External links: {stats.get('external_links', 0)}")
        print(f"    Avg relevance: {stats.get('avg_topic_relevance', 0):.3f}")
        print(f"    Graph density: {stats.get('density', 0):.3f}")

        # 페이지별 관련도
        print("\n  📄 Pages by relevance:")
        for i, page in enumerate(results["pages"], 1):
            relevance_bar = "█" * int(page["topic_relevance"] * 20)
            print(f"    {i}. [{page['topic_relevance']:.3f}] {relevance_bar}")
            print(f"       {page['title']}")
            print(f"       Depth: {page['depth']}, PageRank: {page['pagerank']:.4f}")

        # 검증
        print("\n[5/5] Validation:")

        # AI 관련 페이지 체크
        ai_pages = [
            p
            for p in results["pages"]
            if any(
                word in p["title"].lower()
                for word in ["ai", "machine", "learning", "neural", "deep"]
            )
        ]
        unrelated = [
            p
            for p in results["pages"]
            if "cooking" in p["title"].lower() or "recipe" in p["title"].lower()
        ]

        if ai_pages:
            avg_ai_relevance = sum(p["topic_relevance"] for p in ai_pages) / len(
                ai_pages
            )
            print(f"  ✓ AI-related pages average relevance: {avg_ai_relevance:.3f}")

        if unrelated:
            avg_unrelated_relevance = sum(
                p["topic_relevance"] for p in unrelated
            ) / len(unrelated)
            print(
                f"  ✓ Unrelated pages average relevance: {avg_unrelated_relevance:.3f}"
            )

            if avg_unrelated_relevance < 0.5:
                print("  ✓ Good! Auto-discovery filtered out unrelated content")
            else:
                print("  ⚠ Warning: Unrelated content has high relevance")
        else:
            print("  ✓ Great! No unrelated pages were crawled")

        # 결과 저장
        output_data = {
            "topic": results.get("auto_discovered_topic", "auto-discovered"),
            "seed_urls": [seed_url],
            "statistics": stats,
            "pages": results["pages"],
            "links": results["links"],
        }

        if "performance" in results:
            output_data["performance"] = results["performance"]
            print("\n  ⚡ Performance Summary:")
            perf = results["performance"]
            print(f"    Total time: {perf['total_time']:.2f}s")
            print(f"    Memory usage: {perf['current_memory_mb']:.2f}MB")
            print(f"    CPU usage: {perf['cpu_percent']:.1f}%")

        with open("mock_test_auto_topic_results.json", "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        print("\n  💾 Results saved to: mock_test_auto_topic_results.json")

        print("\n" + "=" * 70)
        print("✓ Auto topic discovery test completed successfully!")
        print("=" * 70)

    except Exception as e:
        print(f"\n✗ Error during test: {e}")
        import traceback

        traceback.print_exc()

    finally:
        # 서버 종료
        print("\n🛑 Shutting down mock server...")
        await server_runner.cleanup()


if __name__ == "__main__":
    asyncio.run(test_auto_topic_discovery())
