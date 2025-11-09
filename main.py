"""
Main execution script for Intelligent Web Crawler
"""

import asyncio
import argparse
import json
import logging
from pathlib import Path

from crawler import (
    IntelligentCrawler,
    ContentExtractor,
    NLPProcessor,
    GraphManager,
)
from crawler.crawler_engine import CrawlConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def setup_argparser() -> argparse.ArgumentParser:
    """Setup command line argument parser"""
    parser = argparse.ArgumentParser(
        description="Intelligent Web Crawler with Semantic Analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("seed_urls", nargs="+", help="Starting URLs to crawl")

    parser.add_argument(
        "--topic",
        type=str,
        required=True,
        help='Topic to filter pages by (e.g., "인공지능 기술")',
    )

    parser.add_argument(
        "--max-pages", type=int, default=50, help="Maximum number of pages to crawl"
    )

    parser.add_argument("--max-depth", type=int, default=3, help="Maximum crawl depth")

    parser.add_argument(
        "--request-delay",
        type=float,
        default=1.0,
        help="Delay between requests to same domain (seconds)",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="crawl_results.json",
        help="Output file for results",
    )

    parser.add_argument(
        "--embedding-model",
        type=str,
        default="jhgan/ko-sroberta-multitask",
        help="Sentence transformer model for embeddings",
    )

    parser.add_argument(
        "--no-robots-txt", action="store_true", help="Disable robots.txt checking"
    )

    parser.add_argument(
        "--relevance-threshold",
        type=float,
        default=0.5,
        help="Minimum topic relevance score (0-1)",
    )

    parser.add_argument(
        "--export-graph",
        type=str,
        choices=["gexf", "graphml", "gml"],
        help="Export graph in specified format",
    )

    return parser


async def run_crawler(args):
    """Run the intelligent crawler"""
    logger.info("Initializing crawler components...")

    # Create configuration
    config = CrawlConfig(
        max_depth=args.max_depth,
        max_pages=args.max_pages,
        request_delay=args.request_delay,
        respect_robots_txt=not args.no_robots_txt,
    )

    # Initialize components
    content_extractor = ContentExtractor()
    nlp_processor = NLPProcessor(embedding_model=args.embedding_model)
    graph_manager = GraphManager()
    crawler = IntelligentCrawler(config)

    logger.info(f"Starting crawl with topic: '{args.topic}'")
    logger.info(f"Seed URLs: {args.seed_urls}")
    logger.info(f"Max pages: {args.max_pages}, Max depth: {args.max_depth}")

    # Run crawler
    try:
        results = await crawler.crawl(
            seed_urls=args.seed_urls,
            topic=args.topic,
            content_extractor=content_extractor,
            nlp_processor=nlp_processor,
            graph_manager=graph_manager,
            relevance_threshold=args.relevance_threshold,
        )

        logger.info("Crawl completed!")

        # Print statistics
        stats = results["statistics"]
        logger.info(f"\n{'=' * 50}")
        logger.info("CRAWL STATISTICS")
        logger.info(f"{'=' * 50}")
        logger.info(f"Pages crawled: {stats['num_pages']}")
        logger.info(f"Links found: {stats['num_links']}")
        logger.info(f"Internal links: {stats.get('internal_links', 0)}")
        logger.info(f"External links: {stats.get('external_links', 0)}")
        logger.info(
            f"Average topic relevance: {stats.get('avg_topic_relevance', 0):.3f}"
        )
        logger.info(f"Graph density: {stats.get('density', 0):.3f}")
        logger.info(f"{'=' * 50}\n")

        # Show top relevant pages
        logger.info("TOP 10 MOST RELEVANT PAGES:")
        for i, page in enumerate(results["pages"][:10], 1):
            logger.info(f"{i}. [{page['topic_relevance']:.3f}] {page['title']}")
            logger.info(f"   URL: {page['url']}")
            logger.info(f"   Summary: {page['summary'][:100]}...")
            logger.info("")

        # Prepare output data (exclude embedding vectors for JSON serialization)
        output_data = {
            "topic": args.topic,
            "seed_urls": args.seed_urls,
            "config": {
                "max_pages": args.max_pages,
                "max_depth": args.max_depth,
                "relevance_threshold": args.relevance_threshold,
            },
            "statistics": stats,
            "pages": [
                {k: v for k, v in page.items() if k != "embedding"}
                for page in results["pages"]
            ],
            "links": results["links"],
        }

        # Save results to JSON
        output_path = Path(args.output)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        logger.info(f"Results saved to: {output_path}")

        # Export graph if requested
        if args.export_graph:
            graph_filename = output_path.stem + f"_graph.{args.export_graph}"
            graph_data = graph_manager.export_graph(format=args.export_graph)
            with open(graph_filename, "w", encoding="utf-8") as f:
                f.write(graph_data)
            logger.info(f"Graph exported to: {graph_filename}")

        return results

    except Exception as e:
        logger.error(f"Crawl failed: {e}", exc_info=True)
        raise


def main():
    """Main entry point"""
    parser = setup_argparser()
    args = parser.parse_args()

    # Run async crawler
    asyncio.run(run_crawler(args))


if __name__ == "__main__":
    main()
