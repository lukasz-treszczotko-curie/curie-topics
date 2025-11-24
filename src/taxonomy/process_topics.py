import argparse
import asyncio
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import logfire
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

from src.taxonomy.api_utils import get_topic_metadata
from src.taxonomy.prompts import get_subtopics_generation_prompt

from src.logging_config import setup_logging
import logging


load_dotenv()
logfire.configure(token=os.getenv("LOGFIRE_TOKEN"))
logfire.instrument_pydantic_ai()

setup_logging(level="INFO")
logger = logging.getLogger(__name__)

OLMO3_7B_THINK = "allenai/olmo-3-7b-think"
SONNET_4_5 = "anthropic/claude-sonnet-4.5"
GEMINI_2_5_FLASH = "google/gemini-2.5-flash-preview-09-2025"


@dataclass
class Category:
    """Represents subfield, field, or domain"""

    id: str
    display_name: str


@dataclass
class Topic:
    id: str
    display_name: str
    description: str
    keywords: List[str]
    subfield: Category
    field: Category
    domain: Category
    subtopics_model: str
    most_cited_works: Optional[List[str]] = None
    most_cited_works_citation_counts: Optional[List[int]] = None
    subtopics: Optional[List[dict]] = (
        None  # List of dicts with 'name' and 'description'
    )

    @classmethod
    def from_dict(cls, data: dict) -> "Topic":
        """Create a Topic instance from a dictionary"""
        return cls(
            id=data["id"],
            display_name=data["display_name"],
            description=data["description"],
            keywords=data["keywords"],
            subfield=Category(**data["subfield"]),
            field=Category(**data["field"]),
            domain=Category(**data["domain"]),
            subtopics_model=data.get("subtopics_model"),
            most_cited_works=data.get("most_cited_works"),
            most_cited_works_citation_counts=data.get(
                "most_cited_works_citation_counts"
            ),
            subtopics=data.get("subtopics"),
        )

    def to_dict(self) -> dict:
        """Convert Topic instance to a dictionary"""
        return {
            "id": self.id,
            "display_name": self.display_name,
            "description": self.description,
            "keywords": self.keywords,
            "subfield": {
                "id": self.subfield.id,
                "display_name": self.subfield.display_name,
            },
            "field": {"id": self.field.id, "display_name": self.field.display_name},
            "domain": {"id": self.domain.id, "display_name": self.domain.display_name},
            "subtopics_model": self.subtopics_model,
            "most_cited_works": self.most_cited_works,
            "most_cited_works_citation_counts": self.most_cited_works_citation_counts,
            "subtopics": self.subtopics,
        }


class SubtopicsOutput(BaseModel):
    """Output model for subtopics generation"""
    subtopics: List[tuple[str, str]]  # List of (name, description) tuples


def load_topics_from_list(file_path: str | Path) -> dict[str, Topic]:
    """Load topics from a JSON file (list format) and return as dict with topic IDs as keys"""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return {topic_data["id"]: Topic.from_dict(topic_data) for topic_data in data}


def load_topics_from_dict(file_path: str | Path) -> dict[str, Topic]:
    """Load topics from a JSON file (dict format) with topic IDs as keys"""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return {
        topic_id: Topic.from_dict(topic_data) for topic_id, topic_data in data.items()
    }


def save_topics(topics_dict: dict[str, Topic], file_path: str | Path) -> None:
    """Save topics dict to a JSON file with topic IDs as keys"""
    data = {topic_id: topic.to_dict() for topic_id, topic in topics_dict.items()}
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(topics_dict)} topics to {file_path}")


async def process_topic(
    topic: Topic,
    agent: Agent,
    subtopics_model_name: str,
    index: int,
    total: int,
) -> Topic:
    """Process a single topic asynchronously"""
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Processing {index}/{total}: {topic.display_name}")
    logger.info(f"{'=' * 60}")

    # Add most cited works and citation counts if not already present
    if topic.most_cited_works is None:
        (
            topic.most_cited_works,
            topic.most_cited_works_citation_counts,
        ) = await get_topic_metadata(topic.id)

    logger.info(f"   ID: {topic.id}")
    logger.info(f"   Field: {topic.field.display_name}")
    logger.info(f"   Domain: {topic.domain.display_name}")
    logger.info(f"   Subfield: {topic.subfield.display_name}")
    logger.info(f"   Keywords: {', '.join(topic.keywords)}")

    # Generate subtopics
    prompt = get_subtopics_generation_prompt(
        topic=topic.display_name,
        keywords=topic.keywords,
        representative_works=topic.most_cited_works,
        citation_counts=topic.most_cited_works_citation_counts,
        domain=topic.domain.display_name,
        field=topic.field.display_name,
        subfield=topic.subfield.display_name,
        description=topic.description,
    )

    result = await agent.run(prompt)
    # Convert tuples to dicts for storage
    topic.subtopics = [
        {"name": name, "description": desc}
        for name, desc in result.output.subtopics
    ]
    topic.subtopics_model = subtopics_model_name

    logger.info(
        f"   Generated {len(topic.subtopics)} subtopics for {topic.display_name}"
    )

    return topic


async def main_async(
    batch_size: int = 10, model_name: str = "allenai/olmo-3-7b-instruct"
):
    openrouter_provider = OpenAIProvider(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
    )

    subtopics_model_name = model_name

    model = OpenAIModel(
        subtopics_model_name,
        provider=openrouter_provider,
    )

    agent = Agent(
        model=model,
        output_type=SubtopicsOutput,
        retries=3,  # Retry up to 3 times on validation errors
    )

    # Get the path to the JSON files
    current_dir = Path(__file__).parent
    input_file = current_dir / "openalex_topics.json"

    # Create a sanitized model name for the filename
    model_name_safe = subtopics_model_name.replace("/", "_").replace(":", "_")
    progress_file = (
        current_dir / f"openalex_topics_with_subtopics_{model_name_safe}.json"
    )

    # Load topics - progress file uses dict format, input file uses list format
    if progress_file.exists():
        logger.info(f"Loading progress from {progress_file}")
        topics_dict = load_topics_from_dict(progress_file)
    else:
        logger.info(f"Loading topics from {input_file}")
        topics_dict = load_topics_from_list(input_file)

    # ========== TEMPORARY FILTER: Machine Learning Topics Only ==========
    # TODO: Remove this filter when ready to process all topics
    ml_keywords = [
        "machine learning",
        "deep learning",
        "neural network",
        "artificial intelligence",
        "reinforcement learning",
        "supervised learning",
        "unsupervised learning",
        "convolutional",
        "recurrent",
        "transformer",
        "attention mechanism",
        "natural language processing",
        "computer vision",
        "generative model",
    ]

    def is_ml_topic(topic: Topic) -> bool:
        """Check if a topic is related to machine learning"""
        text = f"{topic.display_name} {topic.description} {' '.join(topic.keywords)}".lower()
        return any(keyword in text for keyword in ml_keywords)

    # Filter to only ML topics
    all_topics_count = len(topics_dict)
    topics_dict = {tid: t for tid, t in topics_dict.items() if is_ml_topic(t)}
    logger.info(
        f"FILTER ACTIVE: Reduced from {all_topics_count} to {len(topics_dict)} ML-related topics"
    )
    # =====================================================================

    # Filter topics that need processing (those without subtopics)
    topics_to_process = [t for t in topics_dict.values() if t.subtopics is None]
    topics_already_processed = len(topics_dict) - len(topics_to_process)

    logger.info(f"Model: {subtopics_model_name}")
    logger.info(f"Output file: {progress_file}")
    logger.info(f"Loaded {len(topics_dict)} topics total")
    logger.info(f"Already processed: {topics_already_processed}")
    logger.info(f"To process: {len(topics_to_process)}")
    logger.info(f"Batch size: {batch_size}")

    # Process topics in batches
    total_topics = len(topics_to_process)
    for batch_idx in range(0, total_topics, batch_size):
        batch = topics_to_process[batch_idx : batch_idx + batch_size]
        batch_num = batch_idx // batch_size + 1
        total_batches = (total_topics + batch_size - 1) // batch_size

        logger.info(f"\n{'#' * 60}")
        logger.info(
            f"BATCH {batch_num}/{total_batches} - Processing {len(batch)} topics in parallel"
        )
        logger.info(f"{'#' * 60}")

        # Process all topics in this batch in parallel
        tasks = [
            process_topic(
                topic=topic,
                agent=agent,
                subtopics_model_name=subtopics_model_name,
                index=batch_idx + i + 1,
                total=total_topics,
            )
            for i, topic in enumerate(batch)
        ]

        # Wait for all tasks in the batch to complete
        processed_topics = await asyncio.gather(*tasks)

        # Update the topics dict with processed topics
        for topic in processed_topics:
            topics_dict[topic.id] = topic

        # Save progress after each batch
        save_topics(topics_dict, progress_file)
        logger.info(
            f">>> Checkpoint: Saved progress after batch {batch_num} ({batch_idx + len(batch)} topics)"
        )

    logger.info(f"\n{'=' * 60}")
    logger.info("All topics processed!")
    logger.info(f"Final results saved to {progress_file}")
    logger.info(f"{'=' * 60}")


def main():
    """Entry point that runs the async main function"""
    parser = argparse.ArgumentParser(
        description="Process topics and generate subtopics using AI"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of topics to process in parallel (default: 10)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="allenai/olmo-3-7b-instruct",
        help="Model to use for subtopic generation (default: allenai/olmo-3-7b-instruct)",
    )
    args = parser.parse_args()

    logger.info(f"Starting topic processing")
    logger.info(f"Model: {args.model}")
    logger.info(f"Batch size: {args.batch_size}")
    asyncio.run(main_async(batch_size=args.batch_size, model_name=args.model))


if __name__ == "__main__":
    main()
