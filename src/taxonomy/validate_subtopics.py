import argparse
import asyncio
import json
import logging
import os
import random
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import logfire
from dotenv import load_dotenv
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

from src.logging_config import setup_logging
from src.taxonomy.process_topics import Topic, load_topics_from_dict

load_dotenv()
logfire.configure(token=os.getenv("LOGFIRE_TOKEN"))
logfire.instrument_pydantic_ai()

setup_logging(level="INFO")
logger = logging.getLogger(__name__)


class ValidationResult(BaseModel):
    quality_score: float  # 0.0 to 1.0
    critique: str
    mutual_exclusivity_score: float  # 0.0 to 1.0
    exhaustiveness_score: float  # 0.0 to 1.0
    clarity_score: float  # 0.0 to 1.0
    abstraction_level_score: float  # 0.0 to 1.0


def get_validation_prompt(topic: Topic) -> str:
    """Generate a validation prompt for the subtopics"""
    keywords_str = "\n".join(f"- {kw}" for kw in topic.keywords)
    works_str = "\n".join(
        f"- {work} (Citations: {count})"
        for work, count in zip(
            topic.most_cited_works or [], topic.most_cited_works_citation_counts or []
        )
    )
    # Format subtopics with name and description
    subtopics_list = []
    for i, st in enumerate(topic.subtopics or []):
        if isinstance(st, dict):
            name = st.get("name", "")
            description = st.get("description", "")
            subtopics_list.append(f"{i + 1}. **{name}**: {description}")
        else:
            # Fallback for old format (just strings)
            subtopics_list.append(f"{i + 1}. {st}")
    subtopics_str = "\n".join(subtopics_list)

    return f"""# Role
You are an expert taxonomist evaluating the quality of subtopic categorizations in the field of {topic.domain.display_name}.

# Task
Evaluate the following list of subtopics (with descriptions) generated for a given topic. Assess their quality based on specific criteria and provide scores and critique.

# Topic Information
- **Domain**: {topic.domain.display_name}
- **Field**: {topic.field.display_name}
- **Subfield**: {topic.subfield.display_name}
- **Topic**: {topic.display_name}
- **Description**: {topic.description}

## Keywords
{keywords_str}

## Representative Works
{works_str}

## Generated Subtopics
{subtopics_str}

# Evaluation Criteria

## 1. Mutual Exclusivity (0.0 - 1.0)
- Score 1.0: Each subtopic is COMPLETELY DISTINCT with NO OVERLAP. Any piece of content clearly belongs to ONLY ONE subtopic.
- Score 0.5: Some minor overlaps exist, but most subtopics are distinct.
- Score 0.0: Significant overlaps; unclear boundaries between multiple subtopics.

## 2. Cumulative Exhaustiveness (0.0 - 1.0)
- Score 1.0: The subtopics collectively cover ALL aspects of "{topic.display_name}". No important content falls outside the categories.
- Score 0.5: Most aspects are covered, but some minor gaps exist.
- Score 0.0: Major gaps; significant aspects of the topic are not covered.

## 3. Clarity and Specificity (0.0 - 1.0)
- Score 1.0: Each subtopic name is CLEAR, SPECIFIC, and CONCISE (2-6 words). Names are unambiguous. Descriptions (2-4 sentences) clearly explain scope and boundaries.
- Score 0.5: Most subtopics are clear, but some names or descriptions are vague or overly broad.
- Score 0.0: Many subtopics have unclear names, ambiguous descriptions, or poorly defined scope.

## 4. Abstraction Level Consistency (0.0 - 1.0)
- Score 1.0: ALL subtopics are at the SAME LEVEL of abstraction and granularity.
- Score 0.5: Most subtopics are at similar levels, with minor inconsistencies.
- Score 0.0: Subtopics mix different levels of abstraction (e.g., very broad categories mixed with very specific ones).

## 5. Overall Quality Score (0.0 - 1.0)
- Weighted average considering all criteria
- Consider whether the number of subtopics (5-15) is appropriate for the topic complexity
- Consider whether the subtopics meaningfully partition the topic space

# Critique Guidelines
Provide constructive feedback addressing:
1. Specific overlaps between subtopics (if any)
2. Missing aspects of the topic (if any)
3. Unclear or poorly named subtopics (if any)
4. Quality of descriptions - are they informative, clear, and help disambiguate? (if any issues)
5. Inconsistencies in abstraction level (if any)
6. Whether the number of subtopics is appropriate
7. Suggestions for improvement

# Output Format
Provide your evaluation as a structured response with scores and detailed critique.
"""


async def validate_topic(
    topic: Topic,
    agent: Agent,
    index: int,
    total: int,
) -> tuple[Topic, ValidationResult]:
    """Validate subtopics for a single topic"""
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Validating {index}/{total}: {topic.display_name}")
    logger.info(f"{'=' * 60}")

    prompt = get_validation_prompt(topic)
    result = await agent.run(prompt)

    logger.info(f"   Overall Quality: {result.output.quality_score:.2f}")
    logger.info(f"   Mutual Exclusivity: {result.output.mutual_exclusivity_score:.2f}")
    logger.info(f"   Exhaustiveness: {result.output.exhaustiveness_score:.2f}")
    logger.info(f"   Clarity: {result.output.clarity_score:.2f}")
    logger.info(f"   Abstraction Level: {result.output.abstraction_level_score:.2f}")

    return (topic, result.output)


async def main_async(
    input_file: Path,
    output_file: Path,
    num_samples: int,
    batch_size: int = 5,
):
    """Main validation function"""
    # Setup OpenRouter provider
    openrouter_provider = OpenAIProvider(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
    )

    model_name = "google/gemini-3-pro-preview"

    model = OpenAIModel(
        model_name,
        provider=openrouter_provider,
    )

    agent = Agent(
        model=model,
        output_type=ValidationResult,
    )

    # Load topics
    logger.info(f"Loading topics from {input_file}")
    topics_dict = load_topics_from_dict(input_file)

    # Filter topics that have subtopics
    topics_with_subtopics = [
        t
        for t in topics_dict.values()
        if t.subtopics is not None and len(t.subtopics) > 0
    ]

    logger.info(f"Found {len(topics_with_subtopics)} topics with subtopics")

    # Sample random topics
    if num_samples > len(topics_with_subtopics):
        logger.warning(
            f"Requested {num_samples} samples, but only {len(topics_with_subtopics)} available. Using all."
        )
        num_samples = len(topics_with_subtopics)

    sampled_topics = random.sample(topics_with_subtopics, num_samples)
    logger.info(f"Randomly sampled {num_samples} topics for validation")

    # Validate in batches
    results = []
    total_topics = len(sampled_topics)

    for batch_idx in range(0, total_topics, batch_size):
        batch = sampled_topics[batch_idx : batch_idx + batch_size]
        batch_num = batch_idx // batch_size + 1
        total_batches = (total_topics + batch_size - 1) // batch_size

        logger.info(f"\n{'#' * 60}")
        logger.info(
            f"BATCH {batch_num}/{total_batches} - Validating {len(batch)} topics in parallel"
        )
        logger.info(f"{'#' * 60}")

        # Validate all topics in this batch in parallel
        tasks = [
            validate_topic(
                topic=topic,
                agent=agent,
                index=batch_idx + i + 1,
                total=total_topics,
            )
            for i, topic in enumerate(batch)
        ]

        batch_results = await asyncio.gather(*tasks)
        results.extend(batch_results)

    # Compute statistics
    avg_quality = sum(r[1].quality_score for r in results) / len(results)
    avg_mutual_exclusivity = sum(r[1].mutual_exclusivity_score for r in results) / len(
        results
    )
    avg_exhaustiveness = sum(r[1].exhaustiveness_score for r in results) / len(results)
    avg_clarity = sum(r[1].clarity_score for r in results) / len(results)
    avg_abstraction = sum(r[1].abstraction_level_score for r in results) / len(results)

    logger.info(f"\n{'=' * 60}")
    logger.info("VALIDATION SUMMARY")
    logger.info(f"{'=' * 60}")
    logger.info(f"Total topics validated: {len(results)}")
    logger.info(f"Average Quality Score: {avg_quality:.3f}")
    logger.info(f"Average Mutual Exclusivity: {avg_mutual_exclusivity:.3f}")
    logger.info(f"Average Exhaustiveness: {avg_exhaustiveness:.3f}")
    logger.info(f"Average Clarity: {avg_clarity:.3f}")
    logger.info(f"Average Abstraction Level: {avg_abstraction:.3f}")

    # Save results
    timestamp = datetime.now().isoformat()
    output_data = {
        "metadata": {
            "timestamp": timestamp,
            "model": model_name,
            "num_samples": num_samples,
            "batch_size": batch_size,
        },
        "summary": {
            "num_validated": len(results),
            "avg_quality_score": avg_quality,
            "avg_mutual_exclusivity_score": avg_mutual_exclusivity,
            "avg_exhaustiveness_score": avg_exhaustiveness,
            "avg_clarity_score": avg_clarity,
            "avg_abstraction_level_score": avg_abstraction,
        },
        "results": [
            {
                "topic_id": topic.id,
                "topic_name": topic.display_name,
                "subtopics": topic.subtopics,
                "validation": {
                    "quality_score": validation.quality_score,
                    "mutual_exclusivity_score": validation.mutual_exclusivity_score,
                    "exhaustiveness_score": validation.exhaustiveness_score,
                    "clarity_score": validation.clarity_score,
                    "abstraction_level_score": validation.abstraction_level_score,
                    "critique": validation.critique,
                },
            }
            for topic, validation in results
        ],
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    logger.info(f"Results saved to {output_file}")
    logger.info(f"{'=' * 60}")


def main():
    """Entry point"""
    parser = argparse.ArgumentParser(
        description="Validate generated subtopics for a sample of topics"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the input JSON file with subtopics",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save validation results (default: src/taxonomy/validation_results_TIMESTAMP.json)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of random topics to validate (default: 10)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="Number of topics to validate in parallel (default: 5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (optional)",
    )

    args = parser.parse_args()

    # Set random seed if provided
    if args.seed is not None:
        random.seed(args.seed)
        logger.info(f"Random seed set to: {args.seed}")

    input_file = Path(args.input)

    # Generate timestamped filename if no output specified
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = Path(f"src/taxonomy/validation_results_{timestamp}.json")
    else:
        output_file = Path(args.output)

    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
        return

    logger.info(f"Output will be saved to: {output_file}")

    asyncio.run(
        main_async(
            input_file=input_file,
            output_file=output_file,
            num_samples=args.num_samples,
            batch_size=args.batch_size,
        )
    )


if __name__ == "__main__":
    main()
