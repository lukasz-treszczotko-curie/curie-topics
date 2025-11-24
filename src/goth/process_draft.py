import argparse
import asyncio
import json
import os
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import List, Optional

import logfire
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

from src.goth.prompts import get_claim_extraction_prompt
from src.logging_config import setup_logging
import logging


load_dotenv()
logfire.configure(token=os.getenv("LOGFIRE_TOKEN"))
logfire.instrument_pydantic_ai()

setup_logging(level="INFO")
logger = logging.getLogger(__name__)


class RelationType(str, Enum):
    """Predefined relation types for claim dependencies"""
    SUPPORTS = "SUPPORTS"
    CONTRADICTS = "CONTRADICTS"
    REQUIRES = "REQUIRES"
    ELABORATES = "ELABORATES"
    CAUSES = "CAUSES"
    IMPLIES = "IMPLIES"
    MEASURES = "MEASURES"
    CONDITIONAL_ON = "CONDITIONAL_ON"
    PRECEDES = "PRECEDES"
    MECHANISM_FOR = "MECHANISM_FOR"


class Claim(BaseModel):
    """A single atomic claim (node in the knowledge graph)"""
    id: str = Field(description="Unique identifier for the claim (integer as string, e.g., '1', '2', '3')")
    text: str = Field(description="The atomic, verifiable claim statement")


class Relation(BaseModel):
    """A relation between two claims (edge in the knowledge graph)"""
    source: str = Field(description="ID of the source claim")
    target: str = Field(description="ID of the target claim (or 'core_hypothesis')")
    relation_type: RelationType = Field(description="Type of relationship between claims")


class ClaimExtractionOutput(BaseModel):
    """Output model for claim extraction as a knowledge graph"""
    core_hypothesis: Optional[str] = Field(
        default=None,
        description="The core hypothesis being tested, if present"
    )
    claims: List[Claim] = Field(
        description="List of atomic claims as graph entities/nodes"
    )
    relations: List[Relation] = Field(
        default_factory=list,
        description="List of relations between claims as graph edges"
    )


async def process_draft(
    draft_text: str,
    model_name: str = "anthropic/claude-sonnet-4.5"
) -> ClaimExtractionOutput:
    """Process a draft text and extract claims using AI"""

    openrouter_provider = OpenAIProvider(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
    )

    model = OpenAIModel(
        model_name,
        provider=openrouter_provider,
    )

    agent = Agent(
        model=model,
        output_type=ClaimExtractionOutput,
        retries=3,  # Retry up to 3 times on validation errors
    )

    logger.info(f"Processing draft with model: {model_name}")
    logger.info(f"Draft length: {len(draft_text)} characters")

    # Generate the prompt
    prompt = get_claim_extraction_prompt(draft_text)

    # Run the agent
    result = await agent.run(prompt)

    logger.info(f"{'=' * 60}")
    logger.info(f"Extracted core hypothesis: {result.output.core_hypothesis}")
    logger.info(f"Extracted {len(result.output.claims)} claims (entities)")
    logger.info(f"Extracted {len(result.output.relations)} relations (edges)")
    logger.info(f"{'=' * 60}")

    return result.output


async def main_async(
    input_file: Path,
    output_file: Optional[Path] = None,
    model_name: str = "anthropic/claude-sonnet-4.5"
):
    """Main async function to process draft and save results"""

    # Read the input draft
    logger.info(f"Reading draft from {input_file}")
    with open(input_file, "r", encoding="utf-8") as f:
        draft_text = f.read()

    if not draft_text.strip():
        logger.warning("Input file is empty!")
        return

    # Process the draft
    result = await process_draft(draft_text, model_name)

    # Prepare output - convert Pydantic models to dicts for JSON serialization
    output_data = {
        "input_file": str(input_file),
        "model": model_name,
        "timestamp": datetime.now().isoformat(),
        "core_hypothesis": result.core_hypothesis,
        "claims": [claim.model_dump() for claim in result.claims],
        "relations": [relation.model_dump() for relation in result.relations],
        "stats": {
            "num_claims": len(result.claims),
            "num_relations": len(result.relations),
            "relation_types": {
                rel_type.value: sum(1 for r in result.relations if r.relation_type == rel_type)
                for rel_type in RelationType
                if any(r.relation_type == rel_type for r in result.relations)
            }
        }
    }

    # Determine output file path
    if output_file is None:
        # Create output filename based on input and model
        model_name_safe = model_name.replace("/", "_").replace(":", "_")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = input_file.parent / f"claims_{input_file.stem}_{model_name_safe}_{timestamp}.json"

    # Save results
    logger.info(f"Saving results to {output_file}")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    logger.info(f"\n{'=' * 60}")
    logger.info("KNOWLEDGE GRAPH EXTRACTION COMPLETE")
    logger.info(f"{'=' * 60}")
    logger.info(f"Results saved to: {output_file}")
    logger.info(f"\nGraph Statistics:")
    logger.info(f"  - Nodes (Claims): {output_data['stats']['num_claims']}")
    logger.info(f"  - Edges (Relations): {output_data['stats']['num_relations']}")
    if output_data['stats']['relation_types']:
        logger.info(f"\nRelation Types Distribution:")
        for rel_type, count in output_data['stats']['relation_types'].items():
            logger.info(f"  - {rel_type}: {count}")
    logger.info(f"{'=' * 60}")


def main():
    """Entry point that runs the async main function"""
    parser = argparse.ArgumentParser(
        description="Extract claims from scientific draft using AI"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="src/goth/sample_draft.md",
        help="Path to input draft file (default: src/goth/sample_draft.md)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to output JSON file (default: auto-generated in same directory)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="anthropic/claude-sonnet-4.5",
        help="Model to use for claim extraction (default: anthropic/claude-sonnet-4.5)",
    )
    args = parser.parse_args()

    # Convert paths to Path objects
    input_file = Path(args.input)
    output_file = Path(args.output) if args.output else None

    if not input_file.exists():
        logger.error(f"Input file does not exist: {input_file}")
        return

    logger.info(f"Starting claim extraction")
    logger.info(f"Model: {args.model}")
    logger.info(f"Input: {input_file}")

    asyncio.run(main_async(
        input_file=input_file,
        output_file=output_file,
        model_name=args.model
    ))


if __name__ == "__main__":
    main()
