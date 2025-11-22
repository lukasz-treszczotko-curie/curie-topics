def get_subtopics_generation_prompt(
    domain: str,
    field: str,
    subfield: str,
    description: str,
    topic: str,
    keywords: list[str],
    representative_works: list[str],
    citation_counts: list[int],
) -> str:
    keywords_str = "\n".join(f"- {kw}" for kw in keywords)
    works_str = "\n".join(
        f"- {work} NUM OF CITATIONS: {count}"
        for work, count in zip(representative_works, citation_counts)
    )

    return f"""# Role
You are a taxonomist specializing in the field of {domain}.

# Task
Generate a comprehensive list of 5-15 subtopics for the given topic that are MUTUALLY EXCLUSIVE and CUMULATIVELY EXHAUSTIVE.

# Context
- Domain: {domain}
- Field: {field}
- Subfield: {subfield}
- Topic: {topic}
- Description: {description}

## Keywords
{keywords_str}

## Representative Works
{works_str}

# Requirements

## Mutual Exclusivity
Each subtopic must be DISTINCT with NO OVERLAP. Any piece of content should clearly belong to only ONE subtopic.

## Cumulative Exhaustiveness
Together, the subtopics must cover ALL aspects of "{topic}". No content should fall outside the subtopic categories.

## Quality Criteria
1. Generate between 5-15 subtopics (adjust based on topic complexity)
2. Each subtopic should be:
   - Clear and specific
   - At the same level of abstraction
   - Meaningful enough to warrant separate categorization
   - Named concisely (2-6 words)
3. Use the provided keywords and representative works to inform your understanding of the topic scope

# Process
1. Analyze the topic and identify the main dimensions that naturally partition it
2. Consider what categories would best organize the representative works provided
3. Ensure no overlap between categories (mutual exclusivity test)
4. Ensure all aspects of the topic are covered (exhaustiveness test)
5. Add an "Other" or "Miscellaneous" category ONLY if absolutely necessary for exhaustiveness

# Output Format
Return ONLY a JSON array of subtopic names:
["Subtopic 1", "Subtopic 2", "Subtopic 3", ...]

Generate the subtopics now:"""
