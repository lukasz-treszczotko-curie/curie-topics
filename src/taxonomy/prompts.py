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
Generate a comprehensive list of 5-15 subtopics for the given topic that are MUTUALLY EXCLUSIVE and CUMULATIVELY EXHAUSTIVE. Each subtopic must include a descriptive summary (2-4 sentences) that clearly explains its scope and boundaries.

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

## Mutual Exclusivity (CRITICAL)
Each subtopic must be DISTINCT with NO OVERLAP. Any piece of content should clearly belong to only ONE subtopic.

**Common mistakes to avoid:**
- DO NOT mix methodology (how) with task (what) or domain (where). Choose ONE dimension:
  * Either categorize by METHOD (e.g., "Deep Learning", "Statistical Models")
  * OR by TASK (e.g., "Classification", "Prediction")
  * OR by DOMAIN (e.g., "Healthcare", "Finance")
  * But NEVER mix these dimensions in the same taxonomy
- DO NOT mix temporal stages (e.g., "Early Methods" vs "Recent Advances") with conceptual categories
- DO NOT create both a general category and its specializations (e.g., "Optimization" AND "Gradient-Based Optimization")
- Example of BAD taxonomy: ["Neural Networks" (method), "Classification" (task), "Medical Diagnosis" (domain)] - these overlap!
- Example of GOOD taxonomy: ["Image Classification", "Object Detection", "Segmentation"] - all are tasks
- Example of BAD: ["Foundation Models", "Small Models", "Applications"] - mixes model types with use cases
- Example of GOOD: ["Language Models", "Vision Models", "Multimodal Models"] - all are model types

## Cumulative Exhaustiveness
Together, the subtopics must cover ALL aspects of "{topic}". No content should fall outside the subtopic categories.

## Handling Complex Topics
- For interdisciplinary topics: Choose the PRIMARY dimension that best organizes the research
- For topics with both theoretical and applied aspects: Choose EITHER theoretical divisions OR application areas, not both
- For emerging fields: Focus on current established areas rather than speculative future directions
- If a paper could genuinely fit multiple categories, your boundaries need refinement

## Quality Criteria
1. Generate between 5-15 subtopics (adjust based on topic complexity)
2. Each subtopic should be:
   - Clear and specific
   - At the same level of abstraction (avoid mixing very specific with very broad categories)
   - At the same level of granularity (e.g., don't mix "Electric Vehicles" with "Data Analysis")
   - Meaningful enough to warrant separate categorization
   - Named concisely (2-6 words)
   - Named so that its meaning does not have to be interpreted in the context of the original topic (self-explanatory)
3. Each description should:
   - Be 2-4 sentences long
   - Clearly explain what the subtopic covers
   - Explicitly define boundaries - mention what is INCLUDED and what is EXCLUDED
   - Be more expressive than the name alone
   - Help disambiguate from other subtopics by highlighting distinguishing characteristics
   - Use phrases like "focuses specifically on", "excludes", "as distinct from" to clarify scope
4. Use the provided keywords and representative works to inform your understanding of the topic scope
5. **CRITICAL RULE**: NEVER create an "Other", "Miscellaneous", "General", or catch-all category
   - Every subtopic must be a specific, well-defined area of research
   - If content doesn't fit into specific categories, create more specific subtopics instead
   - Better to have more specific subtopics than to have a vague catch-all category

# Process
1. Analyze the topic and identify the main dimensions that naturally partition it
2. Choose ONE classification dimension (method, task, domain, or component) and stick to it consistently
3. Consider what categories would best organize the representative works provided
4. BOUNDARY TEST: For each subtopic, write one sentence explicitly stating what it EXCLUDES
5. OVERLAP TEST: For each pair of subtopics, try to imagine a paper that could fit in both - if you can, revise the boundaries
6. GRANULARITY CHECK: Ensure all subtopics operate at the same level of specificity
7. Ensure all aspects of the topic are covered through specific, well-defined subtopics (no generic catch-alls)
8. FORBIDDEN TERMS CHECK: Verify that NO subtopic uses words like "Other", "Miscellaneous", "General", "Various", "Diverse", "Additional", or similar catch-all terms
9. FINAL VALIDATION: Could an expert place any paper about "{topic}" into exactly one SPECIFIC subtopic without ambiguity?

# Output Format
You must return valid JSON with the following structure:
- Top-level: a JSON object with one field called "subtopics"
- The "subtopics" field contains a list of 2-element arrays
- Each 2-element array contains: [subtopic_name, subtopic_description]
  - First element: subtopic name (string, 2-6 words)
  - Second element: subtopic description (string, 2-4 sentences)

Example with 2 subtopics:

{{
  "subtopics": [
    ["First Subtopic Name", "A detailed 2-4 sentence description explaining what this subtopic covers, its scope, and how it differs from other subtopics."],
    ["Second Subtopic Name", "Another detailed 2-4 sentence description for the second subtopic."]
  ]
}}

IMPORTANT: Each item must be a 2-element array: [name_string, description_string]. Use proper JSON formatting with double quotes and commas.

Generate the subtopics now:"""
