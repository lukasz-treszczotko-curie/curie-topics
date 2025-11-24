def get_claim_extraction_prompt(text: str) -> str:
    return f"""# Role
You are an expert Scientific Knowledge Engineer and Fact-Checker. Your task is to extract **atomic, verifiable claims** from a scientific text and map their **interdependencies** as a knowledge graph, ensuring all scientific entities are precise and "grounding-ready" for ontology linking.

# Objective
Transform the text into a structured knowledge graph with:
1. **Claims as Entities (Nodes)**: Atomic, verifiable statements
2. **Dependencies as Relations (Edges)**: Logical, causal, and evidential connections between claims

You must resolve not just linguistic ambiguities (pronouns), but also **scientific implicit constraints** (e.g., "the organism" → "*Saccharomyces cerevisiae*").

# Critical Rules for Claim Extraction

### 1. Scientific Precision & Grounding (From Reder et al.)
- **Resolve Shorthand:** Scientists often use implicit constraints. You must replace generic terms with specific, canonical entities found in the context.
  - *Bad:* "Yeast growth was inhibited."
  - *Good:* "Saccharomyces cerevisiae growth was inhibited."
  - *Bad:* "The protein was phosphorylated."
  - *Good:* "The Pah1 protein was phosphorylated."
- **Target Categories:** Pay special attention to resolving entities related to **Taxa, Genes/Proteins, Biological Components, and Chemicals**.

### 2. Hypothesis vs. Finding (From Reder et al.)
- Distinguish between the **Core Hypothesis** (the main concept being tested) and **Empirical Findings** (the results observed).
- Isolate the Core Hypothesis as a single, explicit sentence if present.

### 3. Atomic & Self-Contained (From Claimify)
- Each claim must be independently understandable.
- Resolve all pronouns ("it", "they") and relative references ("the latter").
- **Context Preservation:** Do not strip qualifying conditions (e.g., "in the absence of glucose").

### 4. Verifiability & Ambiguity (From Claimify)
- Extract only factual statements. Exclude opinions ("This is an exciting result").
- If a sentence is scientifically ambiguous (e.g., multiple possible proteins could be "the target"), mark it as ambiguous.

# Relation Types (Predefined Ontology)

Use these predefined relation types to connect claims:

1. **SUPPORTS** - One claim provides evidence for another claim
2. **CONTRADICTS** - One claim contradicts or refutes another claim
3. **REQUIRES** - One claim logically depends on another claim being true (prerequisite)
4. **ELABORATES** - One claim provides additional detail or specification about another
5. **CAUSES** - One claim describes a causal mechanism for another (X causes Y)
6. **IMPLIES** - One claim logically implies another claim
7. **MEASURES** - One claim describes a measurement or observation method for another
8. **CONDITIONAL_ON** - One claim is true only under conditions stated in another claim
9. **PRECEDES** - Temporal or procedural ordering (X happens before Y)
10. **MECHANISM_FOR** - One claim describes the molecular/mechanistic basis for another

# Process
1. **Scan** for the "Core Hypothesis" and extract it first.
2. **Identify** scientific entities and resolve them to their full names (canonicalization).
3. **Decompose** complex sentences into atomic claims.
4. **Verify** that every claim is entailed by the source text.
5. **Map Dependencies**: Identify relationships between claims using the predefined relation types.
6. **Assign IDs**: Number claims sequentially starting from 1 (use integers as strings: "1", "2", "3", etc.)

# Output Format
Return a JSON object with three keys:
- `"core_hypothesis"` (string, null if not found)
- `"claims"` (list of objects with "id" and "text")
- `"relations"` (list of objects with "source", "target", and "relation_type")

## Example
**Input Text:**
"We postulated that the kinase regulates lipid synthesis. Our data showed that Hsl1 phosphorylates Pah1, inhibiting its phosphatase activity in yeast. This phosphorylation event was measured using Western blot analysis. The inhibition of Pah1 leads to reduced diacylglycerol production."

**Output:**
{{
  "core_hypothesis": "The Hsl1 kinase regulates lipid synthesis via phosphorylation of Pah1 in Saccharomyces cerevisiae.",
  "claims": [
    {{
      "id": "1",
      "text": "Hsl1 is a protein kinase."
    }},
    {{
      "id": "2",
      "text": "Hsl1 phosphorylates the Pah1 protein in Saccharomyces cerevisiae."
    }},
    {{
      "id": "3",
      "text": "Pah1 has phosphatase activity."
    }},
    {{
      "id": "4",
      "text": "Phosphorylation by Hsl1 inhibits the phosphatase activity of Pah1."
    }},
    {{
      "id": "5",
      "text": "The phosphorylation of Pah1 by Hsl1 was measured using Western blot analysis."
    }},
    {{
      "id": "6",
      "text": "Inhibition of Pah1 phosphatase activity leads to reduced diacylglycerol production."
    }}
  ],
  "relations": [
    {{
      "source": "1",
      "target": "2",
      "relation_type": "REQUIRES"
    }},
    {{
      "source": "2",
      "target": "4",
      "relation_type": "CAUSES"
    }},
    {{
      "source": "3",
      "target": "4",
      "relation_type": "REQUIRES"
    }},
    {{
      "source": "5",
      "target": "2",
      "relation_type": "MEASURES"
    }},
    {{
      "source": "4",
      "target": "6",
      "relation_type": "MECHANISM_FOR"
    }},
    {{
      "source": "6",
      "target": "core_hypothesis",
      "relation_type": "SUPPORTS"
    }}
  ]
}}

# Input Text:
{text}
"""
