## Vector Database Constants
VECTOR_DATABASE_COLLECTION_NAME = "interview_intelligence_engine"

## Embedding Model Constants
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"

## Groq LLM Constants (used for query expansion and HyDE)
GROQ_MODEL = "llama-3.3-70b-versatile"

## Mistral LLM Constants (used for answer generation)
MISTRAL_MODEL = "mistral-large-latest"

## Known companies tracked in the corpus
KNOWN_COMPANIES: list[str] = [
    # Big Tech
    "Amazon",
    "Google",
    "Meta",
    "Apple",
    "Microsoft",
    "NVIDIA",
    "Netflix",
    "Uber",
    "Lyft",
    "Airbnb",
    "Stripe",
    "Coinbase",
    "Twitter",
    "LinkedIn",
    "Salesforce",
    "Adobe",
    "Oracle",
    "IBM",
    "Intel",
    "Qualcomm",
    # Bioinformatics / Genomics / Life Sciences Tech
    "Illumina",
    "10x Genomics",
    "Pacific Biosciences",
    "Oxford Nanopore",
    "Veeva Systems",
    "Benchling",
    "Recursion Pharmaceuticals",
    "Schrödinger",
    "Tempus",
    "Flatiron Health",
    "Foundation Medicine",
    "Guardant Health",
    "Natera",
    "Grail",
    "DNAnexus",
    "Seven Bridges",
    "Ginkgo Bioworks",
    "Twist Bioscience",
    "Adaptive Biotechnologies",
    "23andMe",
    "Moderna",
    "BioNTech",
]