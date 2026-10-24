"""
Helper script to generate a template test.json from chunked_documents.json.
You still need to manually write the queries and answers!
"""

import json
from pathlib import Path

def build_template(chunked_docs_path: Path, output_path: Path, sample_size: int = 10):
    with open(chunked_docs_path, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    template = []
    for i, chunk in enumerate(chunks[:sample_size]):
        template.append({
            "id": f"test_{i+1:03d}",
            "query": "",  # YOU FILL THIS
            "language": "am",
            "domain": "banking",
            "category": "",  # YOU FILL THIS
            "intent": "",  # YOU FILL THIS
            "complexity": {
                "morphology_score": 0.0,  # YOU FILL THIS
                "level": ""  # YOU FILL THIS
            },
            "gold_chunks": [chunk["chunk_id"]],  # AUTO-FILLED
            "gold_answer": "",  # YOU FILL THIS
            "acceptable_answers": [],  # YOU FILL THIS
            "difficulty": "",  # YOU FILL THIS
            "metadata": {
                "source": chunk.get("document_title", "unknown"),
                "chunk_preview": chunk.get("chunk_text", "")[:200],  # HELPFUL CONTEXT
                "annotator": "human_expert"
            }
        })
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(template, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Template created at: {output_path}")
    print(f"   {len(template)} entries ready for manual annotation")

if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    build_template(
        chunked_docs_path=PROJECT_ROOT / "data" / "processed" / "chunked_documents.json",
        output_path=PROJECT_ROOT / "src" / "evaluation" / "dataset" / "test_template.json",
        sample_size=50
    )