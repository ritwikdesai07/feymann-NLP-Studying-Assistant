import os
from typing import List
import numpy as np
from pydantic import BaseModel, Field
from google import genai
from google.genai import types


class KnowledgeGapReport(BaseModel):
    overall_alignment_summary: str = Field(
        description="Concise summary of how well the spoken explanation matches verified sources."
    )
    missing_concepts: List[str] = Field(
        description="Important ideas from the verified sources that were left out or glossed over."
    )
    misconceptions: List[str] = Field(
        description="Imprecise or inaccurate claims compared to the verified sources."
    )
    feynman_prompts: List[str] = Field(
        description="Targeted questions to help the learner explain weak areas simply and accurately."
    )


class FeynmanStudyAssistantV1:
    def __init__(
        self,
        gen_model: str = "gemini-3.8-flash",
        embed_model: str = "gemini-embedding-2",
    ):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("Please set GEMINI_API_KEY in your environment.")
        self.client = genai.Client(api_key=api_key)
        self.gen_model = gen_model
        self.embed_model = embed_model

    def transcribe_audio(self, audio_path: str, mime_type: str = "audio/wav") -> str:
        """Transcribe a recorded spoken explanation into plain text."""
        with open(audio_path, "rb") as audio_file:
            audio_bytes = audio_file.read()

        response = self.client.models.generate_content(
            model=self.gen_model,
            contents=[
                "Transcribe this spoken explanation accurately as plain text. "
                "Preserve technical terms, equations spoken aloud, and exact phrasing.",
                types.Part.from_bytes(data=audio_bytes, mime_type=mime_type),
            ],
        )
        return (response.text or "").strip()

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Generate L2-normalized embedding vectors for a list of strings."""
        result = self.client.models.embed_content(
            model=self.embed_model,
            contents=texts,
        )
        vectors = np.array([emb.values for emb in result.embeddings], dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.clip(norms, a_min=1e-12, a_max=None)

    @staticmethod
    def chunk_text(text: str, max_sentences: int = 2) -> List[str]:
        """Break text into short sentence groups for granular similarity matching."""
        sentences = [s.strip() for s in text.replace("\n", " ").split(".") if s.strip()]
        chunks = []
        for idx in range(0, len(sentences), max_sentences):
            chunks.append(". ".join(sentences[idx : idx + max_sentences]) + ".")
        return chunks

    def score_semantic_coverage(
        self, explanation: str, reference_passages: List[str]
    ) -> dict:
        """Compute cosine similarity between explanation chunks and verified reference passages."""
        exp_chunks = self.chunk_text(explanation, max_sentences=2)
        if not exp_chunks or not reference_passages:
            return {"mean_reference_coverage": 0.0, "passage_matches": []}

        exp_vecs = self.embed_texts(exp_chunks)
        ref_vecs = self.embed_texts(reference_passages)

        # Cosine similarity matrix: rows = reference passages, cols = explanation chunks
        similarity_matrix = ref_vecs @ exp_vecs.T

        passage_matches = []
        best_scores = []
        for r_idx, ref_passage in enumerate(reference_passages):
            best_c_idx = int(np.argmax(similarity_matrix[r_idx]))
            best_score = float(similarity_matrix[r_idx, best_c_idx])
            best_scores.append(best_score)
            passage_matches.append(
                {
                    "reference": ref_passage,
                    "best_user_chunk": exp_chunks[best_c_idx],
                    "similarity": round(best_score, 3),
                }
            )

        return {
            "mean_reference_coverage": round(float(np.mean(best_scores)), 3),
            "passage_matches": passage_matches,
        }

    def analyze_gaps(
        self, explanation: str, reference_passages: List[str]
    ) -> KnowledgeGapReport:
        """Cross-reference the explanation against verified sources to surface gaps."""
        formatted_sources = "\n".join(
            f"- Source {i + 1}: {passage}" for i, passage in enumerate(reference_passages)
        )
        prompt = f"""You are a Feynman Technique study coach.
Compare the learner's explanation against the VERIFIED REFERENCE SOURCES below.
Identify omitted concepts, inaccuracies or jargon crutches, and ask concise questions
that force simple, first-principles clarity.

VERIFIED REFERENCE SOURCES:
{formatted_sources}

LEARNER EXPLANATION:
{explanation}
"""
        response = self.client.models.generate_content(
            model=self.gen_model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=KnowledgeGapReport,
                temperature=0.2,
            ),
        )
        return KnowledgeGapReport.model_validate_json(response.text)

    def evaluate(
        self,
        reference_passages: List[str],
        audio_path: str | None = None,
        transcript: str | None = None,
        mime_type: str = "audio/wav",
    ) -> dict:
        if audio_path:
            spoken_text = self.transcribe_audio(audio_path, mime_type=mime_type)
        elif transcript:
            spoken_text = transcript
        else:
            raise ValueError("Provide either audio_path or transcript.")

        similarity = self.score_semantic_coverage(spoken_text, reference_passages)
        gaps = self.analyze_gaps(spoken_text, reference_passages)

        return {
            "transcript": spoken_text,
            "similarity": similarity,
            "gaps": gaps.model_dump(),
        }


if __name__ == "__main__":
    verified_sources = [
        "Conservation of mechanical energy states that total mechanical energy (kinetic plus potential) remains constant if only conservative forces do work.",
        "Non-conservative forces such as kinetic friction dissipate mechanical energy into thermal energy.",
        "Work equals the dot product of force and displacement over a path.",
    ]

    # Pass audio_path="my_explanation.wav" to transcribe directly from audio,
    # or pass transcript="..." to test immediately with text.
    sample_explanation = (
        "Energy is conserved when there is no friction. If there is friction, "
        "some of the energy turns into heat and slows things down."
    )

    assistant = FeynmanStudyAssistantV1()
    report = assistant.evaluate(
        reference_passages=verified_sources,
        transcript=sample_explanation,
    )

    print("Transcript:\n", report["transcript"], "\n")
    print("Mean Reference Coverage:", report["similarity"]["mean_reference_coverage"])
    for item in report["similarity"]["passage_matches"]:
        print(f"  [{item['similarity']:.3f}] Ref: {item['reference']}")
        print(f"           Closest chunk: {item['best_user_chunk']}")

    print("\nGap Analysis:")
    print("  Summary:", report["gaps"]["overall_alignment_summary"])
    print("  Missing Concepts:", report["gaps"]["missing_concepts"])
    print("  Misconceptions:", report["gaps"]["misconceptions"])
    print("  Feynman Prompts:", report["gaps"]["feynman_prompts"])