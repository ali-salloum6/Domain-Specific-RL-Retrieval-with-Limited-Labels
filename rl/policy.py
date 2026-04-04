import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

# Must match chunking used for cached doc tensors (filename + on-disk metadata).
RL_CHUNK_WORDS = 250
RL_CHUNK_OVERLAP = 50


def rl_disk_chunk_cache_path(repo_root: str | Path, model_name: str, max_chunks: int) -> str:
    """Stable path so different runs do not clobber or reuse incompatible caches."""
    root = Path(repo_root)
    safe = model_name.replace("/", "_").replace(" ", "_")
    name = (
        f"doc_chunks_{safe}_mc{max_chunks}_w{RL_CHUNK_WORDS}_o{RL_CHUNK_OVERLAP}.pt"
    )
    return str(root / "setup" / "data" / "rl_cache" / name)


class DPRPolicy(nn.Module):
    def __init__(self, model_name="jhu-clsp/LegalBERT-DPR-CLERC-ft", device=None):
        super().__init__()
        from sentence_transformers import SentenceTransformer
        self.encoder = SentenceTransformer(model_name)
        self.model_name = model_name
        if device:
            self.encoder.to(device)
        # doc_id -> CPU tensor (n_chunks, dim)
        self._chunk_cache: dict[str, torch.Tensor] = {}
        self._chunk_cache_hits = 0
        self._chunk_cache_misses = 0

    def clear_chunk_cache(self) -> None:
        self._chunk_cache.clear()
        self._chunk_cache_hits = 0
        self._chunk_cache_misses = 0

    def chunk_text(
        self, text: str, chunk_size: int = RL_CHUNK_WORDS, overlap: int = RL_CHUNK_OVERLAP
    ) -> list[str]:
        words = text.split()
        if not words:
            return [""]
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i : i + chunk_size])
            chunks.append(chunk)
            if i + chunk_size >= len(words):
                break
        return chunks

    def _encode_chunk_texts_to_cpu(self, chunk_texts: list[str], batch_size: int) -> torch.Tensor:
        """Run encoder on chunks, return float tensor on CPU (no grad)."""
        device = next(self.encoder.parameters()).device
        was_training = self.encoder.training
        self.encoder.eval()
        out_list = []
        for start_idx in range(0, len(chunk_texts), batch_size):
            batch_texts = chunk_texts[start_idx : start_idx + batch_size]
            c_features = self.encoder.tokenize(batch_texts)
            c_features = {k: v.to(device) for k, v in c_features.items()}
            with torch.no_grad():
                c_emb = self.encoder(c_features)["sentence_embedding"]
            out_list.append(c_emb.cpu())
            del c_features
            del c_emb
            if device.type == "mps":
                torch.mps.empty_cache()
        if was_training:
            self.encoder.train()
        return torch.cat(out_list, dim=0)

    def _doc_chunks_tensor_cpu(
        self,
        doc_id: str,
        text: str,
        max_chunks_per_doc: int,
        batch_size: int,
        use_cache: bool,
    ) -> torch.Tensor:
        if use_cache and doc_id in self._chunk_cache:
            self._chunk_cache_hits += 1
            return self._chunk_cache[doc_id]
        self._chunk_cache_misses += 1
        chunks = self.chunk_text(text)
        if max_chunks_per_doc > 0:
            chunks = chunks[:max_chunks_per_doc]
        if not chunks:
            t = torch.zeros(0, self.encoder.get_sentence_embedding_dimension())
            if use_cache:
                self._chunk_cache[doc_id] = t
            return t
        t = self._encode_chunk_texts_to_cpu(chunks, batch_size)
        if use_cache:
            self._chunk_cache[doc_id] = t
        return t

    def load_chunk_cache(self, path: str | None, max_chunks: int) -> int:
        """Load cache from disk if file exists and metadata matches. Returns number of entries loaded."""
        if not path:
            return 0
        fp = Path(path)
        if not fp.is_file():
            return 0
        blob = torch.load(fp, map_location="cpu")
        if blob.get("model_name") != self.model_name or int(blob.get("max_chunks", -1)) != int(
            max_chunks
        ):
            return 0
        if int(blob.get("chunk_words", -1)) != RL_CHUNK_WORDS or int(
            blob.get("chunk_overlap", -1)
        ) != RL_CHUNK_OVERLAP:
            return 0
        data = blob.get("embeddings", {})
        self._chunk_cache = {str(k): v for k, v in data.items()}
        return len(self._chunk_cache)

    def save_chunk_cache(self, path: str | None, max_chunks: int) -> None:
        if not path:
            return
        fp = Path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_name": self.model_name,
                "max_chunks": max_chunks,
                "chunk_words": RL_CHUNK_WORDS,
                "chunk_overlap": RL_CHUNK_OVERLAP,
                "embeddings": dict(self._chunk_cache),
            },
            fp,
        )

    def forward(
        self,
        query_text: str,
        candidate_ids: list[str],
        candidate_texts: list[str],
        max_chunks_per_doc: int = 24,
        temperature: float = 1.0,
        batch_size: int = 16,
        use_chunk_cache: bool = True,
    ):
        device = next(self.encoder.parameters()).device

        q_features = self.encoder.tokenize([query_text])
        q_features = {k: v.to(device) for k, v in q_features.items()}
        q_emb = self.encoder(q_features)["sentence_embedding"]

        if len(candidate_texts) != len(candidate_ids):
            raise ValueError("candidate_ids and candidate_texts length mismatch")

        per_doc_embs: list[torch.Tensor] = []
        doc_indices: list[int] = []
        for i, doc_id in enumerate(candidate_ids):
            embs_cpu = self._doc_chunks_tensor_cpu(
                doc_id,
                candidate_texts[i],
                max_chunks_per_doc,
                batch_size,
                use_chunk_cache,
            )
            if embs_cpu.numel() == 0:
                continue
            per_doc_embs.append(embs_cpu)
            doc_indices.extend([i] * embs_cpu.shape[0])

        n_cand = len(candidate_texts)
        if not per_doc_embs:
            probs = torch.zeros(n_cand, device=device)
            doc_scores = torch.full((n_cand,), float("-inf"), device=device)
            return probs, doc_scores

        c_embs = torch.cat(per_doc_embs, dim=0).to(device)
        c_embs = c_embs.detach()

        sims = torch.matmul(c_embs, q_emb.T).squeeze(-1)

        doc_indices_tensor = torch.tensor(doc_indices, device=device, dtype=torch.long)
        doc_scores = []
        for i in range(n_cand):
            mask = doc_indices_tensor == i
            if mask.any():
                doc_scores.append(sims[mask].max())
            else:
                doc_scores.append(torch.tensor(float("-inf"), device=device))

        doc_scores = torch.stack(doc_scores)
        probs = F.softmax(doc_scores / temperature, dim=0)

        return probs, doc_scores
