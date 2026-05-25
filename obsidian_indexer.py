#!/usr/bin/env python3
"""
obsidian_indexer.py - Indexação semântica do vault Obsidian
Leve: usa modelos ONNX sem precisar de torch/transformers.

Uso:
  python obsidian_indexer.py index          # Indexa/reindexa tudo
  python obsidian_indexer.py search "query"  # Busca semântica
  python obsidian_indexer.py status          # Status do índice
"""

import os
import sys
import json
import hashlib
import sqlite3
from pathlib import Path
from datetime import datetime

_default_vault = Path.home() / "Documents" / "Obsidian Vault"
_env_vault = os.environ.get("OBSIDIAN_VAULT_PATH", "").strip()
VAULT_PATH = Path(_env_vault) if _env_vault else _default_vault

if not VAULT_PATH.exists():
    print(f"❌ Vault não encontrado: {VAULT_PATH}")
    print("   Sete OBSIDIAN_VAULT_PATH ou edite o script.")
    sys.exit(1)
DB_PATH = Path.home() / ".obsidian_search.db"
CHUNK_SIZE = 400
CHUNK_OVERLAP = 80

def check_deps():
    try:
        import faiss
        import numpy as np
        return True
    except ImportError:
        return False

def install_deps():
    import subprocess
    print("📦 Instalando dependências...")
    import shutil
    # Detectar Python do sistema
    python_exe = shutil.which("python3") or shutil.which("python")
    if not python_exe:
        python_exe = r"C:\Python314\python.exe"
    subprocess.check_call([python_exe, "-m", "pip", "install", "--user", "faiss-cpu", "numpy"])
    print("✅ Dependências instaladas. Rode novamente.")
    sys.exit(0)

if not check_deps():
    install_deps()

import faiss
import numpy as np

# --- Modelo de embedding simples usando tokenização + TF-IDF como fallback,
#     ou sentence-transformers se disponível ---
def get_embedder():
    """Tenta carregar sentence-transformers, fallback para TF-IDF."""
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        print("✅ Usando sentence-transformers (all-MiniLM-L6-v2)")
        return ("sbert", model)
    except ImportError:
        pass
    
    try:
        import sklearn.feature_extraction.text as text_mod
        print("⚠️ Usando TF-IDF como fallback (menos preciso, mas funciona)")
        return ("tfidf", text_mod.TfidfVectorizer(max_features=384))
    except ImportError:
        print("❌ Instale scikit-learn: pip install scikit-learn")
        sys.exit(1)

def get_db():
    db = sqlite3.connect(DB_PATH)
    db.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id TEXT PRIMARY KEY,
            file_path TEXT NOT NULL,
            chunk_index INTEGER,
            content TEXT NOT NULL,
            indexed_at TEXT
        )
    """)
    db.execute("""
        CREATE TABLE IF NOT EXISTS files (
            file_path TEXT PRIMARY KEY,
            last_modified REAL,
            last_indexed TEXT
        )
    """)
    db.commit()
    return db

def file_hash(content):
    return hashlib.md5(content.encode()).hexdigest()

def chunk_text(text):
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunks.append(text[start:end])
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks

def scan_vault():
    ignore = {".obsidian", ".trash", ".git"}
    files = []
    for p in VAULT_PATH.rglob("*.md"):
        if any(part in ignore for part in p.parts):
            continue
        files.append(p)
    return files

def index_vault(use_tfidf=False, force=False):
    db = get_db()
    files = scan_vault()
    
    print(f"📁 Vault: {VAULT_PATH}")
    print(f"📄 Arquivos: {len(files)}")
    
    index_path = DB_PATH.with_suffix(".faiss")
    
    # Detectar se sentence-transformers está disponível
    has_sbert = False
    if not use_tfidf:
        try:
            import sentence_transformers
            has_sbert = True
        except ImportError:
            print("⚠️ sentence-transformers não disponível, usando TF-IDF")
            use_tfidf = True
    
    if use_tfidf:
        # TF-IDF approach
        all_chunks_text = []
        all_meta = []
        
        for fpath in files:
            rel = str(fpath.relative_to(VAULT_PATH))
            try:
                content = fpath.read_text(encoding="utf-8")
                if content.startswith("---"):
                    end = content.find("---", 3)
                    if end > 0:
                        content = content[end+3:]
                chunks = chunk_text(content)
                for ci, chunk in enumerate(chunks):
                    if not chunk.strip():
                        continue
                    cid = file_hash(f"{rel}:{ci}:{chunk}")
                    existing = db.execute("SELECT id FROM chunks WHERE id = ?", (cid,)).fetchone()
                    if existing and not force:
                        continue
                    all_chunks_text.append(chunk)
                    all_meta.append((cid, rel, ci, chunk[:400]))
            except Exception as e:
                print(f"  ⚠️ {rel}: {e}")
        
        if not all_chunks_text and not force:
            print("✅ Nada novo pra indexar.")
            db.close()
            return
        
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer(max_features=384)
        tfidf_matrix = vectorizer.fit_transform(all_chunks_text).toarray().astype("float32")
        
        # Converter para FAISS
        if index_path.exists() and not force:
            index = faiss.read_index(str(index_path))
        else:
            dim = tfidf_matrix.shape[1]
            index = faiss.IndexFlatIP(dim)
        
        faiss.normalize_L2(tfidf_matrix)
        index.add(tfidf_matrix)
        
        for cid, rel, ci, preview in all_meta:
            db.execute("INSERT OR REPLACE INTO chunks (id, file_path, chunk_index, content, indexed_at) VALUES (?, ?, ?, ?, ?)",
                       (cid, rel, ci, preview, datetime.now().isoformat()))
        
        faiss.write_index(index, str(index_path))
        
        # Salvar vectorizer
        import pickle
        pickle.dump(vectorizer, open(str(DB_PATH.with_suffix(".vec")), "wb"))
        
        print(f"\n✅ {index.ntotal} chunks indexados (TF-IDF)")
    
    else:
        # Sentence Transformers approach
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        
        all_texts = []
        all_meta = []
        
        for i, fpath in enumerate(files):
            rel = str(fpath.relative_to(VAULT_PATH))
            mtime = fpath.stat().st_mtime
            
            row = db.execute("SELECT last_modified FROM files WHERE file_path = ?", (rel,)).fetchone()
            if row and row[0] == mtime and not force:
                continue
            
            try:
                content = fpath.read_text(encoding="utf-8")
                if content.startswith("---"):
                    end = content.find("---", 3)
                    if end > 0:
                        content = content[end+3:]
                chunks = chunk_text(content)
                for ci, chunk in enumerate(chunks):
                    if not chunk.strip():
                        continue
                    cid = file_hash(f"{rel}:{ci}:{chunk}")
                    existing = db.execute("SELECT id FROM chunks WHERE id = ?", (cid,)).fetchone()
                    if existing:
                        continue
                    all_texts.append(chunk)
                    all_meta.append((cid, rel, ci, chunk[:400]))
                
                db.execute("INSERT OR REPLACE INTO files (file_path, last_modified, last_indexed) VALUES (?, ?, ?)",
                           (rel, mtime, datetime.now().isoformat()))
            except Exception as e:
                print(f"  ⚠️ {rel}: {e}")
            
            if (i + 1) % 20 == 0:
                print(f"  ... {i+1}/{len(files)}")
        
        if all_texts:
            print(f"🔢 Gerando embeddings ({len(all_texts)} chunks)...")
            embeddings = model.encode(all_texts, show_progress_bar=True, batch_size=32)
            embeddings = embeddings.astype("float32")
            faiss.normalize_L2(embeddings)
            
            if index_path.exists() and not force:
                index = faiss.read_index(str(index_path))
            else:
                dim = embeddings.shape[1]
                index = faiss.IndexFlatIP(dim)
            
            index.add(embeddings)
            
            for cid, rel, ci, preview in all_meta:
                db.execute("INSERT OR REPLACE INTO chunks (id, file_path, chunk_index, content, indexed_at) VALUES (?, ?, ?, ?, ?)",
                           (cid, rel, ci, preview, datetime.now().isoformat()))
            
            faiss.write_index(index, str(index_path))
            print(f"\n✅ {index.ntotal} chunks indexados (SBERT)")
        else:
            if index_path.exists():
                idx_tmp = faiss.read_index(str(index_path))
                total = idx_tmp.ntotal
            else:
                total = 0
            print(f"\n✅ Nada novo. Índice: {total} chunks")
    
    db.commit()
    db.close()
    print(f"💾 Índice: {index_path}")
    print(f"💾 DB: {DB_PATH}")

def search(query, top_k=5):
    index_path = DB_PATH.with_suffix(".faiss")
    if not index_path.exists():
        print("❌ Índice não encontrado. Rode 'index' primeiro.")
        sys.exit(1)
    
    db = get_db()
    index = faiss.read_index(str(index_path))
    
    vec_path = DB_PATH.with_suffix(".vec")
    if vec_path.exists():
        # TF-IDF
        import pickle
        vectorizer = pickle.load(open(vec_path, "rb"))
        q = vectorizer.transform([query]).toarray().astype("float32")
    else:
        # SBERT
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        q = model.encode([query]).astype("float32")
    
    faiss.normalize_L2(q)
    scores, indices = index.search(q, top_k)
    
    all_chunks = db.execute("SELECT rowid, file_path, chunk_index, content FROM chunks ORDER BY rowid").fetchall()
    
    print(f"🔍 \"{query}\" — {index.ntotal} chunks indexados\n")
    for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
        if idx < 0 or idx >= len(all_chunks):
            continue
        _, fpath, chunk_idx, preview = all_chunks[idx]
        bar = "█" * int(float(score) * 20)
        print(f"  {rank+1}. [{float(score):.2%}] {bar}  {fpath} [chunk {chunk_idx}]")
        print(f"     {preview[:200]}...")
        print()
    db.close()

def status():
    db = get_db()
    index_path = DB_PATH.with_suffix(".faiss")
    
    fc = db.execute("SELECT COUNT(*) FROM files").fetchone()[0]
    cc = db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    last = db.execute("SELECT MAX(last_indexed) FROM files").fetchone()[0]
    
    print(f"📁 {VAULT_PATH}")
    print(f"📄 Arquivos: {fc} | Chunks SQLite: {cc}")
    if index_path.exists():
        idx = faiss.read_index(str(index_path))
        print(f"📦 FAISS: {idx.ntotal} chunks, dim={idx.d}")
    else:
        print("❌ Sem índice FAISS")
    print(f"🕐 Última indexação: {last or 'nunca'}")
    db.close()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    
    cmd = sys.argv[1]
    if cmd == "index":
        index_vault(force="--force" in sys.argv)
    elif cmd == "search":
        search(" ".join(sys.argv[2:]))
    elif cmd == "status":
        status()
    else:
        print(f"Comando: index, search, status")
