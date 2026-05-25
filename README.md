# Hermes Expanded Memory — Busca Semântica no Obsidian

> Memória expandida para o Hermes Agent usando busca semântica local no vault do Obsidian.

## O que é

Um sistema de **busca semântica** (não apenas palavras-chave) que indexa automaticamente todas as notas do seu vault do Obsidian e permite que o Hermes Agent encontre informações relevantes por **significado**, mesmo usando palavras diferentes.

## Por que existe

O Hermes Agent tem memória natina limitada (~2.200 chars). Isso é insuficiente para:
- Lembrar decisões e ideias documentadas em dezenas de notas
- Conectar conceitos entre diferentes arquivos
- Recuperar informação sem saber exatamente onde está

A memória expandida resolve isso transformando **todo o vault do Obsidian** em memória pesquisável por IA.

## O que resolve

| Problema | Solução |
|---|---|
| Memória do Hermes limitada a ~2.200 chars | Vault inteiro (milhares de notas) como memória |
| Busca por palavras-chave falha com sinônimos | Busca semântica entende significado |
| Informação dispersa em muitos arquivos | Chunks indexados com similaridade |
| Dependência de serviços externos | 100% local e offline |
| Complexidade de configurar (ex: OpenViking) | Script único, dependências mínimas |

## Como resolve

**Arquitetura:**

```
Vault Obsidian (.md)
       │
       ▼
  Chunking (400 chars + overlap)
       │
       ▼
  Embeddings (sentence-transformers)
  Modelo: all-MiniLM-L6-v2 (90MB)
       │
       ▼
  Índice FAISS (busca vetorial)
       │
       ▼
  Resultados por similaridade
```

**Fluxo:**
1. Script lê todos os `.md` do vault
2. Divide em chunks de ~400 caracteres com overlap
3. Gera embeddings vetoriais com `all-MiniLM-L6-v2`
4. Indexa no FAISS (Facebook AI Similarity Search)
5. Na busca, gera embedding da query e encontra chunks mais similares
6. Retorna resultados com score de similaridade + caminho do arquivo

## Pontos positivos

- 🔒 **100% local e offline** — sem enviar dados pra nuvem
- ⚡ **Rápido** — busca em ~2s (após carregar o modelo)
- 🧠 **Semântico** — entende significado, não só palavras
- 💾 **Leve** — modelo de 90MB, roda em CPU ou GPU
- 🔄 **Incremental** — só reindexa arquivos que mudaram
- 📦 **Sem servidor** — script simples, sem API/serviço rodando
- 🖥️ **Compatível** — roda com GTX 1660 Ti 6GB / 16GB RAM

## Pontos negativos

- ⏱️ **Cold start** — ~2s pra carregar o modelo a cada busca
- 📏 **Chunking perde contexto** — notas muito longas podem ser cortadas no meio
- 🇬🇧**Inglês melhor que PT-BR** — modelo multilíngue mas mais preciso em inglês
- 🔧 **Reindexação manual** — precisa rodar o script após mudanças no vault
- 📦 **Dependências pesadas** — sentence-transformers + torch (~1.5GB)

## Requisitos

- Python 3.10+ (testado no 3.14)
- 4GB+ RAM (recomendado 8GB+)
- 2GB+ disco livre (modelo + dependências)
- GPU opcional (funciona em CPU, GPU acelera)

## Instalação

### 1. Clonar o repositório

```bash
git clone https://github.com/Tarnaph/hermes-expanded-memory.git
cd hermes-expanded-memory
```

### 2. Criar ambiente virtual (recomendado)

```bash
python -m venv venv
venv\Scripts\activate  # Windows
# ou
source venv/bin/activate  # Linux/Mac
```

### 3. Instalar dependências

```bash
pip install sentence-transformers faiss-cpu numpy scikit-learn
```

GPU (opcional, mais rápido):
```bash
pip install faiss-gpu  # substitui faiss-cpu
```

### 4. Configurar variável de ambiente

```bash
# Windows
set OBSIDIAN_VAULT_PATH=C:\Users\seu-usuario\Documents\Obsidian Vault

# Linux/Mac
export OBSIDIAN_VAULT_PATH=~/Documents/Obsidian%20Vault
```

Ou edite o script diretamente:
```python
VAULT_PATH = Path("C:/Users/seu-usuario/Documents/Obsidian Vault")
```

### 5. Indexar o vault

```bash
python obsidian_indexer.py index
```

### 6. Testar

```bash
python obsidian_indexer.py search "sua busca aqui"
python obsidian_indexer.py status
```

## Uso

### Indexar (após mudanças no vault)

```bash
python obsidian_indexer.py index          # Incremental (só muda o que mudou)
python obsidian_indexer.py index --force  # Reindexar tudo
```

### Buscar

```bash
python obsidian_indexer.py search "estratégia de vendas"
python obsidian_indexer.py search "como configuramos o provador IA"
python obsidian_indexer.py search "Mariana Boaventura livros"
```

### Status

```bash
python obsidian_indexer.py status
# 📁 C:\Users\rapha\Documents\Obsidian Vault
# 📄 Arquivos: 25 | Chunks SQLite: 225
# 📦 FAISS: 225 chunks, dim=384
```

## Integração com Hermes Agent

A skill `obsidian-memory` instrui o Hermes a buscar automaticamente nas notas quando relevante:

- Perguntas sobre projetos, ideias, decisões passadas
- "O que eu tenoi sobre X?"
- "Anotei algo sobre X?"

O Hermes busca, lê os arquivos relevantes e sintetiza a resposta citando a fonte.

## Estrutura do projeto

```
hermes-expanded-memory/
├── obsidian_indexer.py      # Script principal
├── README.md                # Este arquivo
└── requirements.txt         # Dependências
```

## Modelos alternativos

O script usa `all-MiniLM-L6-v2` por padrão. Alternativas:

| Modelo | Tamanho | Qualidade | Velocidade |
|---|---|---|---|
| `all-MiniLM-L6-v2` | 90MB | ✅ Boa | ⚡ Rápido |
| `all-mpnet-base-v2` | 420MB | ✅✅ Melhor | 🐢 Mais lento |
| `paraphrase-multilingual-MiniLM-L12-v2` | 470MB | ✅✅ PT-BR melhor | 🐢 Médio |

Troque no script:
```python
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
```

Para trocar:
```python
model = SentenceTransformer("all-mpnet-base-v2")
```

## Licença

MIT — use como quiser.
