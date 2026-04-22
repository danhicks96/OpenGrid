# OpenGrid: A Distributed Peer-to-Peer LLM Inference Network
### Full Project Specification — Ready for GitHub Implementation

***

## Executive Summary

OpenGrid is a permissionless, volunteer-powered distributed inference network that allows anyone with a consumer PC, gaming GPU, or multi-core CPU to contribute compute time and receive inference credits in return. Users interact with a single, conversational API — identical to the OpenAI chat completions spec — while the network transparently routes their request across a mesh of contributor nodes, each holding one or more shards of a quantized open-source model. The system synthesizes proven architectural patterns from BOINC volunteer computing[^1][^2], Petals BitTorrent-style layer distribution[^3][^4], the exo framework's peer equality model[^5][^6], Prime Intellect's decentralized asynchronous inference approach[^7][^8], ARIA Protocol's 1-bit CPU-native P2P model[^9][^10][^11], and TOPLOC's trustless inference verification[^12][^13][^14][^15] — combining them into a single, cohesive, open-source project that any developer can clone and run.

***

## Part 1 — Vision and Design Goals

### 1.1 Why This Exists

Consumer hardware is the largest untapped AI compute reservoir on the planet. A single RTX 4090 gaming GPU sits idle for 16–20 hours per day while its owner is at work. At INT4 quantization, a 70B-parameter model requires only 35 GB of VRAM[^16][^17] — a figure easily reached by pooling two or three modern gaming GPUs. The problem is not compute; it is coordination software.

Frontier AI subscriptions gate access behind monthly fees, rate limits, and safety filters that frustrate creative and research use. Self-hosting requires technical expertise most people do not have. Cloud environments charge GPU-hour rates that price out individual experimenters. OpenGrid solves this by turning idle gaming PCs into a collectively owned inference cluster, governed by a simple credit system: contribute compute, spend compute[^18][^19].

### 1.2 Design Principles

1. **Permissionless participation** — no account approval required to join as a node or consumer.
2. **Heterogeneous hardware tolerance** — the network must run usefully on anything from a CPU-only Raspberry Pi to a multi-GPU workstation[^20][^5].
3. **User sovereignty** — every contributor controls exactly how much disk, VRAM, CPU, and bandwidth they donate, and in what time windows[^9][^21].
4. **Trustless verification** — outputs from untrusted nodes can be verified without re-running full inference[^12][^13][^14].
5. **OpenAI API compatibility** — consumers use the same API they already know; no application changes needed[^5][^11].
6. **No required blockchain or cryptocurrency** — the credit system uses a lightweight signed ledger, not a full proof-of-work chain.

***

## Part 2 — System Architecture Overview

OpenGrid consists of five cooperating layers:

```
┌─────────────────────────────────────────────────────────────────┐
│  Layer 5: Consumer API  (OpenAI-compatible REST/WebSocket)      │
├─────────────────────────────────────────────────────────────────┤
│  Layer 4: Coordinator / Scheduler  (Request routing + DAG mgmt) │
├─────────────────────────────────────────────────────────────────┤
│  Layer 3: P2P Mesh  (DHT peer discovery + gossip health)        │
├─────────────────────────────────────────────────────────────────┤
│  Layer 2: Worker Nodes  (Model shards + KV cache + inference)   │
├─────────────────────────────────────────────────────────────────┤
│  Layer 1: Node Client Daemon  (Benchmark, resource mgmt, ledger)│
└─────────────────────────────────────────────────────────────────┘
```

Each layer is described in full in the sections below. The system is designed so that a single machine can simultaneously run layers 1, 2, 3, and 4 (for small private networks), or they can be spread across many machines for large public networks.

***

## Part 3 — Node Client Daemon (Layer 1)

The daemon is the software every contributor installs. It is the equivalent of the BOINC client[^1][^2] but extended for real-time tensor computation rather than batch scientific work.

### 3.1 Startup Benchmark

On first launch, and on each subsequent major hardware change, the daemon runs a benchmark suite to profile the local machine. This profile is advertised to the P2P mesh so the scheduler can make informed routing decisions[^22][^23][^24].

**Benchmark targets:**

| Benchmark | What it measures | How |
|---|---|---|
| CPU GEMM throughput | INT8 and 1-bit GEMM tokens/sec | Run BitNet b1.58 on a fixed 256-token prompt[^25][^26] |
| GPU GEMM throughput | FP16, INT8, INT4 tokens/sec | vLLM micro-benchmark at fixed batch size |
| GPU VRAM available | Usable VRAM after OS/display overhead | `nvidia-smi` + headroom check |
| RAM available | Free RAM for KV cache and CPU offload | OS API |
| Disk read bandwidth | Shard loading speed | Sequential read of 1 GB temp file |
| Upload bandwidth | Activation egress | 4 MB TCP transfer to bootstrap node |
| Download bandwidth | Shard and prompt ingress | 4 MB TCP transfer from bootstrap node |
| Estimated sustained load | Compute at 50%/80%/100% resource limit | 30-second inference burn-in |

Results are stored in `~/.opengrid/profile.json` and re-run automatically if hardware changes are detected.

### 3.2 Resource Configuration

The daemon exposes a config file (`~/.opengrid/config.toml`) and a GUI tray app that lets contributors set hard limits. Defaults are conservative.

```toml
[resources]
max_gpu_fraction     = 0.50       # 50% of GPU time
max_vram_gb          = 6.0        # Reserve 6 GB of VRAM for model shards
max_ram_gb           = 4.0        # Reserve 4 GB RAM for KV cache
max_disk_gb          = 40.0       # Model shard cache budget
max_upload_mbps      = 20.0       # Egress cap
max_cpu_fraction     = 0.25       # CPU fallback limit

[schedule]
active_hours_start   = "08:00"    # Only serve jobs after 8 AM local
active_hours_end     = "17:00"    # Stop at 5 PM (work hours = idle)
pause_on_battery     = true       # Laptops: pause when unplugged
pause_when_gaming    = true       # Suspend on GPU load > 80% from another process

[jobs]
allowed_types        = ["inference", "validation"]  # No training by default
max_sequence_length  = 4096
```

This mirrors the consent contract model used in ARIA Protocol[^9][^11], ensuring no resource is ever used without explicit permission.

### 3.3 Shard Cache Management

Model weights are distributed as immutable, content-addressed shards — much like BitTorrent pieces. Each shard corresponds to a contiguous range of transformer layers in a quantized model. The daemon maintains a local shard registry and downloads only the shards it has been assigned based on its hardware tier[^27][^4].

```
~/.opengrid/shards/
  llama3-70b-int4/
    shard-000-layers-00-07.safetensors   (≈ 4.4 GB)
    shard-001-layers-08-15.safetensors   (≈ 4.4 GB)
    ...
  bitnet-b158-2b/
    shard-000-layers-00-15.safetensors   (≈ 0.2 GB)
    ...
```

Shards are verified at download with SHA-256 hashes published in the global model registry. Shards are evicted on a least-recently-used basis when disk budget is exceeded[^20][^5].

### 3.4 Credit Ledger

The daemon maintains a local ledger of compute credits earned and spent. Each completed job appends a signed receipt:

```json
{
  "job_id": "abc123",
  "timestamp": 1745123456,
  "node_id": "node-xyz",
  "model": "llama3-70b-int4",
  "shard_range": [0, 7],
  "tokens_processed": 512,
  "credits_earned": 51.2,
  "coordinator_sig": "0xdeadbeef...",
  "toploc_proof": "base64..."
}
```

Credits are denominated in **inference tokens** (IT) — 1 IT = 1 token processed at full precision. Quantized nodes earn fractional rates reflecting quality-adjusted compute (INT4 earns 0.9 IT/token; 1-bit earns 0.5 IT/token). Receipts are signed by the coordinator and locally verifiable. No external blockchain is required.

***

## Part 4 — Worker Nodes (Layer 2)

Worker nodes are machines actively serving inference requests. A single machine running the daemon automatically becomes a worker node if it has shards loaded and resource limits permit.

### 4.1 Inference Engine

The worker inference engine is a thin wrapper around well-tested open-source backends:

| Hardware | Backend | Notes |
|---|---|---|
| NVIDIA GPU (INT4) | vLLM + AWQ/GPTQ kernels | Best throughput for consumer NVIDIA GPUs[^28][^29] |
| NVIDIA GPU (INT8) | vLLM + bitsandbytes | Near-lossless quality, ~50% memory savings vs FP16[^16][^30] |
| Apple Silicon | MLX via exo backend[^5][^6] | Unified memory architecture benefits |
| CPU-only (INT4) | llama.cpp / GGUF | 2-3 tokens/sec for small models |
| CPU-only (1-bit) | bitnet.cpp[^25][^26] | 15–89 tokens/sec for BitNet models; viable for pipeline[^10] |
| Heterogeneous cluster | exo ring partitioning[^20][^5] | Auto-assigns layers by VRAM ratio |

### 4.2 The Work Packet Protocol

When the scheduler assigns a job to a worker, it sends a **work packet** over an encrypted WebSocket connection:

```json
{
  "packet_type": "inference_forward",
  "job_id": "abc123",
  "sequence_id": 42,
  "model_id": "llama3-70b-int4",
  "shard_range": [8, 15],
  "input_activations_b64": "<base64-encoded tensor>",
  "kv_cache_token": "kv-abc123-layer8-15",
  "return_address": "node-next-id",
  "deadline_ms": 2000
}
```

The worker:
1. Loads the specified shard (from RAM/VRAM cache or disk).
2. Runs the forward pass for layers 8–15 on the input activations.
3. Returns output activations to the `return_address` node.
4. Emits a TOPLOC proof alongside the output (see Section 7).

### 4.3 KV Cache Management

For multi-turn conversations and long-context generation, attention key-value pairs must persist between decode steps. Each worker maintains a KV cache store keyed by `(job_id, shard_range)`[^31][^32].

KV cache is the single largest memory consumer in long-context inference. The daemon's `max_ram_gb` setting caps total KV cache size. When the cache fills, it evicts sessions using an LRU policy, forcing those sessions to recompute prefill from scratch on the next request — a graceful degradation rather than a hard failure[^31][^33][^34].

***

## Part 5 — P2P Mesh (Layer 3)

The mesh layer provides peer discovery, health monitoring, and gossip-based state synchronization — the equivalent of the DHT in BitTorrent[^35][^36].

### 5.1 Peer Discovery: Kademlia DHT

OpenGrid uses a **Kademlia-based Distributed Hash Table** for peer discovery, the same approach used by Hivemind[^37][^38], ARIA Protocol[^10][^11], and the vAIn P2P framework[^39].

Each node receives a 160-bit NodeID at first run (hashed from its public key). The DHT maps `NodeID → (IP, port, capability_profile)`. Bootstrap is handled by:

1. A hardcoded list of 3–5 stable bootstrap nodes shipped with the client.
2. A community-maintained DNS seed list (similar to Bitcoin's DNS seeds).
3. Peer exchange (PEX) — once connected to one node, it shares its peer list.

After first bootstrap, the node saves its peer table locally so it can reconnect without the seed list[^40].

### 5.2 Gossip Protocol for Health and Load

Rather than polling a central server, nodes gossip health and load metrics to $$O(\log N)$$ peers every 5 seconds[^41][^42][^43]. Each gossip message contains:

```json
{
  "node_id": "node-xyz",
  "timestamp": 1745123456,
  "seq": 1042,
  "status": "active",
  "vram_free_gb": 5.2,
  "jobs_active": 2,
  "avg_latency_ms": 145,
  "shards_hosted": ["llama3-70b-int4:0-7", "llama3-70b-int4:8-15"],
  "tier": "mid"
}
```

Gossip uses **anti-entropy** — nodes exchange summaries and reconcile stale state, making the system self-healing after partitions[^41][^42]. The fanout is set to `sqrt(N)` by default to balance propagation speed against traffic.

### 5.3 Node Tier Classification

The scheduler classifies nodes into tiers at admission, based on their benchmark profile. Tiers determine which jobs and shard sizes are routed to each node[^44][^45][^21].

| Tier | Typical hardware | VRAM | Eligible jobs |
|---|---|---|---|
| **Light** | CPU-only, integrated GPU | 0 GB | 1-bit model shards, embeddings, validation tasks |
| **Mid** | Consumer GPU 8–12 GB VRAM | 8–12 GB | INT4/INT8 shards, short-context decode, KV host |
| **Heavy** | Consumer GPU 16–24 GB VRAM | 16–24 GB | Primary decode path, long-context KV host, full prefill |
| **Power** | Multi-GPU or workstation GPU | 24+ GB | Full-model hosting, tensor parallel within node |

Tier assignments are updated dynamically as gossip health data reflects actual observed latency and error rates.

***

## Part 6 — Coordinator / Scheduler (Layer 4)

The coordinator is the brain of the system. In a public deployment, a small fleet of coordinator nodes handles routing (coordinators can themselves be contributed by community members as a special node type). In a private deployment, a single machine runs the coordinator.

### 6.1 Request Lifecycle

```
User sends POST /v1/chat/completions
         │
         ▼
[Coordinator] parses request, enriches with user memory context
         │
         ▼
[Scheduler] builds pipeline DAG across available nodes
         │
         ├─ Prefill phase ──► Node A (layers 0-7) ──► Node B (8-15) ──► ... ──► Node N
         │                                                                          │
         │                                                        returns first token logits
         │
         ├─ Decode phase (loop for each new token):
         │   ├─ Route to KV-cache-warm nodes if available
         │   └─ Sample token, check stopping criteria
         │
         ▼
[Coordinator] streams tokens back via SSE
         │
         ▼
User receives streamed response
```

### 6.2 DAG Execution Engine

Each request becomes a **directed acyclic graph** of sub-tasks. The coordinator maintains an in-memory DAG state machine:

```python
class InferenceDAG:
    job_id: str
    tasks: dict[str, Task]         # task_id → Task
    dependencies: dict[str, list]  # task_id → [upstream_task_ids]
    status: dict[str, TaskStatus]  # PENDING | RUNNING | DONE | FAILED
    activations: dict[str, Tensor] # intermediate results

class Task:
    task_id: str
    node_id: str                   # assigned worker
    shard_range: tuple[int, int]
    deadline_ms: int
    retry_count: int = 0
    max_retries: int = 2
```

When a task completes, its output activations are forwarded to all downstream dependent tasks. If a task times out or errors, the coordinator immediately re-routes to a backup node (if available) or marks the task failed and returns an error to the user[^46][^47][^48].

### 6.3 KV Cache Aware Routing

The most impactful scheduling optimization is routing decode steps to nodes that already hold the KV cache for the current session. Cache miss forces expensive recomputation; cache hits can yield up to 87% cache reuse and 88% faster time-to-first-token in production systems[^31][^32][^34].

OpenGrid implements prefix-hash-based routing: when a new request arrives with a known session ID, the coordinator checks its session-to-node mapping and preferentially routes to the node currently holding that KV cache[^32][^33][^34].

### 6.4 Admission Control

Before assigning a task to a node, the coordinator checks:

1. **Shard availability** — does the node hold the correct shard?
2. **Current load** — is `jobs_active` below the node's self-reported capacity?
3. **Latency budget** — can this node reach the coordinator with P99 latency under the SLA?
4. **Reputation score** — has the node been flagged for prior failures or bad proofs?

If no suitable node is available, the request is queued with a backoff, or returned with a 503 status and a `Retry-After` header.

***

## Part 7 — Trustless Verification (TOPLOC)

Because nodes are operated by unknown volunteers, the system must be able to detect and penalize dishonest computation without re-running every inference in full[^12][^13][^14][^15].

### 7.1 How TOPLOC Works

TOPLOC (published at ICML 2025[^14][^15]) uses locality-sensitive hashing of intermediate model activations to generate compact proofs:

- After processing its assigned layers, each worker generates a **TOPLOC proof** — a polynomial-encoded hash of its top-k intermediate activations.
- The proof requires only **258 bytes per 32 tokens**, compared to 262 KB if full activations were stored — a 1000x size reduction[^13][^14].
- Validation is up to **100x faster** than the original inference[^12][^13].
- The method detects unauthorized modifications to models, prompts, or compute precision with **100% accuracy** in empirical evaluations[^13][^14][^15].

### 7.2 Integration into OpenGrid

Every worker includes a TOPLOC proof with its output. The coordinator samples a fraction of proofs for verification — full verification of every proof is optional but available. A separate pool of **validator nodes** (light-tier machines) can verify proofs at high speed without running full inference[^7][^8].

Known TOPLOC limitation: it is robust against prefill tampering but can be evaded during token decoding via speculative decoding attacks (where a node honestly prefills but decodes with a cheaper model)[^49]. OpenGrid addresses this by also applying LOGIC-style log-probability verification for the first 16 decode tokens[^49] and by maintaining a reputation system that flags statistical anomalies in output distributions.

### 7.3 Reputation and Penalty System

Each node maintains a **reputation score** (0–1000) initialized at 500 for new nodes:

| Event | Score change |
|---|---|
| Proof passes validation | +1 |
| Proof fails (softly — hardware jitter) | -5 |
| Proof fails (hard — deliberate tampering detected) | -200 |
| Job timeout (node not at fault) | -2 |
| Job timeout (node fault — no response) | -20 |
| Sustained uptime bonus (24h active) | +10 |

Nodes below score 200 are demoted to validation-only tasks. Nodes below score 100 are temporarily banned for 24 hours. Persistent bad actors are blacklisted by NodeID and can be expelled from the DHT via a community-voted revocation list[^22][^23][^46].

***

## Part 8 — Model Registry and Shard Distribution

### 8.1 Supported Model Catalog

OpenGrid ships with a curated model registry. Priority models are chosen for open licensing, broad community adoption, and quantization quality.

| Model | Full Size (FP16) | INT4 Size | Shards | Min tier |
|---|---|---|---|---|
| BitNet-b1.58-2B[^25][^26] | 4.8 GB | 0.4 GB | 2 | Light (CPU) |
| Llama-3.2-3B-Instruct | 6 GB | 1.5 GB | 4 | Light/Mid |
| Llama-3.1-8B-Instruct | 16 GB | 4 GB | 8 | Mid |
| Llama-3.1-70B-Instruct | 140 GB | 35 GB | 16 | Mid (pooled) |
| Mixtral-8x7B | 88 GB | 22 GB | 8 (MoE) | Mid |
| Falcon-180B | 360 GB | 90 GB | 32 | Heavy (pooled) |

Community-submitted model manifests can add any Hugging Face model as long as a quantized version and shard split are provided and signed.

### 8.2 Shard Manifest Format

```json
{
  "model_id": "llama3-70b-int4",
  "base_model": "meta-llama/Llama-3.1-70B-Instruct",
  "quantization": "awq-int4",
  "total_layers": 80,
  "shards": [
    {
      "shard_id": 0,
      "layers": [0, 9],
      "size_gb": 4.38,
      "sha256": "abc123...",
      "hf_url": "https://huggingface.co/..."
    }
  ]
}
```

***

## Part 9 — Consumer API (Layer 5)

The user-facing API is intentionally identical to the OpenAI chat completions API so any existing application can use OpenGrid with a single `base_url` change[^5][^50][^11].

### 9.1 Endpoints

```
POST   /v1/chat/completions         # Standard chat completions (streaming supported)
GET    /v1/models                   # List available models
POST   /v1/embeddings               # Text embeddings
GET    /v1/network/status           # Network health dashboard
GET    /v1/credits/balance          # Current credit balance
POST   /v1/credits/spend            # Explicit credit reservation for long jobs
```

### 9.2 Credit System

Every request deducts credits from the user's balance:

```
credits_cost = tokens_generated × model_cost_factor × priority_multiplier

where:
  model_cost_factor:   1.0 for 8B, 2.5 for 70B, 0.25 for 2B BitNet
  priority_multiplier: 1.0 (standard) or 2.0 (priority queue)
```

Users earn credits by running a node. A gaming PC contributing a mid-tier node earns roughly 10,000–50,000 IT/hour depending on GPU tier and utilization. A typical 512-token response from a 70B model costs roughly 1,280 IT. Heavy contributors will have more free inference than they can consume[^18][^51].

### 9.3 Local Memory Store

Each user gets a local vector database (default: ChromaDB or LanceDB) that stores their long-term conversation context, project notes, and retrieved documents. This data never leaves the user's machine. The API client enriches prompts with relevant retrieved context before sending to the network, keeping personal data private[^52][^53].

***

## Part 10 — End-to-End Request Flow (Detailed)

This section walks through a complete request from user keystroke to streamed response, tying all components together.

**Scenario:** User sends "Continue the alternate history from 1453 where Constantinople did not fall" with a 2,000-token context.

```
t=0ms   User POST /v1/chat/completions with model=llama3-70b-int4

t=1ms   API server receives request
        → Deducts credit hold from local balance
        → Enriches prompt with local memory (retrieved documents about prior session)
        → Sends to Coordinator

t=2ms   Coordinator receives enriched prompt (2,300 tokens after memory injection)
        → Builds inference DAG:
            - Prefill tasks: 8 pipeline stages, one per shard (layers 0-9, 10-19, ... 70-79)
            - Decode tasks: assigned dynamically per token
        → Runs admission control for each stage

t=5ms   Scheduler queries DHT for available nodes holding each shard
        → Finds 12 eligible nodes across 8 shards
        → Selects path: Node-A(0-9) → Node-B(10-19) → Node-C(20-29) → ... → Node-H(70-79)
        → Reserves KV cache slots on decode-phase nodes

t=8ms   Prefill packets dispatched to all pipeline stages simultaneously (micro-batched)
        → Input embeddings computed locally on coordinator
        → Activations forwarded stage by stage

t=180ms Prefill completes across all stages
        → First token logits returned from Node-H
        → Token sampled: "The"

t=185ms First token streamed to user via SSE
        → Decode loop begins

t=185ms-2800ms  Decode loop (≈15 tokens/sec on pooled mid-tier nodes)
        → Each decode step: coordinator routes to KV-cache-warm nodes
        → TOPLOC proofs sampled every 32 tokens
        → Tokens streamed to user in real time

t=2800ms  ~400 tokens generated, stop sequence met
        → Final response assembled and closed
        → Credits deducted: 2300 input + 400 output = 2700 IT × 2.5 = 6,750 IT
        → Job receipt signed and stored on both coordinator and user node
```

***

## Part 11 — Edge Cases and Failure Modes

This is where many distributed systems projects fail. OpenGrid explicitly handles the following scenarios.

### 11.1 Node Dropout Mid-Request

**Problem:** A worker node drops out partway through a long generation (power cut, gaming starts, user closes laptop).

**Solution:** Petals-style partial restart protocol[^48]. Both client and intermediate workers cache activations at each layer boundary. When dropout is detected (deadline_ms exceeded + no heartbeat), the coordinator:
1. Marks the failed task.
2. Re-routes the prior stage's cached activations to a backup node holding the same shard.
3. Resumes generation from the last valid checkpoint, not from the beginning[^48].

Cost: additional latency of 100–500ms for rerouting. User experience: a brief pause in token streaming, then resumption[^47][^54].

### 11.2 Cold Start — No Nodes Holding Required Shard

**Problem:** A rare or newly-added model has no online nodes holding its shards. User gets a 503.

**Solution:**
- **Warm pool:** The network designates a minimum of 3 "anchor nodes" — always-on servers contributed by the project team or major contributors — that hold full copies of tier-1 models. These handle cold-start traffic and seed shard distribution to new nodes[^55][^56].
- **Background seeding:** When a user requests a model not currently in their shard cache, the daemon begins downloading that model's shards in the background at low priority, earning partial credits upon completion[^27][^4].
- **Fallback:** Users can optionally configure a fallback API endpoint (e.g., a local Ollama instance or a centralized provider) if the distributed network cannot serve the request within a timeout.

### 11.3 Sybil Attack — Fake Nodes Flooding the Network

**Problem:** A malicious actor registers thousands of fake node IDs to game the credit system or poison outputs.

**Solution:**
- **Proof of Useful Work** — credits are only issued for jobs with valid TOPLOC proofs confirmed by independent validators. Fake nodes producing garbage outputs fail validation immediately[^12][^13][^9].
- **Stake requirement for high-tier nodes** — nodes requesting Heavy or Power tier assignments must stake a configurable number of credits (earned first as a Light node). This creates a cost barrier for mass fake node registration[^18][^57].
- **IP diversity requirement** — coordinator limits trust of nodes sharing the same /24 subnet to avoid datacenter-scale Sybil farms.
- **Reputation bootstrapping** — new nodes start at neutral reputation (500) and cannot access high-value jobs until they have proven reliability over at least 1,000 validated tasks[^46][^58].

### 11.4 Eclipse Attack — Isolation of a Node's DHT View

**Problem:** A coordinated attacker fills a victim node's DHT routing table with malicious peers, isolating it from the honest network.

**Solution:** Kademlia's bucket-based peer selection naturally limits the fraction of any node's routing table that a single attacker can occupy. OpenGrid additionally enforces a minimum of 5 nodes per k-bucket from distinct ASNs, making eclipse attacks require coordination across multiple autonomous systems[^59][^39]. The bootstrap node list is hardcoded and signed, preventing an attacker from replacing it.

### 11.5 Prompt Privacy — Inference Nodes Can See Prompts

**Problem:** When a prompt is sent to a pipeline node for processing, that node's operator can in principle read it.

**Solution (tiered):**
- **Informed consent:** Users are clearly notified that standard mode sends prompts to volunteer nodes. This is equivalent to using any API service.
- **Private mode:** For sensitive prompts, the user can restrict routing to nodes they personally trust (e.g., their own second machine, or a trusted friend's node) using a signed allowlist. This sacrifices performance for privacy.
- **Confidential Compute (roadmap):** Integration with TEE (Trusted Execution Environment) nodes — machines running inference inside Intel SGX or AMD SEV enclaves — so that node operators cannot inspect prompt contents[^52][^53]. This is a future roadmap item, not an MVP requirement.

### 11.6 Model Poisoning — Tampered Shard Weights

**Problem:** A node serves modified model weights that subtly alter outputs.

**Solution:** All shards are content-addressed by SHA-256 hash published in the signed model manifest. The daemon verifies shard hashes on download and on each load. TOPLOC proofs also catch deviations from expected intermediate activations, making weight modification detectable even without re-hashing[^12][^13][^14].

### 11.7 Network Partition — DHT Split

**Problem:** A major network partition splits the node graph into two disconnected halves. Users on either side see degraded service.

**Solution:** Gossip protocol with **anti-entropy reconciliation** — when a node re-connects after a partition, it exchanges state digests with peers and fills in gaps[^41][^42]. Coordinator nodes maintain redundant DHT connections to geographically diverse bootstrap nodes, making full partition rare. Partial partition (some shards unavailable) triggers graceful degradation to smaller models.

### 11.8 Bandwidth Abuse — High-Volume Free Riding

**Problem:** A user configures their node to produce minimal useful work but requests maximal inference.

**Solution:** Credit balance must be positive before inference is served. Credits are earned only for validated completed jobs. The system tracks the ratio of credits earned to credits spent per NodeID; nodes with earn-to-spend ratio below 0.01 over a rolling 24-hour window are flagged and rate-limited until the ratio recovers[^51].

### 11.9 Thermal Throttling Mid-Job

**Problem:** A node's GPU throttles under load, causing decode latency to spike mid-generation.

**Solution:** The daemon's benchmark includes a 30-second sustained burn-in to detect thermal throttling behavior. Nodes that throttle heavily are classified one tier lower than their peak benchmark would suggest. Mid-job latency spikes trigger a soft timeout; if the node's P95 response time exceeds 3x its advertised SLA, the coordinator re-routes remaining decode steps to a backup node without aborting the session[^44][^60][^54].

### 11.10 Long-Context KV Cache Eviction

**Problem:** A user is running a very long simulation (50k+ tokens). The KV cache for their session is evicted from the assigned worker node due to memory pressure.

**Solution:** 
- **Cache pinning:** Users with sufficient credit balance can pin their session's KV cache by paying a small reservation fee (credits per GB-hour).
- **Graceful recompute:** On cache miss, the coordinator re-runs prefill for the full context before resuming generation. This is expensive but transparent to the user (pause in streaming, then resumption).
- **Tiered storage:** Warm KV cache in VRAM → cold KV cache on RAM → archived KV cache on disk (model-defined max, e.g., 30k tokens)[^31][^33][^34].

***

## Part 12 — Technology Stack

### 12.1 Core Dependencies

| Component | Technology | Rationale |
|---|---|---|
| Inference engine (GPU) | vLLM | Production-grade, OpenAI-compatible, KV cache support[^34] |
| Inference engine (CPU/1-bit) | bitnet.cpp[^25][^26] | 2.37–6.17x speedup on x86; 1-bit native support |
| P2P networking | libp2p (Python) | Used by Hivemind[^37][^38]; battle-tested DHT + gossip |
| Model quantization | AutoAWQ / bitsandbytes[^28][^29] | AWQ INT4 for best quality/speed; bitsandbytes for INT8 |
| Shard verification | TOPLOC[^12][^13][^14] | Trustless proof at 258 bytes/32 tokens |
| KV cache routing | Custom (inspired by llm-d[^31][^32]) | Prefix-hash based, session-affinity routing |
| API server | FastAPI + uvicorn | OpenAI-compatible, async, streaming SSE |
| Local memory | LanceDB | Embedded vector DB, no external service |
| Config/GUI | Tauri 2.0 (desktop app) | Cross-platform, lightweight, Rust+WebView[^9] |
| Credit ledger | SQLite + ECDSA signatures | Local, portable, no blockchain required |
| Node discovery | Kademlia DHT (hivemind / py-libp2p) | Scales to 10k+ peers with O(log N) lookups[^37][^61] |

### 12.2 Repository Structure

```
opengrid/
├── README.md
├── pyproject.toml
├── opengrid/
│   ├── __init__.py
│   ├── daemon/
│   │   ├── benchmark.py          # Hardware profiling
│   │   ├── config.py             # TOML config management
│   │   ├── shard_manager.py      # Download, verify, evict shards
│   │   ├── credit_ledger.py      # SQLite ledger + signing
│   │   └── resource_guard.py     # Enforce resource limits
│   ├── node/
│   │   ├── worker.py             # Inference job execution
│   │   ├── inference_engine.py   # vLLM / bitnet.cpp abstraction
│   │   ├── kv_cache.py           # KV cache store + eviction
│   │   └── toploc_prover.py      # TOPLOC proof generation
│   ├── mesh/
│   │   ├── dht.py                # Kademlia DHT wrapper
│   │   ├── gossip.py             # Health/load gossip
│   │   ├── peer_registry.py      # In-memory peer table
│   │   └── bootstrap.py          # Bootstrap node list + DNS seed
│   ├── coordinator/
│   │   ├── scheduler.py          # Request → DAG → node assignment
│   │   ├── dag_executor.py       # DAG state machine
│   │   ├── kv_router.py          # KV-cache-aware routing
│   │   ├── admission.py          # Node eligibility checks
│   │   └── reputation.py        # Reputation scoring
│   ├── api/
│   │   ├── server.py             # FastAPI app
│   │   ├── routes/
│   │   │   ├── completions.py    # POST /v1/chat/completions
│   │   │   ├── models.py         # GET /v1/models
│   │   │   └── credits.py        # Credit balance endpoints
│   │   └── middleware/
│   │       ├── auth.py           # API key validation
│   │       └── credit_check.py  # Pre-flight credit check
│   ├── registry/
│   │   ├── model_registry.py     # Model manifest store
│   │   └── manifests/
│   │       ├── llama3-70b-int4.json
│   │       ├── bitnet-b158-2b.json
│   │       └── ...
│   └── memory/
│       ├── local_store.py        # LanceDB vector store wrapper
│       └── retriever.py          # RAG retrieval for prompt enrichment
├── desktop/
│   ├── src-tauri/               # Tauri Rust backend
│   └── src/                     # React frontend for node management GUI
├── tests/
│   ├── unit/
│   ├── integration/
│   └── e2e/
├── scripts/
│   ├── setup_bootstrap_node.sh
│   ├── benchmark_node.py
│   └── generate_shard_manifest.py
└── docs/
    ├── architecture.md
    ├── contributing.md
    ├── credit-system.md
    └── security-model.md
```

***

## Part 13 — Deployment and Bootstrap Strategy

### 13.1 Phase 1 — Private Network (MVP)

The first milestone is a working private network of 2–10 machines, controlled by the project team or early contributors. Goals:
- Validate the DAG executor handles node dropout without full restarts.
- Verify TOPLOC proofs round-trip correctly between worker and validator.
- Measure real-world tokens/sec on a pooled mid-tier network vs. single-node.
- Test the credit ledger under concurrent requests.

**Minimum viable hardware for Phase 1:** Two machines with 12+ GB VRAM each (e.g., two RTX 3080 or RTX 4070 nodes) can run Llama-3.1-8B-Instruct with full pipeline parallelism across 8 shards.

### 13.2 Phase 2 — Public Alpha

Open the DHT to external contributors. Introduce the desktop app (Tauri) for one-click node setup. Goals:
- Achieve 50+ contributor nodes.
- Test DHT stability under node churn.
- Run Llama-3.1-70B-Instruct on pooled community hardware.
- Enable credit earning and spending for non-technical users.

### 13.3 Phase 3 — Public Beta

- Full model catalog (70B, Mixtral, Falcon-180B).
- KV cache-aware routing fully deployed.
- TOPLOC validation running on 10% of sampled jobs.
- Desktop app available on Windows, macOS (Apple Silicon), and Linux.
- REST API documented and stabilized.

### 13.4 Phase 4 — Ecosystem

- Public model submission (any Hugging Face model + quantization config).
- Developer SDK (Python, Node.js) for building applications on OpenGrid.
- Opt-in TEE (trusted execution environment) node tier for privacy-sensitive workloads.
- Community governance for model catalog and protocol changes.

***

## Part 14 — Performance Expectations

Based on results from Petals[^3][^4], exo[^20][^5], and ARIA Protocol[^9][^10], realistic performance targets for the public network:

| Scenario | Expected throughput | Notes |
|---|---|---|
| Llama-3.2-3B, 4 mid-tier nodes | 25–40 tokens/sec | Near single-GPU performance |
| Llama-3.1-8B, 4 mid-tier nodes | 12–20 tokens/sec | Sufficient for interactive chat |
| Llama-3.1-70B, 8 pooled nodes | 4–8 tokens/sec | Petals achieves ~6 tokens/sec[^62][^4] |
| BitNet-2B, 10 CPU nodes | 15–37 tokens/sec | ARIA validates 37 t/s single CPU[^10][^11] |
| Large batch (batch=16), 70B | High throughput | Pipeline parallelism excels at batch[^63] |

**Important:** Time-to-first-token for interactive use will typically be 300–800ms for 70B models on a geo-distributed network, which is acceptable for non-latency-critical tasks. Interactive chat is best served by same-region nodes. Long-form generation (simulation, writing, batch jobs) is where the network excels — latency matters less, throughput matters more.

***

## Part 15 — What Differentiates OpenGrid from Existing Projects

| Project | Key similarity | What OpenGrid adds |
|---|---|---|
| **Petals**[^3][^4] | Pipeline parallelism, BitTorrent-style distribution | Credit system, user GUI, TOPLOC verification, 1-bit model support, KV-cache-aware routing |
| **exo**[^20][^5] | P2P device equality, auto-partitioning | Public permissionless network, reputation system, credit economy |
| **ARIA Protocol**[^9][^10] | 1-bit CPU-native P2P inference | GPU support, KV cache management, multi-model registry |
| **Hivemind**[^37][^38] | DHT-based P2P, decentralized training | Inference focus, consumer UX, credit system |
| **Prime Intellect**[^7][^8] | Decentralized async training/inference | Consumer-first, no token/blockchain requirement, simpler onboarding |
| **BOINC**[^1][^2] | Volunteer computing, credit system, benchmark | Real-time inference instead of batch jobs, GPU-first |

***

## Part 16 — Risks and Mitigations Summary

| Risk | Severity | Mitigation |
|---|---|---|
| Insufficient node count at launch | High | Anchor nodes + incentivized early adopter credits |
| Latency too high for interactive use | Medium | KV-cache routing + same-region node preferences |
| Malicious nodes poisoning outputs | High | TOPLOC proofs + reputation scoring + redundant validation |
| Privacy — prompts visible to nodes | Medium | Informed consent + private mode + TEE roadmap |
| Credit farming without useful work | Medium | Proof of Useful Work + earn/spend ratio monitoring |
| Regulatory risk (compute marketplace) | Low | Non-monetary credit system avoids MSB classification |
| Model licensing violations | Medium | Curated registry with verified open-license models only |
| Network partition / low availability | Medium | Gossip anti-entropy + fallback API option |
| Cold start for rare models | Low | Anchor nodes + background seeding |
| Thermal throttling degrading service | Low | Benchmark burn-in + mid-job re-routing |

***

## Conclusion

OpenGrid is not a speculative idea — every individual component described in this specification has been independently validated in production or peer-reviewed research. Petals demonstrated that 70B-parameter models run at interactive speeds on pooled internet-connected consumer GPUs[^3][^64]. ARIA Protocol proved that 1-bit models achieve 89 tokens/sec on a single consumer CPU[^9][^10]. TOPLOC makes trustless verification practical at under 260 bytes per 32 tokens[^13][^14]. BOINC showed that millions of volunteers will donate compute for the right reward structure[^1][^2]. The exo framework demonstrated zero-configuration P2P device clustering[^5][^6]. What has not yet existed is a single project that assembles all of these pieces into a consumer-friendly, credit-based, permissionless inference network with a first-class local memory system and a drop-in OpenAI-compatible API.

OpenGrid is that project.

---

## References

1. [[PDF] A Platform for Volunteer Computing - BOINC](https://boinc.berkeley.edu/boinc_a_platform_for_volunteer_computing.pdf) - When the BOINC client requests jobs from a server, it includes a list of platforms it supports, as w...

2. [[DOC] A Runtime System for Volunteer Computing - BOINC](https://boinc.berkeley.edu/boinc_papers/api/text.doc) - A BOINC client program runs on the volunteered hosts and manages the execution of applications. Toge...

3. [Distributed Inference and Fine-tuning of Large Language Models ...](https://arxiv.org/abs/2312.08361) - Large language models (LLMs) are useful in many NLP tasks and become more capable with size, with th...

4. [bigscience-workshop/petals: Run LLMs at home, BitTorrent ... - GitHub](https://github.com/bigscience-workshop/petals) - 🌸 Run LLMs at home, BitTorrent-style. Fine-tuning and inference up to 10x faster than offloading - b...

5. [GitHub - exo-explore/exo: Run your own AI cluster at home with everyday devices 📱💻 🖥️⌚](https://github.com/exo-explore/exo?tab=readme-ov-file) - Run your own AI cluster at home with everyday devices 📱💻 🖥️⌚ - exo-explore/exo

6. [Exo: Run your own AI cluster at home using everyday devices, …](https://jimmysong.io/ai/exo/) - exo: Run your own AI cluster at home using everyday devices, supporting distributed inference and a ...

7. [The First Globally Distributed Reinforcement Learning Training of a ...](https://www.primeintellect.ai/blog/intellect-2) - Prime-RL: Our Decentralized Training Framework. Our INTELLECT-2 infrastructure mainly consists of th...

8. [What Is Prime Intellect? Decentralized AI Protocol Explained](https://www.gate.com/learn/articles/open-ai-founding-members-invest-a-quick-dive-into-the-decentralized-ai-breakthrough-prime-intellect/7323) - Prime Intellect is a decentralized peer-to-peer AI computing protocol backed by OpenAI founding memb...

9. [I built P2P network where every CPU becomes an AI inference node ...](https://news.ycombinator.com/item?id=46980924) - Hey HN, I've been working on ARIA Protocol — an open-source P2P network for distributed AI inference...

10. [Distributed 1-bit LLM inference over P2P - 50 nodes validated, 100 ...](https://www.reddit.com/r/LocalLLaMA/comments/1sbsjfd/distributed_1bit_llm_inference_over_p2p_50_nodes/) - It's a peer-to-peer distributed inference system built specifically for 1-bit quantized models (tern...

11. [ARIA – P2P distributed inference protocol for 1-bit LLMs on CPU](https://news.ycombinator.com/item?id=46887868)

12. [TOPLOC: is a novel method for verifiable inference that ... - GitHub](https://github.com/PrimeIntellect-ai/toploc) - TOPLOC leverages locality sensitive hashing of intermediate activations to verify that LLM providers...

13. [TOPLOC: A Locality Sensitive Hashing Scheme for Trustless ...](https://www.primeintellect.ai/blog/toploc) - We introduce TOPLOC, a novel method for verifiable inference. TOPLOC employs a compact locality sens...

14. [A Locality Sensitive Hashing Scheme for Trustless Verifiable Inference](https://icml.cc/virtual/2025/poster/46281) - In this work, we propose TOPLOC, an inference verification method that can reduce the storage cost o...

15. [A Locality Sensitive Hashing Scheme for Trustless Verifiable Inference](https://arxiv.org/abs/2501.16007) - TOPLOC leverages a compact locality sensitive hashing mechanism for intermediate activations which c...

16. [Run 70B LLMs in 4 Bits — INT8, GPTQ, AWQ & GGUF [2026]](https://www.meta-intelligence.tech/en/insight-quantization) - Run 70B parameter LLMs on consumer GPUs using quantization. Complete guide to INT8, GPTQ, AWQ, NF4, ...

17. [Quantization for LLM Inference: From FP16 to INT4 | Scaling Thoughts](https://scalingthoughts.com/blog/quantization-for-llm-inference/) - Quantization cuts memory and speeds up inference. But naive 8-bit quantization breaks at 6.7B+ param...

18. [Tokenomics of decentralized GPU computing…](https://www.expresscomputer.in/news/tokenomics-of-decentralized-gpu-computing/134232/) - Fundamentally, tokenomics examines how digital tokens are created, managed, and distributed to estab...

19. [2. The Volunteer Computing Network | White Paper - GitBook](https://netmind-power.gitbook.io/white-paper/2.-the-volunteer-computing-network) - NetMind power is built upon the concept of Volunteer Computing. Volunteer Computing is a system that...

20. [GitHub - raj-poojary/exo-distributed-inference: Run your own AI cluster at home with everyday devices 📱💻 🖥️⌚](https://github.com/raj-poojary/exo-distributed-inference) - Run your own AI cluster at home with everyday devices 📱💻 🖥️⌚ - raj-poojary/exo-distributed-inference

21. [BOINC overview - GitHub](https://github.com/BOINC/boinc/wiki/BOINC-overview) - BOINC has a client/server architecture. The server distributes jobs, while the client runs on 'worke...

22. [[PDF] SETI@home: an experiment in public-resource computing - DISCO](https://disco.ethz.ch/courses/fs10/seminar/paper/michael-7.pdf) - The client program repeatedly gets a work unit from the data/result server, analyzes it, then return...

23. [SETI@home: An Experiment in Public-Resource Computing](https://setiathome.berkeley.edu/sah_papers/cacm.php) - The signal data is divided into fixed-size work units that are distributed, via the Internet, to a c...

24. [A Federated Approach to Train and Deploy Machine Learning Models](https://docs.edgeimpulse.com/projects/expert-network/federated-learning-raspberry-pi) - This profiling gives us an estimate of the RAM, ROM, and inference time of the model on a target har...

25. [[2410.16144] 1-bit AI Infra: Part 1.1, Fast and Lossless BitNet b1.58 ...](https://arxiv.org/abs/2410.16144) - We develop a set of kernels to support fast and lossless inference of ternary BitNet b1.58 LLMs on C...

26. [GitHub - microsoft/BitNet: Official inference framework for 1-bit LLMs](https://github.com/microsoft/BitNet) - bitnet.cpp is the official inference framework for 1-bit LLMs (eg, BitNet b1.58). It offers a suite ...

27. [Petals - Run large language models at home or in a distributed ...](https://jimmysong.io/en/ai/petals/) - Run large language models at home or in a distributed swarm for collaborative inference and fine-tun...

28. [Quantization: INT8 and INT4 | EngineersOfAI](https://engineersofai.com/docs/llms/llm-inference/quantization-int8-int4) - Master LLM quantization techniques — from LLM.int8() to GPTQ and AWQ — to run large models on commod...

29. [What Is Quantization in LLMs: Techniques, Trade-offs & GPU VRAM ...](https://deploybase.ai/articles/what-is-quantization-llm) - Understand LLM quantization techniques: INT8, INT4, GPTQ, AWQ. Quality vs speed trade-offs and GPU m...

30. [A Practical Guide to LLM Quantization (int8/int4) | Hivenet](https://www.hivenet.com/post/llm-quantization-guide) - What quantization is, when to use int8 or int4, how it affects quality, and a simple evaluation loop...

31. [Master KV cache aware routing with llm-d for efficient AI inference](https://developers.redhat.com/articles/2025/10/07/master-kv-cache-aware-routing-llm-d-efficient-ai-inference) - Learn how llm-d's KV cache aware routing reduces latency and improves throughput by directing reques...

32. [KV-Cache Wins You Can Feel: Building AI-Aware LLM Routing on ...](https://research.ibm.com/publications/kv-cache-wins-you-can-feel-building-ai-aware-llm-routing-on-kubernetes) - KV-Cache Wins You Can Feel: Building AI-Aware LLM Routing on Kubernetes for KubeCon EU 2026 by Tyler...

33. [Router Guide | NVIDIA Dynamo Documentation](https://docs.nvidia.com/dynamo/latest/user-guides/kv-cache-aware-routing) - Enable KV-aware routing using Router for Dynamo deployments

34. [The Results: A Leap In...](https://docs.d.run/en/blogs/2025/kvcache-wins-you-can-see) - d.run (DaoCloud Runs Intelligence)，揭示一个新一代软件体系下的全新算力世界，让算力更自由。

35. [A Crash Course in P2P - ByteByteGo Newsletter](https://blog.bytebytego.com/p/a-crash-course-in-p2p) - Distributed Hash Tables (DHTs): It is a decentralized method for peer discovery commonly used in P2P...

36. [Use a DHT for a gossip protocol? - Stack Overflow](https://stackoverflow.com/questions/50485796/use-a-dht-for-a-gossip-protocol) - I'm trying to implement a p2p network working on a Kademlia DHT. I want to be able to gossip a messa...

37. [GitHub - learning-at-home/hivemind: Decentralized deep learning in PyTorch. Built to train models on thousands of volunteers across the world.](https://github.com/learning-at-home/hivemind/tree/master) - Decentralized deep learning in PyTorch. Built to train models on thousands of volunteers across the ...

38. [[PDF] OpenDiLoCo: An Open-Source Framework for Globally Distributed ...](https://arxiv.org/pdf/2407.07852.pdf) - Unlike the torch.distributed implementation, our. Hivemind implementation wraps both optimizers into...

39. [GitHub - 50RC3/vAIn_p2p_AGI: This repository implements a peer-to-peer decentralized topology & a reputation-based tier system. It leverages federated learning for AI model training. Using DHT for peer discovery, proof-of-stake for voting, and multi-agent systems for model updates. With blockchain for tokenomics and smart contracts, providing a robust AGI Dev framework .](https://github.com/50RC3/vAIn_p2p_AGI) - This repository implements a peer-to-peer decentralized topology & a reputation-based tier system. I...

40. [ELI5: Initial node discovery on a decentralized peer to peer network.](https://www.reddit.com/r/explainlikeimfive/comments/2ay5zm/eli5_initial_node_discovery_on_a_decentralized/) - The first time a client attempts to connect to a decentralized p2p network, the clients goes through...

41. [Gossip Protocols: Spreading Information in Distributed Systems with ...](https://blogs.cornell.edu/info2040/2019/09/20/gossip-protocols-spreading-information-in-distributed-systems-with-peer-to-peer-communications/) - The idea is that information will spread from “infected” nodes to their peers, so information will s...

42. [Revisiting Gossip Protocols: A Vision for Emergent Coordination in ...](https://arxiv.org/html/2508.01531v1) - Gossip protocols follow a peer-to-peer, symmetric interaction model. Any agent may initiate exchange...

43. [How to Implement Gossip Protocol for Distributed Systems Using Go](https://www.linkedin.com/pulse/how-implement-gossip-protocol-distributed-systems-using-agarwal-lfnwc) - Discover how the Gossip Protocol powers scalable, fault-tolerant distributed systems like Cassandra ...

44. [A distributed inference framework with dynamic scheduling capability](https://www.sciencedirect.com/science/article/abs/pii/S0167739X2400400X) - In this study, we propose DIDS, a distributed inference framework with dynamic scheduling capability...

45. [llm-d: Kubernetes-native distributed inferencing - Red Hat Developer](https://developers.redhat.com/articles/2025/05/20/llm-d-kubernetes-native-distributed-inferencing) - llm-d delivers Kubernetes-native distributed inference with advanced optimizations, reducing latency...

46. [A Framework for Node-Level Fault Tolerance in Distributed Real-time Systems](https://www.cse.chalmers.se/~johan/publications/Aidemark_DSN05.pdf)

47. [Adaptive fault tolerance mechanisms for ensuring high availability of ...](https://www.nature.com/articles/s41598-025-25590-4) - To enhance the reliability and scale of digital twins in the context of distributed edge computing, ...

48. [[PDF] PETALS: Collaborative Inference and Fine-tuning of Large Models](https://aclanthology.org/2023.acl-demo.54.pdf)

49. [LOGIC: Trustless Inference through Log-Probability Verification](https://inference.net/blog/logic/) - We introduce LOGIC, a practical method for verifying inference in decentralized GPU networks. LOGIC ...

50. [Solving the inference problem for open source AI projects with ...](https://github.blog/ai-and-ml/llms/solving-the-inference-problem-for-open-source-ai-projects-with-github-models/) - How using GitHub's free inference API can make your AI-powered open source software more accessible.

51. [An Incentive System for Volunteer Computing - BOINC](https://boinc.berkeley.edu/boinc_papers/credit/text.php) - In this paper we discuss the design of the credit accounting system in BOINC (Berkeley Open Infrastr...

52. [AI Design Reviews: Preventing LLM Data Leakage and Privacy Risks](https://iosentrix.com/blog/ai-design-review-llm-security-and-compliance) - This blog explores LLM data leakage prevention strategies and how structured AI privacy controls emb...

53. [Identifying and Mitigating Privacy Risks Stemming from Language ...](https://arxiv.org/html/2310.01424v2) - We present the first SoK on data privacy for LLMs. We (i) identify a taxonomy of salient dimensions ...

54. [[PDF] intelligent proactive fault tolerance at the edge - arXiv](https://arxiv.org/pdf/2302.05336.pdf) - In order to tackle this challenge we propose a composite deep learning architecture that predicts th...

55. [Cold Start Latency In LLM Inference: Causes, Metrics & Fixes](https://acecloud.ai/blog/cold-start-latency-llm-inference/) - Cold start latency is a deployment bottleneck that turns GPU capacity into startup delay during infe...

56. [Reducing Cold Start Latency for LLM Inference with NVIDIA Run:ai ...](https://developer.nvidia.com/blog/reducing-cold-start-latency-for-llm-inference-with-nvidia-runai-model-streamer/) - Deploying large language models (LLMs) poses a challenge in optimizing inference efficiency. In part...

57. [Tokenized Compute Credits - OpenxAI Docs](https://docs.openxai.org/tokenomics-and-economic-design/tokenized-compute-credits) - OpenxAI introduces tokenized compute assets such as tGPU and tCPU. Credits are minted against verifi...

58. [[PDF] Design Time Reliability Analysis of Distributed Fault Tolerance ...](https://users.ece.cmu.edu/~koopman/pubs/latronico05_dsn_rel_analysis_distrft.pdf) - Fault diagnosis procedures aim to keep the num- ber of active faults within the bounds of the maximu...

59. [[PDF] Kelips∗: Building an Efficient and Stable P2P DHT Through ...](http://iptps03.cs.berkeley.edu/final-papers/kelips.pdf)

60. [Scheduling Inference Workloads on Distributed Edge Clusters with ...](https://arxiv.org/abs/2301.13618) - In this paper, we focus on the problem of scheduling inference queries on DNN models in edge network...

61. [Learning@home hivemind](https://learning-at-home.github.io) - A library to train large neural networks across the internet. Imagine training GPT-3 on thousands of...

62. [Petals: distributed shared GPU running and fine-tuning of large language models, sharing GPU resources like a BitTorrent network](https://www.aisharenet.com/en/petals/) - General Introduction Petals is an open source project developed by the BigScience Workshop to run La...

63. [Parallelization Strategies for Dense LLM Deployment - arXiv](https://arxiv.org/html/2603.05692v1) - Our empirical evaluations reveal that Tensor Parallelism (TP) improves the latency objectives while ...

64. [Petals: decentralized inference and finetuning of large language ...](https://research.yandex.com/blog/petals-decentralized-inference-and-finetuning-of-large-language-models) - Large language models are among the most significant recent advances in machine learning. Still, lev...

