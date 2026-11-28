# Optional LM Studio provider for maintained Chroma Qt

Choose **lmstudio** only when testing an explicitly selected, independently verified local deployment. The normal GUI router has no LM Studio deployment enrolled. It reports an unavailable provider and preserves retrieved evidence. Search Only remains provider-free. Selecting LM Studio suppresses generation-based query rewriting and Groq model refresh.

The optional runtime is Windows Git Bash, Node.js 22 or newer, the installed LM Studio CLI, and the existing Foreman single slot. Install the pinned client libraries with npm ci --ignore-scripts from this directory when preparing a separate verification environment. The runtime uses the existing loopback service on 127.0.0.1:1234 and an instance-specific handle for the already-loaded GGUF. It performs no model load or configuration write. The app uses the standard Foreman slot directory and refuses operation when a slot override is present.

The maintained PyQt worker uses the shared admission and structural publication paths. The bridge refuses unbound identity and context, reserves the full output budget, prohibits alternate providers and keeps the slot while cancelled generation can still be running. A transport failure after dispatch requires runner idle confirmation before lease release. Only one request using the shared slot can run at a time. Clients outside that coordination can still contend, so the bridge checks the runner's idle and queue status before dispatch.

The pinned SDK and a fake loopback peer can verify the exact render, tokenize and raw dispatch sequence. That test cannot establish the installed engine's special-token handling. A real deployment needs separate evidence tying the exact app build, inference engine, resident model and active template to a reproducible raw framing rule. The deployment's render-probe digest, instance reference, effective context and output ceiling must agree at runtime. Verify framing before requesting permission to execute Q-20261128-001. Keep this provider unenrolled while that proof or any required smoke result is missing.

## Synthetic verification

With the optional libraries installed, run each case in a fresh process.

    node --conditions=browser bridge.test.mjs normal
    node --conditions=browser bridge.test.mjs tool
    node --conditions=browser bridge.test.mjs truncated
    node --conditions=browser bridge.test.mjs unclosed
    node --conditions=browser bridge.test.mjs cancel
    node --conditions=browser bridge.test.mjs ambiguous
    node --conditions=browser bridge.test.mjs changed

The fake peer uses an ephemeral loopback port and no model. The Python synthetic tests use the real router admission and PyQt worker with fixture inputs and synthetic outputs. The fixture file is shared with the proposed live packet. Its insufficient-evidence case demonstrates that an irrelevant verbatim excerpt can pass span validation and still fail the smoke's relevance criterion. No real collection is needed.

See the [locked #22 packet](https://github.com/sriharshaguthikonda/easy-local-rag/pull/78) and [separate activation](https://github.com/sriharshaguthikonda/easy-local-rag/pull/79).
