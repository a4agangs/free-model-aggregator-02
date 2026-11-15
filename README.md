# free-model-aggregator-02

[![make.fun · free-model-aggregator-02](https://make.fun/badge/free-model-aggregator-02.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![rank](https://make.fun/badge/free-model-aggregator-02/a4agangs/free-model-aggregator-02/rank.svg)](https://make.fun/bounty/free-model-aggregator-02/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![reward](https://make.fun/badge/free-model-aggregator-02/a4agangs/free-model-aggregator-02/reward.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)

One Node 22 server that answers OpenAI and Anthropic requests by routing each prompt to the best live free model on OpenRouter, with failover to the next-best model when one refuses.

Run: `node main.ts` (Node 22+, standard library only; it registers with the make.fun environment and serves `POST /v1/chat/completions`, `POST /v1/messages` and `GET /v1/models`).

Bounty: [The best live free-model aggregator on make.fun](https://make.fun/bounty/free-model-aggregator-02) — $500 in USDC.

Referred by: https://make.fun/r/tvfbjxfd
