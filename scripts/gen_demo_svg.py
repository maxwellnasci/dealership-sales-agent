"""Gera um SVG animado estilo terminal a partir de um transcript.

Produz `docs/demo.svg` — uma animação (CSS) que "revela" a conversa linha a
linha, como se o terminal estivesse rodando a demo. Autocontido (sem libs
externas, sem rede), reproduz no README do GitHub quando embutido via <img>.

Rodar:  python scripts/gen_demo_svg.py
"""

from __future__ import annotations

from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "demo.svg"

# Cores (paleta GitHub dark).
BG = "#0d1117"
FG = "#c9d1d9"
DIM = "#8b949e"
LEAD = "#58a6ff"       # azul
AGENT = "#3fb950"      # verde
REP = "#db6d28"        # laranja (vendedor humano)
HILITE = "#d29922"     # âmbar (anotações/garantias)

# Transcript: (cor, texto). Texto real capturado da demo com DeepSeek/OpenRouter.
LINES: list[tuple[str, str]] = [
    (DIM, "$ python scripts/demo.py"),
    (DIM, "Modo do LLM: LLM real (DeepSeek via OpenRouter)"),
    (FG, ""),
    (LEAD, "🧑 Lead:  Oi, boa tarde!"),
    (AGENT, "🤖 Léo:   Boa tarde! Aqui é o Léo, da revenda. 😊 Como posso te ajudar?"),
    (FG, ""),
    (LEAD, "🧑 Lead:  Procuro um SUV até 140 mil, pra família"),
    (AGENT, "🤖 Léo:   Ótimas opções de SUV nessa faixa:"),
    (AGENT, "          • Jeep Renegade 2021 (61.000 km) — R$ 99.900,00"),
    (AGENT, "          • Fiat Pulse 2023 (22.000 km) — R$ 99.900,00"),
    (AGENT, "          • VW Nivus 2022 (29.000 km) — R$ 114.900,00"),
    (AGENT, "          Quer simular o financiamento de alguma?"),
    (FG, ""),
    (LEAD, "🧑 Lead:  Gostei do primeiro. Como fica o financiamento?"),
    (AGENT, "🤖 Léo:   Jeep Renegade 2021 — entrada R$ 19.980,00,"),
    (AGENT, "          48x de R$ 2.552,80 (total R$ 142.514,40)."),
    (HILITE, "          ↳ memória: lembrou do carro entre mensagens ✓"),
    (FG, ""),
    (LEAD, "🧑 Lead:  Dá pra agendar um test-drive?"),
    (AGENT, "🤖 Léo:   Consigo agendar para 19/10 às 09h00. Esse horário fica bom?"),
    (HILITE, "          ↳ agenda: exatamente 1 horário, sem ambiguidade ✓"),
    (FG, ""),
    (LEAD, "🧑 Lead:  Consigo desconto? Quero falar com um vendedor"),
    (HILITE, "⏸  [handoff] grafo pausado via interrupt() — aguardando humano"),
    (REP, "🧑‍💼 Carlos: Aqui é o Carlos. Consigo R$ 3.000 à vista — fechamos?"),
    (AGENT, "🤖 (resume) mensagem do vendedor repassada ao lead ✓"),
    (FG, ""),
    (HILITE, "✓ 32 testes · anti-alucinação (Pydantic) · checkpointing · HITL"),
]

ROW_H = 26
PAD_X = 22
PAD_TOP = 54
FONT_SIZE = 15
STEP = 0.42          # atraso entre linhas (s)
REVEAL = 0.45        # duração do fade-in de cada linha (s)
TAIL = 3.0           # tempo parado no fim antes de reiniciar (s)

n = len(LINES)
total = n * STEP + TAIL + 1.0
width = 860
height = PAD_TOP + n * ROW_H + 20


def build() -> str:
    # Keyframes: aparece no seu delay, segura, some no fim para reiniciar o loop.
    hold_start = (REVEAL / total) * 100
    hold_end = ((total - 0.6) / total) * 100
    keyframes = (
        "@keyframes reveal{"
        "0%{opacity:0;transform:translateY(3px)}"
        f"{hold_start:.2f}%{{opacity:1;transform:translateY(0)}}"
        f"{hold_end:.2f}%{{opacity:1;transform:translateY(0)}}"
        "100%{opacity:0}"
        "}"
    )
    rows = []
    for i, (color, text) in enumerate(LINES):
        delay = i * STEP
        y = PAD_TOP + i * ROW_H
        content = escape(text) or " "
        rows.append(
            f'<text x="{PAD_X}" y="{y}" fill="{color}" '
            f'style="animation:reveal {total:.2f}s ease both infinite;'
            f'animation-delay:{delay:.2f}s">{content}</text>'
        )
    text_block = "\n    ".join(rows)

    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" \
viewBox="0 0 {width} {height}" font-family="ui-monospace,'SF Mono',Menlo,Consolas,monospace" \
font-size="{FONT_SIZE}">
  <style>
    {keyframes}
    text {{ white-space: pre; }}
  </style>
  <rect width="{width}" height="{height}" rx="10" fill="{BG}"/>
  <rect width="{width}" height="36" rx="10" fill="#161b22"/>
  <rect y="26" width="{width}" height="10" fill="#161b22"/>
  <circle cx="20" cy="18" r="6" fill="#ff5f56"/>
  <circle cx="40" cy="18" r="6" fill="#ffbd2e"/>
  <circle cx="60" cy="18" r="6" fill="#27c93f"/>
  <text x="{width // 2}" y="23" fill="{DIM}" text-anchor="middle" font-size="13">\
dealership-sales-agent — demo</text>
    {text_block}
</svg>
"""


def main() -> None:
    OUT.write_text(build(), encoding="utf-8")
    print(f"SVG gerado: {OUT}  ({OUT.stat().st_size} bytes, {n} linhas, loop {total:.1f}s)")


if __name__ == "__main__":
    main()
