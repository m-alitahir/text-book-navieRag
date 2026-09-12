# TextbookNavieRAG

A local Retrieval-Augmented Generation (RAG) chatbot that answers questions about a textbook (built and tested on a Chemistry 11 / PCTB textbook), grounded strictly in the book's own content. Everything runs on your own machine through Ollama — no cloud API, no API keys, no data leaves your computer.

## What it does

- Extracts text from the textbook PDF and removes repeated boilerplate (headers/footers/watermarks) before chunking.
- Splits the text into overlapping chunks (paragraph → sentence → word → raw-character fallback for edge cases).
- Embeds every chunk locally and stores the vectors for search.
- Given a question, retrieves the most relevant chunks by cosine similarity, filtered by a calibrated similarity threshold so off-topic questions correctly return little or nothing.
- Cleans up the user's question first: fixes typos and normalizes chemical formula casing (e.g. `h2so4` → `H2SO4`) without altering the actual chemistry.
- Generates an answer constrained to only the retrieved context — if the book doesn't cover something, it says so instead of guessing from the model's general knowledge.
- Detects false premises: if a question assumes something the book directly contradicts, the answer leads with that correction instead of dancing around it.
- Formats chemistry notation properly in the reply (`H₂O`, `Na⁺`, `SO₄²⁻`) and strips out any LaTeX syntax that slips in, converting it to real Unicode.
- Is hardened against prompt injection — user input is always treated strictly as a question, never as an instruction to the assistant.
- Ships with a simple browser chat window so a non-technical person can just type and ask.

See `DEPENDENCIES.md` for the exact models and libraries used, and `textbook_navierag.py` for the full implementation (single file, chunking through chat UI).

## Setup & run

1. Install [Ollama](https://ollama.com) and pull the two models (names in `DEPENDENCIES.md`).
2. Install the Python dependencies listed in `DEPENDENCIES.md`.
3. Place your textbook PDF in the same folder and set its filename in `textbook_navierag.py` (`PDF_PATH`).
4. Run:
   ```
   python textbook_navierag.py
   ```
   First run builds the chunk/embedding index (a few minutes) and caches it to `chunk_embeddings.pkl`; later runs reuse the cache. It then launches the chat UI — open the local URL it prints (usually `http://127.0.0.1:7860`) and start asking questions.

## Problems faced and how they were solved

**Infinite loop while chunking.** Some PDF text had no spaces at all (extraction artifacts), so the word-splitter kept "splitting" the same block into itself forever. Fixed by adding a raw character-level fallback for text that has no spaces to split on.

**Watermark text was hijacking every search result.** The textbook had a boilerplate line ("Web version — not for sale") printed on every one of its 344 pages. Because it was so repetitive, it looked artificially "similar" to itself and dominated results. Fixed by detecting lines that repeat across most pages and stripping them before chunking.

**The pipeline was silently dying mid-run, every time.** A PDF font-encoding bug produced a stray Unicode character that Windows' default console encoding couldn't print, crashing the script with no clear error — and it always crashed before reaching the step that actually mattered. Fixed by forcing UTF-8 output encoding at startup, and traced the real cause by checking exactly where output stopped rather than guessing.

**The similarity threshold was a guess, not a measurement.** An arbitrary cutoff either let irrelevant chunks through or dropped good ones. Fixed by empirically comparing on-topic vs. off-topic query scores and calibrating the threshold to the gap actually observed, instead of picking a number.

**A crafted message hijacked the assistant.** A message like "ignore all previous instructions..." got the model to break character entirely. Fixed by hardening the system prompt to treat all user input strictly as a chemistry question, never as a command, and explicitly refuse anything that looks like an injection attempt.

**The model answered questions the book never covered.** Asked about something outside the textbook's scope, it filled the gap with its own pretrained knowledge instead of admitting the book didn't cover it. Fixed by tightening the grounding instructions to explicitly forbid filling in missing specifics even when the context is topically related.

**Query cleanup was silently dropping words.** Typo-correction on the question sometimes deleted legitimate words (e.g. "group 16 elements" losing "group"). Fixed by rewriting the cleanup instructions as an explicit numbered rule list that forbids adding, removing, or reordering words.

**Formulas typed in the wrong case didn't match anything.** A question like `h2so4` wouldn't retrieve the right chunks because real formulas are cased (`H2SO4`). Fixed by adding a rule that normalizes element-symbol casing only, without touching atom counts, subscripts, or charge — a casing fix, not a chemistry correction.

**A false-premise question got a confusing non-answer.** When a question assumed something the book directly contradicted, the model would just say "the context doesn't state that" instead of correcting the premise, even though the correct fact was right there in the retrieved chunks. Diagnosed by isolating the exact chunks involved and testing the same context directly against the model. Fixed with an explicit instruction (plus a worked example) requiring the answer to lead with "that premise is incorrect" whenever the context contradicts the question's assumption.

**Chemistry formulas showed up as raw LaTeX** (`$C_9H_8O_4$`) instead of readable notation. The model was copying that syntax directly from how the question was typed — telling it "don't use LaTeX" in the prompt didn't stop the copying behavior. Fixed with a deterministic Python function (`latex_to_unicode`) that converts LaTeX-style subscripts/superscripts to real Unicode characters, applied to both the incoming question and the outgoing answer — a code-level fix instead of relying on the model following an instruction.
