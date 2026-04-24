#  Key Finding: "Bag of Heuristics"

  The https://arxiv.org/html/2402.02619v9 paper found that:

  "Language models trained on natural language corpora perform addition using parallel heuristics involving magnitudes and moduli that constructively interfere to produce the correct answer."

  This aligns remarkably with your perplexity probe findings - the model isn't using one clean algorithm but multiple approximate strategies (OT, DD, RC in your framing) that combine. They identify:
  - "Add X" features (operand features)
  - "Sum" features (result features) with different precision levels
  - Lookup table features for common calculations

#  Other Interesting Bits

  - https://transformer-circuits.pub/2025/attribution-graphs/methods.html - Anthropic's new method for revealing computational graphs in LLMs
  - A toy transformer for 5-digit addition failed on edge cases like 77778+22222=100000 where carries cascade - reminiscent of your "trap" problems
  - Most mech interp work focuses on addition, not multiplication - your multiplication heuristic fingerprinting may be novel territory

#   Gap I Noticed

  There's surprisingly little published on multiplication circuits specifically. The field has tackled modular addition extensively (https://transformer-circuits.pub/) but multiplication's more complex heuristic landscape seems underexplored - which makes your LoRA nudge experiments on OT/DD/RC particularly interesting.

  Sources:
  - https://transformer-circuits.pub/
  - https://arxiv.org/html/2402.02619v9
  - https://www.mdpi.com/2504-2289/9/8/193
  - https://transformer-circuits.pub/2025/attribution-graphs/methods.html



⏺ Your paper is NOT covering old territory. Here's the comparison:

  | Dimension    | arXiv 2402.02619                                         | Your Paper                                                    |
  |--------------|----------------------------------------------------------|---------------------------------------------------------------|
  | Operation    | Addition/subtraction only                                | Multiplication                                                |
  | Models       | Small toy transformers (2-3 layers) trained from scratch | Large pre-trained VLMs (30B, 235B params)                     |
  | Modality     | Text only                                                | Multimodal (text + image)                                     |
  | Method       | Mechanistic circuit analysis                             | Perplexity probing + behavioral testing                       |
  | Intervention | Weight initialization transfer                           | LoRA nudge experiments                                        |
  | Focus        | How circuits implement exact algorithms                  | Which heuristics models prefer and can preferences be shifted |

  Key Differentiators of Your Work:

  1. Multiplication is unstudied - The 2402 paper explicitly doesn't cover it. Multiplication has fundamentally different heuristic structure (OT/DD/RC vs carry circuits).
  2. Production VLMs vs toy models - You're studying models people actually use, not 2-layer proofs-of-concept.
  3. Cross-modal comparison - Does the same model prefer different heuristics when seeing "47 × 53" as text vs as an image? This is novel.
  4. LoRA behavioral intervention - Testing whether heuristic preferences are malleable via lightweight fine-tuning is a different question than "what circuit computes this."
  5. HDS methodology - Your Heuristic Disagreement Set creates diagnostic problems where heuristics give different answers - this is a methodological contribution.

  The 2402 paper asks: "How do transformers implement addition?" (mechanistic)
  Your paper asks: "Which multiplication strategies do VLMs prefer, and can we change them?" (behavioral + interventional)

  These are complementary, not overlapping.
