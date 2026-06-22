# Growth memo — Will It Travel? → Sarvam Studio

*Built and stress-tested end-to-end on real Indian content (Tamil, Hindi,
English) running on Sarvam's own stack: Saarika (STT) → Mayura (translation) →
Bulbul (TTS). Numbers below are from a small pilot plus a deliberate failure-mode
sweep. Treat magnitudes as directional; the mechanisms are the real finding.*

---

## 1. What I found

**a) With a translator as good as Mayura, most plain content dubs *cleanly* —
feasibility is not the bottleneck.** Instructional, narrative, and conversational
speech round-trips with very little meaning loss (clean clips ~0.03–0.05 loss).
The implication is counter-intuitive: for the bulk of a catalogue, *whether it
can be dubbed well* is already "yes." The friction is **decision and cost**, not
technical quality. A tool that says "this travels cleanly, here's the upside"
removes hesitation, not a blocker.

**b) The real bottleneck is form and figurative language, not meaning or any one
language.** I ran failure hypotheses on purpose. Dense idioms (~0.27 loss) and
shayari (~0.15) register real meaning loss and score lower. But a **poem**
round-trips at ~0.07 — its *meaning* survives even though its rhyme and rhythm,
the whole point, are destroyed. So the expensive jobs aren't a particular
language pair; they're **wordplay, poetry, comedy timing, and culture-bound
references** — content where *form* carries the value.

**c) Translating is not localizing — and that gap is the product.** The sharpest,
most defensible signal isn't the aggregate score; it's the term-level view: live
Mayura output next to the right call. "Baby oil" must stay "baby oil," not become
"child oil" in Tamil; "cuticle" should localize to a native word, not transliterate.
This is the craft layer Studio sells, made visible.

**d) The riskiest content is exactly where automated QA is weakest.** Native-
script slang and poetic wordplay are both hardest to dub *and* hardest to auto-
check (the idiom/culture dictionaries are tuned for Roman/code-mixed text). That
argues for routing those jobs to human adaptors, not for pretending a score
covers them.

## 2. Which creator segments Studio should chase first

1. **Education / finance / how-to / instructional.** Cleanest dubs, lowest
   adaptation cost, highest margin, easiest to automate at volume. **Land here
   first** — fastest time-to-value and the best proof-of-quality case studies.
2. **Explainer creators expanding outward.** Plain declarative speech travels in
   any of the three directions → near-automated pipeline → high throughput.
3. **Defer comedy / poetry / reaction.** Wordplay- and timing-dependent, weakest
   auto-QA. Worth it for reach, but only behind a priced human-adaptation tier —
   don't lead with it or you'll set a bad quality precedent.

## 3. The GTM insight (the contrarian bet)

**You're not selling dubbing. You're selling audience expansion — a second and
third channel from work the creator already shot.** Re-frame the pitch from
*cost-per-minute of audio* to *new audience unlocked*.

**The "Will It Travel?" score is the wedge.** Ship it as a free, public widget. A
creator pastes a video and sees "this travels cleanly into Hindi and Tamil —
that's an estimated N× audience you're leaving on the table." One screen does
three jobs:
- **Lead-gen** (an upside number they can't unsee),
- **Qualification** (high score = cheap to serve = high margin; the funnel
  pre-sorts itself by gross margin), and
- **Honest expectation-setting** (a "needs heavy adaptation" verdict pre-sells the
  premium tier instead of producing a disappointed customer).

Price by outcome, not minutes: clean dubs as commodity throughput; low-score,
wordplay-heavy jobs as priced adaptation projects.

## 4. What I'd do in week one at Studio

**Experiment 1 — Score-as-funnel.** Embed the score as a free tool on the Studio
site. *Hypothesis:* self-serve score + audience-expansion estimate converts to a
paid dub at a higher rate than cold outreach. *Metric:* score-completed → paid
conversion vs cold-outreach control; CAC by channel.

**Experiment 2 — Lead with high-score segments.** Target education/finance/how-to
creators for the first 50 jobs. *Hypothesis:* high-score jobs deliver faster,
need fewer revisions, higher gross margin. *Metric:* revisions/job, delivery
time, gross margin **by score band**; CSAT.

**Experiment 3 — A priced adaptation tier for low-score jobs.** Route wordplay-
and idiom-heavy jobs to a human-in-the-loop tier at a premium SLA. *Hypothesis:*
matching effort to difficulty cuts revision loops and churn. *Metric:* revision
rate and 90-day retention, standard pipeline vs adaptation tier.

## 5. How I validated it, and what it can't do (say this out loud)

I treated my own model as something to break, not defend:
- **It measures meaning travel, not form or fluency.** A poem scores "clean"
  because its meaning round-trips; the lost rhyme is invisible to back-translation.
- **Calibration must track the translator.** Moving from a generic engine to
  Mayura dropped round-trip losses across the board and flattened the score; I
  recalibrated the drift floor to Mayura's measured baseline to restore the
  spread. A constant tuned for one translator is wrong for another.
- **Proper-noun errors slip through** (Mayura rendered "Messi" as "mess"; the
  round-trip didn't catch it). I built and then *removed* a zero-frequency
  detector because it false-flagged legitimate terms like "golgappa" — a fragile
  rule is worse than a stated blind spot.
- **The aggregate score is a transparent heuristic, not a model trained on
  labeled dub quality.** The reliable layer is the concrete localization gap.

**Biggest next upgrade:** label ~500 human-rated dubs and *learn* the signal
weights instead of hand-tuning them. Then add fluency/naturalness scoring (the
gap the round-trip can't see) and named-entity consistency checks.

---

*The build proves the work can be done; this memo argues the score isn't a
feature — it's the go-to-market. And it's honest about exactly where the
automation stops and a human adaptor earns their fee.*
