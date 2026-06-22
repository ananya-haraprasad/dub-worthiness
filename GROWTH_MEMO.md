# Growth memo — Will It Travel? → Sarvam Studio

*Built and tested on real Indian YouTube content (Tamil, Hindi, English). The
numbers below are from a small pilot (3 videos run end-to-end) plus what fell
out of building the five engines. Treat the magnitudes as directional and the
mechanisms as the real finding.*

---

## 1. What I found

**a) Most short-form creator content dubs *cleanly* — feasibility is not the
bottleneck.** A Tamil cricket-reaction, a Hindi cooking clip, and an English
skincare clip all scored **88–94 ("Dubs cleanly")**. Plain, instructional, or
narrative speech round-trips with little meaning loss. The implication is
counter-intuitive: for the bulk of the catalogue, *whether it can be dubbed well*
is already a solved "yes." The real friction is **decision and cost**, not
quality. A tool that tells a creator "this will dub cleanly, here's the upside"
removes the hesitation, not a technical blocker.

**b) Meaning-loss is strongly *direction-asymmetric* — Tamil is the expensive
pair.** The English→Hindi round-trip lost **~2%** of meaning; Hindi→Tamil **~19%**;
**Tamil→(English/Hindi) ~29%**. Tamil is consistently the hardest direction.
That's not a bug — it's a pricing and staffing signal: jobs *out of* Tamil (or
*into* Tamil) need more adaptation budget and a human in the loop, while
English→Hindi can be near-automated.

**c) The riskiest content is exactly where automated QA is weakest.** The idiom
engine is tuned to Roman/code-mixed text, so native-script (Devanagari/Tamil)
slang is under-detected — and slang-dense comedy is also the genre that benchmarks
worst (entertainment ≈ 54 vs education ≈ 82). So comedy/entertainment is
*doubly* hard: hardest to dub **and** hardest to auto-QA. That argues for routing
those jobs to human adaptors, not for pretending the score covers them.

**d) Cultural references are sparse in casual content but cluster in
topical/sports/news.** The clips surfaced near-zero references except the
sports clip ("Gujarat Titans"). Reference risk is a *genre* property — it tells
Studio which jobs need *reference localisation* (swapping JEE→TNPSC, RCB→CSK),
not just translation.

## 2. Which creator segments Studio should chase first

Backed by the scores and the benchmark table baked into the tool:

1. **Education / finance / how-to / instructional (score 80–95).** Cleanest
   dubs, lowest adaptation cost, highest margin, easiest to automate at volume.
   **Land here first** — fastest time-to-value and the best proof-of-quality
   case studies.
2. **Hindi- and English-source explainer creators expanding outward.** Low
   semantic loss in those directions → near-automated pipeline → high throughput.
3. **Defer comedy / entertainment / reaction.** Highest semantic loss, slang-
   and timing-dependent, weakest auto-QA. Worth it for reach, but only with a
   priced human-adaptation tier — don't lead with it or you'll set a bad
   quality precedent.

## 3. The GTM insight (the contrarian bet)

**You're not selling dubbing. You're selling audience expansion — a second and
third channel from the creator's existing work.** Creators don't want "a dub";
they want reach they didn't have to re-shoot for. Re-frame the entire pitch from
*cost-per-minute of audio* to *new audience unlocked*.

Concretely: **the "Will It Travel?" score *is* the wedge.** Ship it as a free,
public widget. A creator pastes a video and instantly sees "this dubs cleanly
into Hindi and Tamil — that's an estimated N× audience you're leaving on the
table." That single screen does three jobs at once:
- **Lead-gen** for the creator (an upside number they can't unsee),
- **Qualification** for Studio (high score = cheap to serve = high margin; the
  tool literally pre-sorts the funnel by gross margin), and
- **Honest expectation-setting** (a "Heavy localisation" verdict pre-sells the
  premium tier instead of producing a disappointed customer).

Pricing should follow outcomes, not minutes: tiered by the *score* (clean dubs
priced as commodity throughput; low-score jobs priced as adaptation projects).

## 4. What I'd do in week one at Studio

**Experiment 1 — Score-as-funnel.** Embed the "Will It Travel?" score as a free tool
on the Studio site.
- *Hypothesis:* creators who self-serve a score + audience-expansion estimate
  convert to a paid dub at a materially higher rate than cold outreach.
- *Metric:* score-completed → paid conversion rate vs a cold-outreach control;
  CAC by channel.

**Experiment 2 — Lead with high-score segments.** Target education/finance/how-to
creators (predicted score ≥ 80) for the first 50 jobs.
- *Hypothesis:* high-score jobs deliver faster, need fewer revisions, and carry
  higher gross margin than entertainment jobs.
- *Metric:* avg revisions/job, delivery time, and gross margin **segmented by
  score band**; CSAT.

**Experiment 3 — A priced "adaptation tier" for low-score / Tamil-out jobs.**
Route anything scoring < 60 (or any Tamil-source job, given ~29% loss) to a
human-in-the-loop adaptation tier at a premium SLA/price.
- *Hypothesis:* matching effort to difficulty (instead of one flat dub product)
  cuts revision loops and churn on hard jobs.
- *Metric:* revision rate and 90-day retention for low-score jobs on the standard
  pipeline vs the adaptation tier.

---

*The build proves the work can be done; this memo is the argument that the
score is not a feature — it's the go-to-market.*
