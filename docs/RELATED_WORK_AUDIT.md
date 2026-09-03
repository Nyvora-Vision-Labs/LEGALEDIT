# Recent related work for LegalEdit (2024--2026)

Compiled 2026-09-01. Verified twice. Every entry was first checked by fetching
its arXiv abstract page, then re-checked against the arXiv Atom API
(`export.arxiv.org/api/query`), diffing the title and the full author list of
each entry against the record programmatically. Journal placements were
confirmed through CrossRef. Anything that could not be verified this way is not
in the list.

**Corrections the second pass found**, all now fixed in `refs.bib`:

- `bean2025construct` had **41 authors instead of 42**: Emma Beharry was dropped
  and everyone after position 11 was shifted by one. The author field has been
  regenerated from the API record.
- `you2026plainqafact` was dated 2025 from the arXiv submission. CrossRef gives
  the journal record as *Journal of Biomedical Informatics* 178:105019, June
  **2026**, so the key and year were changed and the citation in `main.tex`
  updated.
- `korobeynikova2025detect` was dated 2026 from the v2 revision. arXiv
  2510.22212 is October **2025**, so the key and year now match the identifier.
- `magesh2025hallucinationfree` gained its full journal reference from CrossRef
  (DOI 10.1111/jels.12413): *Journal of Empirical Legal Studies* 22(2):216--242.

Two entries flagged by the automated name diff, `Rystr{\o}m` in Bean et al. and
`{\c{C}}a{\u{g}}r{\i} {\c{C}}{\"o}ltekin` in Zhang and Çöltekin, are correct
LaTeX escapes for Rystrøm and Çağrı Çöltekin; the mismatch was in the
normaliser, not the bibliography.

**Third pass (2026-09-02) re-verified every entry in `refs.bib`, not only the
recent ones**, by opening the arXiv abstract page, ACL Anthology entry or
CrossRef record for each of the 61 references and diffing the title, author
list, venue, year and page range against the bibliography. Four errors were
found and fixed:

- `zhou2024lemaj` was wrong in two ways. Seven of the eleven given names were
  fabricated: the Anthology record for `2025.nllp-1.23` gives Sujeong Cha,
  Arijit Ghosh Chowdhury, Prashanth Kallur Ramaswamy, Hannah R Marlowe, Carina
  Suzana Negreanu, Kitty Boxall and Diana Mincu, where the entry had Cyrus Cha,
  Andrea Chowdhury, Ananya Ramaswamy, Anna Marlowe, Cristian Negreanu, Nick
  Boxall and Dan Mincu. The key also named a first author (Zhou) and a year
  (2024) that belong to neither the paper nor its venue. The entry is now
  `enguehard2025lemaj` with the printed author list.
- `qiang2025redefining` was recorded as a COLING 2025 proceedings paper. It is
  not in the ACL Anthology; the arXiv listing for 2502.08281 carries no venue
  comment, and the submission postdates the conference. It is now an arXiv
  preprint entry.
- The citation of LeMAJ in Section 6.4 sat against the phrase "a fine-tuned
  112M-parameter encoder", which is JUDGEBERT and not LeMAJ. It now attaches to
  the prompted-judge clause it actually supports.
- Two claims overstated their sources. Zhang and Coltekin write that
  high-opposition items are "more difficult to predict and are often
  underestimated", not that they are the hardest and are systematically
  underestimated; JudgeSense reports that "model scale does not predict
  consistency" and says nothing about the largest and newest models being the
  least consistent. Both sentences in `main.tex` were rewritten to the source
  wording.

Missing but uncontroversial fields were also filled in from the verified
records: pages 7219--7234 for `sai2021perturbation`, 163--175 for
`anschutz2023negation`, 4263--4294 for `trienes2024infolossqa`, article number
1223924 for `beauchemin2023meaningbert`, and the SSRN working paper number
4696936 for `hagan2023good`.

Everything else matched. In particular the F1 of 0.67 attributed to Hariri and
Ho in the Introduction is their own abstract's figure, the 17 to 33 percent
hallucination range attributed to Magesh et al. is theirs, the 14.9 against
14.1 percent flip rates attributed to Yang et al. are theirs on MedQA, and the
page range 296--304 for `garimella2022text` is correct despite a search summary
that reported 224--234.

The paper's four claims that need external support are:

- **C1.** Sanity checks that move surface form and meaning together cannot
  establish that a metric measures meaning. (Requirement 0, the paper's thesis.)
- **C2.** Correlation with human ratings is not evidence of construct validity.
- **C3.** Similarity metrics are insensitive to meaning-changing edits;
  entailment is the better starting point.
- **C4.** A prompted judge's behaviour is set by its rubric, not only by its
  model, and rater disagreement is signal rather than noise.

Each entry is tagged with the claims it supports and with where in the paper it
should go.

---

## 1. Validity of benchmarks and metrics

### Bean et al. (2025), *Measuring what Matters: Construct Validity in Large Language Model Benchmarks*
arXiv:2511.04703, submitted 3 November 2025. NeurIPS 2025 Datasets and
Benchmarks Track. 42 authors (Bean, Kearns, Romanou, Hafner, Mayne, Batzner,
Foroutan, ... Rocher, Mahdi), reporting the work of 29 expert reviewers.

Systematic review of 445 LLM benchmarks drawn from leading NLP and ML
conferences. Finds recurring patterns in the measured phenomena, the tasks and
the scoring metrics that undermine the validity of the claims those benchmarks
support, and issues eight recommendations for benchmark construction. Reported
in coverage of the review: only 16% of the surveyed benchmarks used uncertainty
estimates or statistical tests, and 27% relied on convenience sampling.

*Supports C1, C2.* This is the strongest single anchor for the paper's central
move. Our argument is one instance of the general failure the review documents:
a scoring metric whose reported checks do not represent the phenomenon claimed.
Belongs in the Related Work meta-evaluation paragraph and in Section 3's opening.

### Norman, Rivera and Hughes (2026), *Reliability without Validity: A Systematic, Large-Scale Evaluation of LLM-as-a-Judge Models Across Agreement, Consistency, and Bias*
arXiv:2606.19544, submitted 17 June 2026.

21 judges from nine providers over MT-Bench, JudgeBench and RewardBench; three
protocols (agreement, consistency, bias audit), 118 runs, ~541,000 judgments.
Four findings: exact-match agreement systematically overstates discriminative
ability and deflates by 33--41 points when corrected to Cohen's kappa; judge
rankings shift by up to 14 positions across benchmarks; test--retest reliability
above 0.95 coexists with severe position bias, a "consistency--bias paradox";
verbosity bias is small under a single pairwise rubric. Distilled into a
"Minimum Viable Validation Protocol".

*Supports C1, C2, C4.* The title states our thesis for the judge case: a scorer
can be perfectly reliable on the checks it is given and still not be valid. The
consistency--bias paradox is the exact analogue of our judge scoring 10.00 on
identical pairs and 1.06 on unrelated ones, a flawless result on both
conventional checks, while awarding a clause whose legal force has been
reversed 6.69 out of ten (margin 0.370).

### Rao and Callison-Burch (2026), *Agreement Metrics for LLM-as-Judge Evaluation: What to Report and Why*
arXiv:2606.00093, v1 25 May 2026, v2 31 July 2026.

Argues that the reported agreement of a rubric-based judge with human labels is
fixed by protocol choices made before any statistic is computed: the judgment
scale, retained cases, handling of abstentions, and pooling across items and
criteria. Shows that for non-degenerate binary verdicts Pearson r, Spearman rho,
Kendall tau-b, phi and MCC are literally the same statistic; that Cohen's kappa
differs only by a marginal-mismatch factor; and that on a rubric benchmark,
protocol choice alone moves accuracy from 0.551 to 0.899 and carries kappa
across zero without changing one verdict. Ends in a reporting checklist.

*Supports C1, C2, C4.* Directly reinforces Requirement 2 (identical supervision)
and our Krippendorff alpha discussion: the choice of coefficient is a
substantive claim, not a formatting detail. Our nominal-versus-ordinal alpha
point is a special case of theirs.

### Pita (2026), *Correct codes for the wrong reasons? Validating LLMs as measurement instruments for theoretical constructs*
arXiv:2606.28574, submitted 26 June 2026. Single author.

When an LLM codes a construct as a human annotator would, that agreement
establishes reliability but leaves construct validity untouched: the instrument
may be theory-naive and reach the code through a correlate. Proposes "grain
calibration", decomposing a construct into clause-level components, testing each
against the text with extractive evidence, and combining them under an explicit
theory-derived rule so that the process, not just the output, is evidence.

*Supports C1, C2, C4.* Almost a restatement of our finding that a ridge
regression over word counts hits the human ceiling. Good for the Section 6.3
paragraph on what the annotation rewards, and for the judge section.

### Gur-Arieh, Marasović and Geva (2026), *Faithfulness Metrics Don't Measure Faithfulness: A Meta-Evaluation with Ground Truth*
arXiv:2605.25052, submitted 24 May 2026.

Constructs tasks whose outputs reveal which intermediate computations produced
them, so that ground-truth faithfulness labels can be assigned automatically.
Releases BonaFide (3,066 labelled chains of thought, 13 tasks, 10 models) and
finds most published faithfulness metrics perform near chance, with strong
prediction biases; the best reaches 0.70 AUROC at the chain level.

*Supports C1, C2.* Different construct (CoT faithfulness, not meaning
preservation) so it should be cited as a parallel, not as a neighbour: the
recurring pattern is metrics validated against proxies rather than against the
property claimed. Use one clause, not a sentence.

### Deviyani and Diaz (2025), *Contextual Metric Meta-Evaluation by Measuring Local Metric Accuracy*
arXiv:2503.19828, 25 March 2025. NAACL 2025 Findings.

Existing meta-evaluation makes global statements about metric quality over
arbitrary system outputs, but metrics are applied in constrained contexts.
Introduces local metric accuracy and shows across translation, ASR and ranking
that both the absolute and relative effectiveness of metrics shift with the
evaluation context.

*Supports C2.* Supports the observation that our diagnostic and our correlation
evaluation run on different text and answer different questions, and that a
single global ranking of metrics is the wrong object.

### Pacchiardi, Tesic, Cheke and Hernández-Orallo (2024), *Leaving the barn door open for Clever Hans: Simple features predict LLM benchmark answers*
arXiv:2410.11672, 15 October 2024.

Internal validity of benchmarks: simple n-gram classifiers, lacking the
capability under test, score highly on several modern multiple-choice
benchmarks, and there is evidence LLMs exploit the same superficial patterns.

*Supports C2.* This is the general form of Requirement 3. Our surface-feature
ridge regression is the Clever Hans control for a metric benchmark rather than
for a capability benchmark. Strong cite for Section 3.3.

---

## 2. Metric sensitivity to meaning-changing edits

### Mujahid, Wright and Augenstein (2025/2026), *Stress Testing Factual Consistency Metrics for Long-Document Summarization*
arXiv:2511.07689, v1 10 Nov 2025, v2 29 April 2026. ACL 2026.

Evaluates six reference-free factuality metrics under seven
factuality-*preserving* perturbations (paraphrasing, simplification, synonym
replacement, logically equivalent negations, vocabulary reduction, compression,
source insertion) across science-fiction, legal and scientific benchmarks.
Existing short-form metrics produce inconsistent scores for semantically
equivalent summaries and degrade on information-dense claims. Recommends
training on meaning-preserving variations.

*Supports C3.* The mirror image of LegalEdit: they hold meaning fixed and move
form, we hold form fixed and move meaning. Citing both makes the dissociation
argument symmetric and is a much better companion to MENLI than another 2021
challenge set. Their legal-domain inclusion is a bonus.

### Dumpala, Jaiswal, Sastry, Milios, Oore and Sajjad (2024), *VISLA Benchmark: Evaluating Embedding Sensitivity to Semantic and Lexical Alterations*
arXiv:2404.16365, 25 April 2024.

A three-way semantic (in)equivalence task over sentence triplets, evaluated on
34 vision-language and 20 unimodal language models. Finds surprising difficulty
distinguishing lexical from semantic variation, and that encoded semantics are
highly sensitive to lexical information.

*Supports C3.* Independent evidence, outside NLG metrics, that embedding
representations confound lexical and semantic change. Exactly the confound
LegalEdit dissociates. Good companion cite to `cripwell2024document` and
`kew2023bless`.

### Yang, Levy, Goldberg and Wallace (2026), *Compared to What? Baselines and Metrics for Counterfactual Prompting*
arXiv:2605.01048, v1 1 May 2026, v2 20 Aug 2026. COLM 2026.

Argues that a counterfactual edit is a compound treatment bundling the variable
of interest with incidental surface variation, so an observed effect cannot be
attributed to the target factor without a meaning-preserving paraphrase
baseline. On MedQA, a 14.9% flip rate under a gender intervention is
statistically indistinguishable from the 14.1% induced by paraphrase alone;
re-analysing MedPerturb, only 5 of 120 reported effects survive. Per-sample
metrics are far more powerful than aggregate ones.

*Supports C3, and is a caveat against us.* This is the sharpest methodological
objection available to LegalEdit and we should raise it ourselves in
Limitations. Our defence is real and worth stating: the margin is normalised
against each metric's own identical-to-unrelated range, so the identical probe
*is* a null-edit baseline, and our fluency filter bounds incidental variation.
But we do not run a paraphrase-control arm, and we should say so.

### You and Guo (2025), *PlainQAFact: Retrieval-augmented Factual Consistency Evaluation Metric for Biomedical Plain Language Summarization*
arXiv:2503.08890, v1 11 March 2025, v4 19 March 2026. Published as *Journal of
Biomedical Informatics* 178:105019 (2026); cite as 2026.

Entailment- and QA-based factual consistency metrics fail on plain language
summarization because of "elaborative explanation": simplification for lay
readers adds definitions, background and examples absent from the source.
Introduces PlainFact (human-annotated) and a sentence-type-aware,
retrieval-augmented QA metric that outperforms existing metrics.

*Supports C3.* The nearest neighbour to our problem in another high-stakes
domain: meaning preservation under simplification for lay readers, where
off-the-shelf entailment breaks for a domain-specific reason. Also the honest
counterweight to our "entailment is the better starting point" conclusion.

---

## 3. LLM judges and rubrics

### Roy et al. (2026), *PReMISE: Policy Rubrics as Measurement Specifications for LLM Judges*
arXiv:2605.30803, 29 May 2026. Roy, Pujari, Kumarage, Peris, Gupta, Rumshisky,
Natarajan, Saligrama.

Judge scores depend strongly on the rubric conditioning them; a vague rubric
rewards polished answers that invent facts. Discovers policy-level rubric sets
from preference data and audits them for structural adequacy, reliability,
preference fit and adversarial robustness; no existing rubric source achieves
all three of reliability, preference predictiveness and adversarial robustness
at once. Two repairs: preference-rank selection raises judge accuracy 65.0% ->
68.6%; reliability-constrained refinement cuts exploit vulnerability 46.4% ->
36.0%.

*Supports C4.* The direct external support for our strongest judge result, that
the FrJudge rubric and not the model is what holds the margin down to 0.417,
against 0.687 under a bare instruction. Their
framing, rubric as measurement specification, is the language our Section 6.4
wants.

### Bellibatlu, Raff and Zhang (2026), *JudgeSense: A Benchmark for Prompt Sensitivity in LLM-as-a-Judge Systems*
arXiv:2604.23478, v1 26 April 2026, v2 7 May 2026.

Hand-validated prompt-paraphrase pairs over factuality, coherence, relevance and
preference, testing verdict stability under semantically equivalent rephrasing.
Coherence best distinguishes judge behaviour; factuality judgments are stable;
pairwise tasks show consistent position bias; model scale is not a reliable
proxy for consistency and the largest, newest models are not the most
consistent.

*Supports C4.* Backs our prompt-sensitivity experiment (margin 0.417 -> 0.687
across three wordings, a range of 0.270) and our finding that a reasoning model does not help.
Their "scale is not a proxy for consistency" is our deepseek-reasoner control
stated generally.

### Xu, Lindholm, Raina, Palmer Olsen and Hershcovich (2026), *LP-Eval: Rubric and Dataset for Measuring the Quality of Legal Proposition Generation*
arXiv:2605.19815, 19 May 2026.

A three-step evaluation rubric for legal propositions from CJEU decisions,
co-designed with legal experts, decomposing quality into formal validity and
substantive dimensions; releases two experts' annotations over 100 LLM-generated
propositions. Rubric-guided LLM judgments align more closely with expert
assessment than direct overall scoring, but **remain insensitive to
finer-grained distinctions captured by human experts**.

*Supports C4.* The closest published finding to ours in the legal domain, from a
different jurisdiction, a different task and a different rubric: a rubric buys
alignment and costs sensitivity. Their insensitivity is qualitative; our margin
statistic quantifies it. This is the single best new citation in the set.

### Hu et al. (2026), *Evaluation of Large Language Models in Legal Applications: Challenges, Methods, and Future Directions*
arXiv:2601.15267, 21 January 2026. 18 authors (Hu, Liu, Wang, Li, Wu, Li, Xu,
Huo, Su, Zheng, Zheng, Ai, Liu, Bian, Liu, Clarke, Shen, Kao).

Survey of legal LLM evaluation organised around outcome correctness, reasoning
reliability and trustworthiness; reviews benchmarks by task design, dataset and
metric, and identifies where current approaches fall short.

*Supports C2.* Useful one-line framing cite for the legal-evaluation paragraph,
and it replaces two older single-purpose legal-resource citations.

---

## 4. Disagreement as signal

### Xu and Jurgens (2026), *Beyond Consensus: Perspectivist Modeling and Evaluation of Annotator Disagreement in NLP*
arXiv:2601.09065, v1 14 January 2026, v2 17 January 2026.

Survey. A domain-agnostic taxonomy of disagreement sources across data, task and
annotator factors; a common framework for modelling approaches defined by
prediction target and pooling structure; documents the shift from consensus
learning to explicitly modelling disagreement and to capturing structured
relationships among annotators. Notes most fairness evaluations remain
descriptive rather than normative.

*Supports C4.* A single current survey that subsumes the four 2019--2022
citations currently carrying our disagreement paragraph. Its
prediction-target/pooling framework is also the right vocabulary for our
Appendix training variants (scalar, distributional, quantile heads).

### Zhang and Çöltekin (2026), *Quantifying and Predicting Disagreement in Graded Human Ratings*
arXiv:2605.01168, 1 May 2026. 5th Workshop on Perspectivist Approaches to NLP,
LREC.

Investigates variation patterns in *graded* human ratings, asks whether the
degree of disagreement is predictable from textual features, and proposes an
Opposition Index quantifying perspective opposition on an item. Finds a moderate
correlation between predicted and observed variance, and that high-opposition
items are harder to predict and are systematically underestimated.

*Supports C4.* The graded-rating case is ours exactly: FrJudge is a ten-point
ordinal scale with a bimodal distribution and a median within-item range of 5
points. Their Opposition Index is the natural formal counterpart to our
strict/lenient annotator split.

### Bonagiri et al. (2026), *STABLEVAL: Disagreement-Aware and Stable Evaluation of AI Systems*
arXiv:2605.02122, v1 4 May 2026, v2 1 June 2026.

Majority-vote aggregation discards annotator reliability and item ambiguity,
giving system rankings that are unstable across annotator subsets. Models latent
item correctness and annotator-specific confusion to produce posterior expected
item credit and calibrated system scores; formalises ranking stability as a
first-class objective; shows majority vote degrades under annotator
heterogeneity while STABLEVAL stays stable.

*Supports C4.* Directly relevant to our point that the mean of five ratings is a
quantity no annotator asserted. Their "aggregation distorts annotator behaviour"
result is the mechanism behind our two-bound ceiling.

---

## 5. Legal simplification and legal deployment stakes

### Hariri and Ho (2025), *AI for Statutory Simplification: A Comprehensive State Legal Corpus and Labor Benchmark*
arXiv:2508.19365, 26 August 2025. ICAIL 2025.

One U.S. state has claimed to eliminate a third of its state code using AI, yet
systematic evaluation of accuracy and risk is absent. Introduces LaborBench,
built from a U.S. Department of Labor dataset in which teams of lawyers compile
unemployment-insurance differences across 50 states over 101 dimensions in a
six-month process, and StateCodes, an 8.7 GB state statute and regulation
corpus. Benchmarks retrieval and frontier LLMs: useful for preliminary research,
but overall accuracy (F1 0.67) is far below the claims made for end-to-end
regulatory simplification.

*Supports C1, and motivates the whole paper.* This is a better opening stake
than the Moffatt anecdote alone, because it is a documented deployment at
statutory scale with a measured error rate rather than a single lawsuit. It
should sit alongside Moffatt in the Introduction, not replace it: Moffatt
establishes legal cognisability, Hariri and Ho establish scale.

### Magesh, Surani, Dahl, Suzgun, Manning and Ho (2024), *Hallucination-Free? Assessing the Reliability of Leading AI Legal Research Tools*
arXiv:2405.20362, 30 May 2024. *Journal of Empirical Legal Studies*
22(2):216--242 (2025), DOI 10.1111/jels.12413.

First preregistered empirical evaluation of proprietary RAG-based legal research
tools. Lexis+ AI, Westlaw AI-Assisted Research and Ask Practical Law AI
hallucinate between 17% and 33% of the time, against vendor claims of
"eliminating" or "avoiding" hallucination.

*Supports C1.* The vendor-claim structure is our structure: a sanity check
("hallucination-free") advertised as validity. Good Introduction cite.

### Dahl, Magesh, Suzgun and Ho (2024), *Large Legal Fictions: Profiling Legal Hallucinations in Large Language Models*
arXiv:2401.01301. Journal of Legal Analysis 16(1):64--93, 2024.

Legal hallucination rates of 58% (ChatGPT-4) to 88% (Llama 2) on verifiable
questions about random federal cases; models fail to correct users' incorrect
legal assumptions and cannot reliably predict their own hallucinations.

*Supports C1.* Secondary to Magesh et al.; cite one of the two, not both, unless
the contra-factual finding is used.

### Korobeynikova, Battisti, Fischer and Gao (2025/2026), *DETECT: Determining Ease and Textual Clarity of German Text Simplifications*
arXiv:2510.22212, v1 25 October 2025, v2 6 March 2026; cite as 2025 to match
the identifier.

General-purpose metrics (SARI, BLEU, BERTScore) insufficiently capture German
simplification quality; LENS-style trained metrics existed only for English.
DETECT evaluates simplicity, meaning preservation and fluency, trained entirely
on synthetic LLM-generated quality scores, and validated against the largest
German human evaluation set for the task.

*Supports C3.* Evidence that the "trained metric for a non-English language"
pattern is now general, not a one-off, and it updates our claim about the state
of non-English simplification metrics. Its synthetic-supervision design is also
a live alternative to the FrJudge-style human corpus.

### Qiang, Huang, Zhu, Yuan, Zhang and Yu (2025), *Redefining Simplicity: Benchmarking Large Language Models from Lexical to Document Simplification*
arXiv:2502.08281, 12 February 2025. COLING 2025.

First comprehensive comparison of LLMs against non-LLM methods across lexical,
syntactic, sentence and document simplification, with automatic and human
evaluation. LLMs win on all four tasks and often exceed the quality of existing
human-annotated references.

*Supports C3.* Motivates why metric quality now matters more than system
quality: if outputs exceed the references, reference-based metrics are the
binding constraint. Replaces older generic simplification citations.

### Dahan, Toledano, Gafter, Pardo, Tsur, Zahavi and Sulem (2026), *Translate or Simplify First: An Analysis of Cross-lingual Text Simplification in English and French*
arXiv:2604.23844, 26 April 2026.

Five prompting strategies for cross-lingual simplification between English and
French, over five corpora (Wikipedia and medical) and seven LLMs, evaluated with
automatic metrics, linguistic feature analysis and human judgment of simplicity
and meaning preservation. Direct prompting gives the highest BLEU;
translate-then-simplify gives the highest simplicity.

*Supports C3.* Current French simplification work, sharing an author with
`sulem2018bleu`. Confirms that French simplification corpora remain
non-legal (Wikipedia and medical), which is a claim our Related Work makes and
should now support with a 2026 citation rather than 2018--2020 ones.

### Ovcharov (2026), *Multi-Legal-Bench: Evaluating LLMs on Legal Reasoning Across Jurisdictions, Languages, and Legal Traditions*
arXiv:2605.29738, v1 28 May 2026, v2 7 August 2026.

Cross-jurisdictional benchmark over six countries including France, four
language families and 165 million court decisions; five tasks mapped to national
court-registry metadata. No model dominates any language; rankings shift with
task and jurisdiction.

*Supports C2, weakly.* Useful only as evidence that French legal NLP resources
now exist but are about reasoning and classification, not simplification or
meaning preservation, which keeps our "only annotated corpus for this construct"
claim alive. One clause at most.

---

## What to drop or demote

Recommendations, given the instruction to weight the paper toward current work.

| Current citation | Action | Reason |
| --- | --- | --- |
| `pavlick2019inherent`, `plank2022problem`, `uma2021learning`, `peterson2019human` (four-cite block, Related Work) | Reduce to two and add Xu and Jurgens 2026 | One 2026 survey covers the block; keep `uma2021learning` and `peterson2019human` because the Appendix training variants use their soft-label method by name |
| `rebuffel2021questeval`, `fabbri2022qafacteval`, `laban2022summac`, `zha2023alignscore` (four English-adequacy entailment cites) | Reduce to two, add Mujahid et al. 2026 and You and Guo 2025 | Four 2021--2023 citations for one uncontested background claim |
| `sai2021perturbation` | Keep, but pair with Mujahid et al. 2026 | Still the origin of the perturbation-checklist method |
| `hendrycks2021cuad`, `garneau2022criminal` | Keep one, add Hu et al. 2026 | The survey covers the "legal NLP is mostly English and not about simplification" claim more currently |
| `hayes2007answering`, `laban2020summary` | Delete from refs.bib | Cited zero times |
| `papineni2002bleu`, `xu-etal-2016-optimizing`, `krippendorff2004content`, `artstein2008inter`, `koenker1978regression`, `zadrozny2002transforming` | Keep | Definitional citations for named methods; age is not a defect here |
| `chen2023menli` | Keep, prominent | Still the closest prior work, and the paper already has a dedicated paragraph |

## Net effect on the paper

- Introduction gains a measured deployment stake (Hariri and Ho; Magesh et al.)
  in place of relying on a single 2024 tribunal decision.
- Related Work gains a validity paragraph (Bean et al.; Norman et al.;
  Pacchiardi et al.) that states the general form of our argument, so our
  contribution reads as an instance of a recognised problem rather than as an
  isolated complaint.
- Requirement 3 gains its general precedent (Pacchiardi et al.).
- Section 5 gains a symmetric companion to MENLI (Mujahid et al., who hold
  meaning fixed and move form) and a representation-level cite (VISLA).
- The judge section gains three 2026 citations that independently establish its
  three findings: rubric as the binding constraint (PReMISE, LP-Eval), prompt
  sensitivity above model capability (JudgeSense), and agreement statistics as
  protocol choices (Rao and Callison-Burch).
- Limitations gains a real objection we had not stated (Yang et al.: no
  paraphrase-control arm).

## Fourth pass (2026-09-02): independent existence check of all 61 entries

This pass asked one question only: does each reference in `refs.bib` correspond
to a real record, with the title, authors, year and venue the entry claims? It
was run independently of the passes above, against three indexes.

Method. Every entry was matched by title against DBLP and CrossRef, and the
sixteen entries carrying an arXiv identifier were also fetched by identifier
from the arXiv Atom API, which is the strict check: a fabricated identifier
returns nothing, and a real identifier attached to the wrong paper returns a
title that does not match. All sixteen resolved, and the returned titles and
author lists matched the bibliography. The full author list of each arXiv entry
was diffed position by position against the API record; the four apparent
mismatches were all artefacts of the comparison (`de la Clergerie` against
`Clergerie`, `Marasovi{\'c}`, `Hern{\'a}ndez-Orallo`, `Palmer Olsen`), not
errors in the entry. `bean2025construct` was re-counted against arXiv 2511.04703
and still carries 42 authors in the printed order.

Entries whose title match scored low were checked by hand, and each is a
limitation of the matcher rather than a bad reference: `krippendorff2004content`
is a book that CrossRef indexes by chapter, `hagan2023good` is an SSRN working
paper (abstract 4696936, presented at the JURIX 2023 AI and Access to Justice
workshop), `koenker1978regression` is Econometrica 46(1):33--50 under DOI
10.2307/1913643, `card2020power` is EMNLP 2020 pages 9263--9274 under DOI
10.18653/v1/2020.emnlp-main.745, and `antoun2024camembert2` was confirmed by its
arXiv identifier after CrossRef returned the original CamemBERT paper instead.

The recent venue attributions that a title search cannot settle were each
confirmed at the venue: JUDGEBERT at EMNLP 2025 main, pages 92--118; LeMAJ at
NLLP 2025, pages 318--337, with all eleven authors as printed; Bean et al. at
the NeurIPS 2025 Datasets and Benchmarks track; Yang et al. as a COLM 2026
conference paper, which the arXiv comment field states explicitly; Zhang and
Çöltekin at NLPerspectives @ LREC 2026; Mujahid et al. at ACL 2026,
2026.acl-long.1472; Deviyani and Diaz in Findings of NAACL 2025,
2025.findings-naacl.276; Hariri and Ho at ICAIL 2025; Magesh et al. in the
Journal of Empirical Legal Studies 22(2):216--242; You and Guo in the Journal of
Biomedical Informatics 178:105019.

No fabricated reference was found, and no correction to an existing field was
needed. One entry is now out of date rather than wrong: `korobeynikova2025detect`
is recorded as the October 2025 arXiv preprint, and the paper has since appeared
as EACL 2026 long papers 2852--2882 (`2026.eacl-long.131`). Updating it would
move the key and year to 2026.

Coverage was also checked in both directions: all 61 keys cited in `main.tex`
exist in `refs.bib`, and all 61 entries in `refs.bib` are cited.
