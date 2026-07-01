\# Methodology



This project treats preference pair quality as a pair-level diagnostic problem.



Instead of directly training a new reward model, the goal is to analyze whether a preference pair itself is clear, ambiguous, genuinely disagreed upon, or potentially low-quality.



\## 1. Problem Setting



A preference pair usually contains:



\- a prompt

\- response A

\- response B

\- multiple human annotations



In many preference datasets, the pair is treated as a simple A/B comparison. However, human annotators may disagree for different reasons. Some disagreement reflects genuine subjective preference, while some may come from ambiguity, formatting differences, safety framing, factuality issues, or response length imbalance.



Therefore, this project focuses on diagnosing the quality of the preference pair itself.



\## 2. Ambiguity Score



The ambiguity score measures whether human annotators have difficulty forming a stable preference.



Possible features include:



\- annotation entropy

\- majority strength

\- tie rate

\- slight preference rate

\- clear preference rate



A high ambiguity score suggests that the preference pair may not have a clear and stable human preference.



\## 3. Confound Score



The confound score measures whether the comparison may be affected by non-core quality factors.



Possible features include:



\- response length imbalance

\- format mismatch

\- refusal mismatch

\- caution mismatch

\- response similarity

\- verbosity difference

\- structure difference



A high confound score suggests that annotators may be influenced by factors such as length, format, refusal behavior, or safety framing rather than only response quality.



\## 4. Pair-Level Diagnostic Types



The planned framework aims to distinguish four types of preference pairs:



| Type | Ambiguity | Confound | Interpretation |

|---|---:|---:|---|

| Clear preference pair | Low | Low | The pair has a stable and relatively clean preference signal. |

| Genuine disagreement pair | High | Low | Annotators disagree, but the disagreement may reflect valid subjective differences. |

| Confounded but stable pair | Low | High | Annotators agree, but the pair may be affected by length, format, or other confounds. |

| Ambiguous / low-quality pair | High | High | The pair may require manual review before being used for training or evaluation. |



\## 5. Next Step



The next step is to implement a no-training diagnostic score based on annotation distribution and text-level features.



The goal is not to predict which response is better. Instead, the goal is to identify preference pairs that are ambiguous, confounded, or potentially low-quality.

