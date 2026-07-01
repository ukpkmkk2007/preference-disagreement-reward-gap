# preference-pair-quality-diagnostics

\# Preference Pair Quality Diagnostics



This project studies preference pair quality in human feedback datasets.



It starts from a small-scale baseline reproduction of preference disagreement analysis and moves toward a no-training diagnostic framework for identifying clear, ambiguous, genuinely disagreed-upon, and potentially low-quality preference pairs.



\## Motivation



Preference pairs are often treated as clean A/B comparison samples. However, human annotators may disagree for many reasons, including:



\- response length and verbosity

\- format differences

\- safety or refusal behavior

\- factuality and instruction following

\- task ambiguity

\- different preferences for detail, concision, or tone



Therefore, disagreement should not be treated simply as noise. Some disagreement may reflect genuine subjective preference, while some may indicate that the preference pair itself is ambiguous or confounded.



The goal of this project is to analyze preference pair quality at the data level.



\## Current Work



The current version contains a small-scale reproduction and diagnostic analysis based on MultiPref-style preference pairs.



The implemented components include:



\- parsing human preference annotations

\- reproducing disagreement definitions

\- constructing pair-level text features

\- running a small-scale single-value reward model baseline

\- computing preference accuracy

\- computing Diverging ID AUROC

\- conducting qualitative case studies on high-confidence errors and diverging examples

\- preparing a no-training preference pair quality diagnostic framework



\## Baseline Evaluation



The baseline uses a single-value reward model to score response A and response B separately.



If the score of response A is higher, the model predicts A.  

If the score of response B is higher, the model predicts B.



The model prediction is then compared with the human majority preference.



For disagreement detection, the absolute reward gap is used:



```text

reward\_gap = abs(score\_a - score\_b)

