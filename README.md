## Proposed Question
(LLM Creativity evaluation) Does naming unrelated words predict creativity in LLMs? *Does this correlate with their reasoning skills?

https://www.pnas.org/doi/epub/10.1073/pnas.2022340118

___

## Problem Outline
We proceed to investigate whether we can determine the ‘creativity’ of an LLM. We adapt the metric proposed in [1] to apply to LLMS. 

## Methodology 

We consider two readily available models - **GPT4-o (GPT)** and **Gemini 1.5 Flash (Gemini)** - both of which have already been benchmarked on many of the popular LLM-reasoning datasets, e.g. MMLU, MATH, and GPQA. 

To measuring LLM creativity, we directly adapt the Divergent Association Task (DAT) presented in [1], requiring the models to generate 10 unrelated words, and then computing the semantic distance between 7 of the 10 generated words, consistent with [1]'s methodology.  For a discussion for how the DAT provides a strong measure of creativity, and how it has positive correlation with other established measures of creativity, see [1].

We implement a custom version of Global Vectors for Word Representation (GloVe), using the [common crawl (840B tokens, 2.2M vocab, cased, 300d vectors) pre-trained weights](https://nlp.stanford.edu/projects/glove/) which we use to compute the semantic distance between generated random words. GloVe provides a vectorised representation of our generated words, which we use to compute the cosine distance. For a given set of unrelated words, the final score is given by the average cosine distance. 

[1] provides us with experimental results which we can use to determine model performance - it suggests that under 50 is poor, 75-80 is average, and 95+ is a very high score. 

First, we verified our GloVe implementation by generating the same score matrices presented in [1], and ensuring our scores matched those in the paper. 

We then prompted each model three times - the first two prompts provided are the same “Generate a list of 10 unrelated words”, and the final prompt is “Generate a list of 10 unrelated words - maximise the semantic distance between them”. The score matrices are calculated along with the average scores. We compare the performance of the two models against each other, and we also compared their DAT scores with common LLM 'reasoning' benchmarks to see if there was any correlation between creativity (as outlined in [1]) and reasoning.
## Results
The results for this experiment can be found in this [Google sheet.](https://docs.google.com/spreadsheets/d/1Hv4O9wbxoC4vxe1XyVB7H9pr0bSTqatjMs9aAgbnMiw/edit?usp=sharing)

Both models exhibit above-average performance relative to the scores set out in [1], with GPT scoring an average of 79.93 and Gemini 83.46. These scores place both models in the ‘average to good’ performance band for the DAT task. We also see that additional prompting aimed at maximising semantic distance between generated words had little effect on creativity scores, indicating that the models’ performance in divergent thinking tasks is not easily influenced by naive prompt modifications.

Interestingly, GPT consistently outperformed Gemini in reasoning benchmarks, such as MMLU, MATH, and GPQA, suggesting that it is better suited for structured knowledge tasks. Indeed, the inverse correlation observed between performance on reasoning and creativity tasks suggests that these two cognitive functions may be independent of each other, if not in opposition. 

While GPT is optimised for high-precision text-based tasks, Gemini appears to be build with multi-modality and extended context windows as its main focus. It may be that these multi-modal design choices contribute to its ability to perform well at creative tasks. These initial findings may hint at a trade-off between creative and reasoning tasks in LLMs; reasoning at the cost of creative flexibility. 

Further experiments could explore more in-distribution tasks, for example, proposing various unique elegant solutions to mathematical problems. Typically, the difference between a brute-force solution and an elegant solution is the creativity with which the problem is approached. This may provide a better measure for LLMs designed with reasoning as their primary focus.

## Conclusion
Our initial investigation suggests that performance on creative tasks such as DAT does not necessarily correlate with performance on reasoning tasks, as measured by benchmarks like MMLU. Indeed, these results suggest the opposite may be true — models optimized for structured-knowledge and factual reasoning tasks may exhibit reduced performance on creative tasks, likely due to their architectural focus on logical consistency and structured outputs.

The trend in SoA LLMs development is towards multi-modality, this additional capacity appears to benefit models in creative tasks such as DAT. Multi-modal models like Gemini 1.5 Flash, designed for handling diverse input types, excel in creativity over purely text-based models. To achieve a fair comparison, creative tasks should perhaps be measured using text-only models, factoring out any potential contribution from multi-modal training. 

Both models demonstrated strong performance in DAT; however, the inverse relationship** between creativity and reasoning abilities may be due to differences in architectural optimization. GPT-4o, with its strong reasoning capabilities, excels in structured knowledge tasks, while Gemini 1.5 Flash’s broader input context and multi-modal strengths favor creative output, suggesting a potential reasoning-creativity trade-off.

To measure creativity more accurately, future experiments could integrate multi-modal elements of LLMs, such as combining tasks like DAT with creative visual assessments (creative image generation). Additionally, a more nuanced approach to creativity could involve evaluating a model’s ability to produce innovative solutions to structured problems, like mathematics, where elegant problem-solving reflects deeper creative capabilities. 

This would align the creativity measurement more closely with the model’s training distribution, providing a more reliable measure of creative reasoning within the framework of structured knowledge tasks.

## References
	1.	D. L. Kenett, J. Anaki, and M. Faust, “Investigating the structure of semantic networks in low and high creative persons,” Proceedings of the National Academy of Sciences of the United States of America, vol. 118, no. 8, e2022340118, 2021. DOI: https://doi.org/10.1073/pnas.2022340118.
