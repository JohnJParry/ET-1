(LLM Creativity evaluation) Does naming unrelated words predict creativity in LLMs? *Does this correlate with their reasoning skills?

https://www.pnas.org/doi/epub/10.1073/pnas.2022340118
___
We proceed to investigate whether we can determine the ‘‘creativity’ of a LLM. We adapt the metric proposed in [Naming unrelated words predicts creativity](https://www.pnas.org/doi/epub/10.1073/pnas.2022340118) to apply to LLMS. 

Related code can be found at: **blah.github**@@@@@@@@@@@

## Methodology 

We consider two readily available models - **GPT4-o (GPT)** and **Gemini 1.5 Flash (Gemini)** - both of which have already been benchmarked on many of the popular LLM-reasoning datasets, e.g. MMLU, MATH, and GPQA. 

For measuring LLM creativity, we directly adapt the Divergent Association Task (DAT) presented in [1], requiring the models to generate 10 unrelated words, and then computing the semantic distance between 7 of the 10 generated words. For a further discussion for how the DAT provides a strong measure of creativity, and how it has strong positive correlation with other established measures of creativity, see [1].

We implement a custom version of Global Vectors for Word Representation (GloVe), using the [common crawl (840B tokens, 2.2M vocab, cased, 300d vectors) pre-trained weights](https://nlp.stanford.edu/projects/glove/) which we use to compute the semantic distance between generated random words. GloVe provides a vectorised representation of our generated words, which we use to compute the cosine distance. For a given set of unrelated words, the final score is given by the average cosine distance. 

The original paper provides us with experimental performance bounds, which we can use to determine model performance - it suggests that under 50 is poor, 75-80 is average, and 95+ is a very high score. 

We first re-calculated the presented unrelated word lists in the paper, to verify our GloVe implementations agree - indeed they do. 

We prompt each model three times - the first two prompts provided are the same “Generate a list of 10 unrelated words”, and the final prompt is “Generate a list of 10 unrelated words - maximise the semantic distance between them”. The score matrices are calculated along with the average scores. We compare the performance of the two models, along with comparing this to each models performance on common LLM ‘reasoning’ benchmarks. 
## Results

Both models exhibit above average performance relative to the scores determined by [1], GPT scoring on average 79.93 and Gemini scoring on average 83.46, putting both models in the average - good band. 

The additional prompting to maximise semantic distance between generated words appears to have little influence on the scores. 

What’s interesting to see is that, on each of the common reasoning metrics between the models (MMLU, MATH, GPQA), GPT scored notably higher. However, we see that performance on ‘‘reasoning’ tasks is not necessarily indicative (or indeed correlated) to model ‘creativity’. 
### Creativity vs Reasoning

The creativity metric proposed explicitly measures divergent thinking - in the case of the task proposed to the LLM, this involves generating unrelated words given one input prompt. Reasoning measures discourage models exhibiting this behaviour, and promotes convergent thinking - reaching a solution to the problem proposed (especially seen in datasets such as MMLU). 

This may suggest why performance on these two metrics is inversely correlated. 

Moreover, the use cases for these models differs - GPT being optimised for high-precision text based tasks, and Gemini focusing on multimodality and extended context windows. It would seem sensible that an LLM with superior image generation capabilities would be deemed more ‘creative’, and hence score higher in the ‘creativity’ test. 

Structured knowledge tasks (MMLU) and creativity tasks (DAT) seem to be at odds with each other, and indeed, we may expect to see a similar inverse correlation in human-focused studies. 

![[./resources/scores.png]]
## Conclusion

Our initial investigation suggests that performance on creative tasks such as the Divergent Association Task (DAT) does not necessarily correlate with performance on reasoning tasks, as measured by benchmarks like MMLU. Indeed, these results suggest the opposite may be true—models optimized for structured-knowledge and factual reasoning tasks may exhibit **reduced performance on creative tasks**, likely due to their architectural focus on logical consistency and structured outputs.

Moreover, while the trend in state-of-the-art (SoA) LLMs leans towards **multi-modality**, this additional capacity appears to benefit models in creative tasks such as DAT. Multi-modal models like Gemini 1.5 Flash, designed for handling diverse input types, excel in creativity over purely text-based models. To achieve a fair comparison, creative tasks should perhaps be measured more consistently within text-only models.

Both models demonstrated strong performance in their respective tasks; however, the **inverse relationship** between creativity and reasoning abilities may be due to differences in **architectural optimization**. GPT-4o, with its strong reasoning capabilities, excels in structured knowledge tasks, while Gemini 1.5 Flash’s broader input context and multi-modal strengths favor creative output, suggesting a potential **reasoning-creativity trade-off**.

To measure creativity more accurately, future experiments could integrate **multi-modal elements** of LLMs, such as combining tasks like DAT with creative visual assessments. Additionally, a more nuanced approach to creativity could involve evaluating a model’s ability to produce **innovative solutions** to structured problems, like mathematics, where elegant problem-solving reflects deeper creative capabilities, as opposed to brute-force methods that indicate weaker creativity.

This would align the creativity measurement more closely with the model’s training distribution, providing a more reliable measure of creative reasoning within the framework of structured knowledge tasks.§
