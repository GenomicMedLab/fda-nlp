# huggingface-fda

### About
   
Even with major updates, as new scientific research is published and new therapeutics are approved by the FDA, there become disparities between computable data sources and the true state of treatment knowledge. In order to improve data parity between computable knowledge and the published state of treatment knowledge, we aim to create tooling that is able to automate (or assist) ingestion of live data sourcing in a scalable manner. Using advanced NLP techniques, we can capture therapeutic data as it becomes available from definitive sources, regulatory approval bodies, and scientific publications. 

### Current Approach

The current strategy for creating this pipeline involves utilizing the Huggingface ecosystem to perform key NLP tasks on key pieces of data. Amazon Titan (LLM) is also being explored for potential in performing similar NLP tasks. 

This project is very much still in an exploratory phase and all work is done out of ipynb currently. Data files are not included in the repo but can be provided upon request. Any files named 'explore' are almost always a reference scratchpad.

**File order**:  
  
*FDA*  
- download_json.ipynb
  
*HuggingFace*  
- ner_tagging.ipynb
  
*EvalSet*
- select-random.ipynb
- evaluate.ipynb

### TO DO  

*From 10/24/23 Check-in:*  
- [ ] Look under the hood to calculate a proper PR curve
- [ ] Evaluate for GENETIC and DISEASE
- [ ] Look under the hood for proper post-processing

*From 10/5/23 NLP Meeting:*  
- [x] Look to see if there is a 'toolbox' function that rejoins chunks
- [x] Create a labeled dataset for evaluation purposes (32 to 100 sentences)
- [ ] ~~If no toolbox, add in post-processing~~
  
*From 9/14/23 NLP Meeting:*
- [x] Upload preprocessing for label dataset
- [x] Introduce post-processing (joining ner chunks?)
- [ ] ~~Revisit preprocessing steps (white space, TM characters, splits, etc.)~~
- [ ] ~~Characterize ADE model for better understanding~~

### Related Links
- https://www.dgidb.org/
- https://huggingface.co/
- https://aws.amazon.com/bedrock/titan/
- https://www.fda.gov/drugs/drug-approvals-and-databases/drugsfda-data-files
