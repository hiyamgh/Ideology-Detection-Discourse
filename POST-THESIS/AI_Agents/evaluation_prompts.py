query_analysis_prompt = """
You will be presented with a query.

**Task:**
Analyze the query to do the following:
- Carefully read each sentence of the query.
- Identify the core problem or question being asked.

**Output**:
- Key question: Summary description of the core problem being asked. 

**Query**:
{query}

Answer:
"""

document_analysis_prompt = """
You will be presented with a query, key question of the query, and a set of sentences.

##### Task:
Analyze the set of sentences to:
- Thoroughly examine each sentence.
- List all sentences from that are relevant with respect to the query.
- Briefly explain how each sentence listed is relevant to the query.

##### Input:
Query: 
{query}

##### Query key question: 
{query_key_question}

##### Sentences:
{sentences}

Answer:
"""


judgement_prompt = """
You will be presented with a query, key question of the query, and a set of sentences, and an analysis of the sentences.

**Task:**
Assess if the analysis of the sentences is relevant to the query in one word:
- Yes: If the analysis is relevant to the query.
- No: Otherwise.

**Important Notes:**
- Respond using only one of the following two words without quotation marks: Yes or No.

##### Input:
Query: 
{query}

##### Query key question: 
{query_key_question}

##### Sentences:
{sentences}

#### Analysis of the sentences
{analysis}
"""