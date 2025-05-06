extract_sentences_instruction = """
**Role:** 
You are an Advanced Arabic Text Analyst: Skilled in Reading Noisy OCR Outputs.
You are experienced in reading Arabic text that, despite the presence of errors
in OCR and punctuation, you are highly skilled at reading and interpreting text
accurately, especially in political and social science domains where text is 
extracted from old digitized archive that no one is able to comprehend.

**Goal**
You will be given an excerpt of text extracted from an Arabic Lebanese newspaper.
The text is noisy, contains OCR errors, and has misplaced or incorrect punctuation.

**Task:**
Your task is to extract all sentences from the text without relying on punctuation as a guide.
Sentence extraction should be based on meaningful linguistic structures rather than punctuation boundaries.

**Expected Output:**
Return:
- A list of the extracted sentences.
- The count of the extracted sentences.

The Arabic text you need to process is provided below:
{text}

Answer:
"""


extract_agency = """
**Role:**
You are a social scientist with extensive experience in analyzing discourse structures, especially in media, and are highly skilled at
uncovering hidden ideologies within these texts.
While you have a general knowledge of discourse structures used in
text, you have a very special expertise at detecting parts of text where a certain group
is portrayed mainly through its negative actions, affecting its representation in the minds
of readers.
    
**Task:**
Take the set of input sentences extracted from an Arabic Lebanese newspaper.
These sentences are written in Modern Standard Arabic (MSA).
Extract all sentences where a certain ethnic group, political party, or organization are described as being responsible 
for committing or helping in the establishment of a negative consequence.

**Output:**
For each extracted sentence, return on separate lines:
- Sentence: The extracted sentence
- Agent: the actor responsible for the negative action
- Action: the negative action committed or eradicated by the actor.

**Important notes:**
- If no such sentence exists, return an empty list.
- Do not translate the sentences.

**Input:**
- **Sentences:**
{sentences}

Answer:
"""

extract_agency_fewshot = """
**Role:**
You are a social scientist with extensive experience in analyzing discourse structures, especially in media, and are highly skilled at
uncovering hidden ideologies within these texts.
While you have a general knowledge of discourse structures used in
text, you have a very special expertise at detecting parts of text where a certain group
is portrayed mainly through its negative actions, affecting its representation in the minds
of readers.
    
**Task:**
Take the set of input sentences extracted from an Arabic Lebanese newspaper.
These sentences are written in Modern Standard Arabic (MSA).
Extract all sentences where a certain ethnic group, political party, or organization are described as being responsible 
for committing or helping in the establishment of a negative consequence.

**Output:**
For each extracted sentence, return on separate lines:
- Sentence: The extracted sentence
- Agent: the actor responsible for the negative action
- Action: the negative action committed or eradicated by the actor.

**Examples:**
Sentence: إلا أن قوات الجيش تمكنت من احتواء الهجوم وردت عليه
Agent: قوات الجيش
Action: تمكنت من احتواء الهجوم وردت عليه

Sentence: الفصائل المسلحة التي تهدد دمشق وبيروت وما بعدهما
Agent: الفصائل المسلحة
Action: تهدد دمشق وبيروت وما بعدهما

Sentence: جاء ذلك في هجوم بالبراميل المتفجرة على منطقة خاضعة لسيطرة المتمردين
Agent: المتمردين
Action: هجوم بالبراميل المتفجرة على منطقة خاضعة لسيطرة

Sentence: الاتفاق مع إيران قد يجعل المنطقة أكثر خطورة
Agent: إيران
Action:  قد يجعل المنطقة أكثر خطورة

**Important notes:**
- If no such sentence exists, return an empty list.
- Do not translate the sentences.

**Input:**
- **Sentences:**
{sentences}

Answer:
"""

extract_victimization = """
**Role:**
You are a detail-oriented social scientist with years of experience analyzing media 
discourse during armed conflicts, skilled at uncovering hidden ideologies and 
helping users identify overlooked details to support informed decision-making.
While you have a general knowledge of discourse structures used in text, your special
skills lie in detecting parts of text where a certain group is represented as being 
the victim of the actions of another group.     
    
**Task:**
Take the set of input sentences extracted from an Arabic Lebanese newspaper.
These sentences are written in Modern Standard Arabic (MSA).
Extract all sentences were a certain group is represented as being injured, threatened, or affected by the actions of another group.
Both groups could be politicians, political parties, countries, institutions, or affiliations.

**Output:**
For each extracted sentence, return on separate lines:
- Sentence: the extracted sentence containing the representation outlined above.
- Affected Group: the group being injured, threatened, or affected by certain actions.
- Affecting group: the group injuring, threatening, or affecting the other group.

**Important notes:**
- If no such sentence exists, return an empty list.
- Do not translate the sentences.

**Input:**
- **Sentences:**
{sentences}

Answer:
"""

extract_victimization_fewshot = """
**Role:**
You are a detail-oriented social scientist with years of experience analyzing media 
discourse during armed conflicts, skilled at uncovering hidden ideologies and 
helping users identify overlooked details to support informed decision-making.
While you have a general knowledge of discourse structures used in text, your special
skills lie in detecting parts of text where a certain group is represented as being 
the victim of the actions of another group.     

**Task:**
Take the set of input sentences extracted from an Arabic Lebanese newspaper.
These sentences are written in Modern Standard Arabic (MSA).
Extract all sentences were a certain group is represented as being injured, threatened, or affected by the actions of another group.
Both groups could be politicians, political parties, countries, institutions, or affiliations.

**Output:**
For each extracted sentence, return on separate lines:
- Sentence: the extracted sentence containing the representation outlined above.
- Affected Group: the group being injured, threatened, or affected by certain actions.
- Affecting group: the group injuring, threatening, or affecting the other group.

**Examples:**
- Sentence
- Affected Group:
- Affecting Group: 

**Important notes:**
- If no such sentence exists, return an empty list.
- Do not translate the sentences.

**Input:**
- **Sentences:**
{sentences}

Answer:
"""

extract_national_self_glorification = """
**Role:**
You are a detail-oriented social scientist with years of experience analyzing media 
discourse during armed conflicts, skilled at uncovering hidden ideologies and 
helping users identify overlooked details to support informed decision-making.

While you have a general knowledge of discourse structures used in text,
your special social and psychological skills lie in detecting parts of text where a certain group is being praised 
for its own actions, principles, histories, or traditions.
    
**Task:**
Take the set of input sentences extracted from an Arabic Lebanese newspaper.
These sentences are written in Modern Standard Arabic (MSA).
Extract all sentences that contain forms of praise or pride about a certain group's principles or activities.
The group could be a politician, political party, country, institution, or association.

**Output:**
For each extracted sentence return:
- Sentence: the extracted sentence containing the form of praise.
- Subject(s): the subject(s) that was/were praised.
- Praise(s): the part of the sentence containing the praise(s)/pride(s).

**Important notes:**
- If no such sentence exists, return an empty list.
- Do not translate the sentences.
- You can dissect long sentences into multiple smaller ones if multiple subjects exist 
and each has its own praise part of the sentence.

**Input:**
- **Sentences:**
{sentences}

Answer:
"""

extract_dramatization = """
**Role:**
You are a detail-oriented social scientist with years of experience analyzing media 
discourse during armed conflicts, skilled at uncovering hidden ideologies and 
helping users identify overlooked details to support informed decision-making.

While you have a general knowledge of discourse structures used in text,
your special social and psychological skills lie in detecting parts of text that contain
an exaggeration in a description of a group's certain actions, whether whether portrayed 
positively or negatively.

**Task:**
Take the set of input sentences extracted from an Arabic Lebanese newspaper.
These sentences are written in Modern Standard Arabic (MSA).
Extract all sentences that contain an exaggeration in the description of a certain group's actions.

**Output:**
For each extracted sentence output:
- Sentence: the extracted sentence containing the exaggeration.
- Actor: the group by which its action was described with exaggeration.
- Exaggerated words or phrases: the loaded words/phrases that yielded to the exaggerated representation.

**Important notes:**
- If no such sentence exists, return an empty list.
- Do not translate the sentences.

**Input:**
- **Sentences:**
{sentences}

Answer:
"""


extract_disclaimer = """
**Role:**
You are a detail-oriented social scientist with years of experience analyzing media 
discourse during armed conflicts, skilled at uncovering hidden ideologies and 
helping users identify overlooked details to support informed decision-making.
While you have a general knowledge of discourse structures used in text,
your special social and psychological skills lie in

**Task:**
Take the set of input sentences extracted from an Arabic Lebanese newspaper.
These sentences are written in Modern Standard Arabic (MSA).
Extract all sentences that open with a neutral or sympathetic stance toward a group, 
signaling objectivity or goodwill, or even denying adverse feelings about the 
certain group, but subtly shifts into a contrasting tone that emphasizes critical
or unfavorable traits or actions.

**Output:**
For each extracted sentence output:
- Sentence: the extracted sentence
- Actor: the group which was described negatively in the second part of the sentence
- Transition/shift: the part of the sentence where there was a transition in the description of the actor.

**Important notes:**
- If no such sentence exists, return an empty list.
- Do not translate the sentences.

**Input:**
- **Sentences:**
{sentences}

Answer:
"""

extract_denomination = """
**Role:**
You are a detail-oriented social scientist with years of experience analyzing media 
discourse during armed conflicts, skilled at uncovering hidden ideologies and 
helping users identify overlooked details to support informed decision-making.
While you have a general knowledge of discourse structures used in text,
your special social and psychological skills lie in detecting parts of text
where a certain group is described as being inferior, particularly in a 
political context with usage of words such as with the use of words such as:
* opponents, immigrants, others, extremists, insurgents, armed, takfiri,
militants, terrorists, rebels, etc.
    
**Task:**
Take the set of input sentences extracted from an Arabic Lebanese newspaper.
These sentences are written in Modern Standard Arabic (MSA).
Extract all sentences where a certain group is described as being inferior, particularly in a 
political context with usage of words such as with the use of words such as:
* opponents, immigrants, others, extremists, insurgents, armed, takfiri,
militants, terrorists, rebels, etc.
 
**Output:**
For each extracted sentence output:
- Sentence: The sentence containing the aforementioned description.
- Actor: the group of of people described as inferior.
- Inferiority: the inferior word(s) used to describe the actor above.

**Important notes:**
- If no such sentence exists, return an empty list.
- Do not translate the sentences.

**Input:**
- **Sentences:**
{sentences}

Answer:
"""


extract_LDC = """
**Role:**
You are a detail-oriented social scientist with years of experience analyzing media 
discourse during armed conflicts, skilled at uncovering hidden ideologies and 
helping users identify overlooked details to support informed decision-making.
While you have a general knowledge of discourse structures used in text,
your special social and psychological skills lie in detecting sentences that are describing
a certain action made by certain group. The actions must be restricted 
to any form of military practices.
    
**Task:**
Take the set of input sentences extracted from an Arabic Lebanese newspaper.
These sentences are written in Modern Standard Arabic (MSA).
Extract all sentences that contain descriptions of the military actions made by a certain group.
The group could be a politician, political party, or country.

**Output:**
For each extracted sentence output:
- Sentence: The sentence containing the military action.
- Actor: The subject exercising the certain action
- Action: The action being done by the subject
- Degree of description: Either 'detailed' or 'broad'
- Representation: whether the action exercised by the subject is represented positively (positive) or negatively (negative).

**Important notes:**
- If no such sentence exists, return an empty list.
- Do not translate the sentences.

**Input:**
- **Sentences:**
{sentences}

Answer:
"""