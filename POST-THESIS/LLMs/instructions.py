instruction_active_passive = """
You are a social scientist applying discourse analysis over Lebanese newspapers.
You will be given text taken from a Lebanese newspaper
Identify sentences that contain either an active or a passive voice:
For each sentence you retrieve, return the following:
If the sentence contains a passive voice, output:
- Voice: <Passive>
- Sentence: <the extracted sentence>
- Passive phrase(s): <the verbs or phrases that were used in the passive form>.
If the sentence contains an active voice, output:
- Voice: <Active>
- Sentence: <the extracted sentence>
- Active Agent(s): <the agent(s) that was the active voice in the sentence>
Provide a justification explaining the reasoning behind your choice.
"""

instruction_relexicalization = """
You are a social scientist applying discourse analysis over Lebanese newspapers.
You will be given text taken from a Lebanese newspaper.
Identify the different personnel present in text. Personnel could be politicians or political Parties.
For each personnel identified, summarize the unique different ways in which they are referred to in text.
Return results in the following format:
* Personnel: <The personnel>
* Representations: <the different representations in text, separated by comma>
"""

instruction_modality = """
Your task is, given an excerpt of text taken from a Lebanese newspaper
Identify all sentences that contain:
* A clear subject that is either a politician, political party, country, or ethnicity.
* can, could, may, might, must, should, shall, would, will
If there is such a sentence, then for each sentence return:
- Sentence: the sentence containing any of the form(s) above.
- Form: the form(s) in the sentence.
- Subject(s): the subject(s) associated with form(s) above.
- Justification: Explanation of the reasoning behind your choice
If there is no such sentence:
- return 'None'
Provide a justification explaining the reasoning behind your choice.
"""


instruction_nominalization = """
You will be given an excerpt of text taken from a Lebanese newspaper.
Extract the sentences were a process is converted into a noun.
Look for the presence of the following scenarios to validate your extraction:
- A verb is exchanged with a simple noun or phrase
- The exchange above reduced the effect of the sentence were come meaning becomes missing
If you find such scenarios in the text, output:
- Sentence: <the extracted sentence>
- Reduction phrases: <the phrase(s) from the extracted sentence were a meaning is reduced>
- Justification: Provide a justification explaining the reasoning behind your choice.
If no such scenario exists, output:
- Sentence: None
- Reduction phrases: None
"""


instruction_repetition = """
You will be given an excerpt of text taken from a Lebanese newspaper.
Read the text and extract all the sentences, or clauses, that are repeated multiple times in the text.
Return each repeated sentence or clause on a separate line in the format:
Repeated sentence/clause #n: <the repeated sentence/clause>, number of repeats: <number of times it occurred>.
"""


########### CDA ############################
# note: the negative representations are considered positive in Itani's thesis
instruction_level_description_completeness = """
You will be given an excerpt of text taken from a Lebanese newspaper.
Extract all sentences that are describing a certain action made by: a politician, political party, or country.
The actions must be restricted to any form of military practices.
For each extracted sentence output:
- Sentence: The sentence containing the military action.
- Actor: The subject exercising the certain action
- Action: The action being done by the subject
- Degree of description: Either 'detailed' or 'broad'
- Representation: whether the action exercised by the subject is represented positively (positive) or negatively (negative).
"""

instruction_denomination = """
You will be given an excerpt of text taken from a Lebanese newspaper.
Extract all sentences where a certain ethnic group, political party, or organization are described as inferior with the use of words such as:
* opponents, immigrants, others, extremists, insurgents, armed, takfiri, militants, terrorists, rebels, etc.
For each extracted sentence output:
- Sentence: The sentence containing the aforementioned description.
- Actor: the group of of people described as inferior.
- Inferiority: the inferior word(s) used to describe the actor above. 
"""

instruction_agency = """
You will be given an excerpt of text taken from a Lebanese newspaper.
Extract all sentences where a certain ethnic group, political party, or organization are described as being responsible for committing or helping in the establishment of a negative consequence.
For each extracted sentence output:
- Sentence: The extracted sentence
- Agent: the actor responsible for the negative action
- Action: the negative action committed or eradicated by the actor.
"""


# must be improved a bit
instruction_disclaimer = """
You will be given an excerpt of text taken from a Lebanese newspaper.
Extract all sentences that:
- start by denying adverse feelings about a certain group: ethnicity, political party, or country
- the next part of the sentence, despite the start, is full of negative things or actions about the group.
For each extracted sentence output:
- Sentence: the extracted sentence
- Actor: the group which was described negatively in the second part of the sentence
- Transition: the part of the sentence where there was a transition in the description of the actor.
"""


instruction_dramatization = """
You will be given an excerpt of text taken from a Lebanese newspaper.
Extract all sentences that contain an exaggeration in a description of a group's certain actions.
For each extracted sentence output:
- Sentence: the extracted sentence containing the exaggeration.
- Actor: the group by which its action was described with exaggeration.
- Exaggerated words: the loaded words that yielded to the exaggerated representation.
"""