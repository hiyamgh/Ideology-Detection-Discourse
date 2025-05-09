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
Action: قد يجعل المنطقة أكثر خطورة

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
Both groups should be well known: politicians, political parties, countries, institutions, or affiliations.

**Output:**
For each extracted sentence, return on separate lines:
- Sentence: the extracted sentence containing the representation outlined above.
- Negatively affected Group: the group being injured, threatened, or affected by certain actions.
- Affecting group: the group injuring, threatening, or affecting the other group in a negative way.

**Important notes:**
- If no such sentence exists, return an empty list.
- Do not translate the sentences.
- If the affected group not known, don't add the sentence to the list of results.
- If the affecting group not known, don't add the sentence to the list of results.
- If the effect is not negative, don't add the sentence to the list of results.

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
- Sentence: وضع حد للابتزاز الذي يمارسه المتمردون الذين قطعوا إمدادات المياه عن سكان دمشق
- Affected Group:  سكان دمشق
- Affecting Group: المتمردون

- Sentence: حزب الله يطلق قذائف صاروخية وأسلحة آلية من داخل وفوق المباني السكنية
- Affected Group: داخل وفوق المباني السكنية
- Affecting Group: حزب الله

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

extract_national_self_glorification_fewshot = """
**Role:**
You are a detail-oriented social scientist with years of experience analyzing media 
discourse during armed conflicts, skilled at uncovering hidden ideologies and 
helping users identify overlooked details to support informed decision-making.

While you have a general knowledge of discourse structures used in text,
your special social and psychological skills lie in detecting parts of text where a 
certain group is being praised  for its own actions, principles, histories, or traditions.

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

**Examples:**
- Sentence: وكان الكمين الذي نصبه الجيش السوري ذا تأثير كبير، إذ رفع من معنويات الجيش السوري المرتفعة أصلاً، وأحبط العدو الذي ظن أن حلب أصبحت في متناوله.
- Subject(s):  الجيش السوري
- Praise(s):  ذا تأثير كبير, فع من معنويات الجيش السوري المرتفعة أصلاً, أحبط العدو الذي ظن أن حلب أصبحت في متناوله

- Sentence: يضاف إلى ذلك انتصارات حزب الله الأخيرة في سلسلة جبال لبنان الشرقية
- Subject(s): حزب الله
- Praise(s): انتصارات

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
- If the actor is not known, don't add the sentence to the list of results.

**Input:**
- **Sentences:**
{sentences}

Answer:
"""

extract_dramatization_fewshot = """
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
Extract all sentences that contain an exaggeration, overreaction, or sensationalism
in the description of a certain group's actions.

**Output:**
For each extracted sentence output:
- Sentence: the extracted sentence containing the exaggeration.
- Actor: the group by which its action was described with exaggeration.
- Exaggerated words or phrases: the loaded words/phrases that yielded to the exaggerated representation.

**Examples:**
- Sentence: وكان الهجوم الذي شنته وحدات الجيش السوري وقوات حزب الله… أعمق من المتوقع، ما أدى حتى الآن إلى خسائر أقل من المتوقع.
- Actor: الهجوم الذي شنته وحدات الجيش السوري وقوات حزب الله
- Exaggerated words or phrases: أعمق من المتوقع, خسائر أقل من المتوقع

- Sentence: الجيش السوري بعد اختراقه الناجح
- Actor: الجيش السوري
- Exaggerated words or phrases: اختراقه الناجح

- Sentence: ولم يساهم المسلحون بشكل كبير في تهديد العاصمة السورية فحسب، بل أيضاً الشريط الحدودي اللبناني، فضلاً عن الطريق السريع الدولي الحيوي الذي يربط البلدين.
- Actor: المسلحون
- Exaggerated words or phrases: بشكل كبير في تهديد العاصمة السورية, الشريط الحدودي اللبناني, الطريق السريع الدولي الحيوي الذي يربط البلدين, 

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
- Explanation justifying teh choice of the transition

**Important notes:**
- If no such sentence exists, return an empty list.
- Do not translate the sentences.

**Input:**
- **Sentences:**
{sentences}

Answer:
"""

extract_disclaimer_fewshot = """
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

**Examples:**
- Sentence: ورغم أن أنصار الشريعة لم تعلن عن فشل أو حل مركز قيادتها، إلا أن مصدراً ميدانياً يعتقد أن مصير المركز، كما سابقيه، كان الفشل.
- Actor: أنصار الشريعة
- Transition/Shift: ورغم أن...إلا أن مصدراً ميدانياً يعتقد...أن مصير المركز...كان الفشل

- Sentence: ويقوم التنظيم بمهاجمة محيط المطار لتخفيف الضغط على تدمر، على الرغم من أنه يدرك جيداً أنه لا يستطيع اختراق المطار.
- Actor: التنظيم
- Transition/Shift: ويقوم...بمهاجمة...لتخفيف الضغط على تدمر، على الرغم من أنه يدرك جيداً أنه لا يستطيع اختراق المطار

- Sentence: إن الاتفاق النووي الإيراني مع القوى العالمية يعني "يومًا سعيدًا" إذا منع البلاد من الحصول على ترسانة نووية، لكن الاتفاق قد يثبت أنه سيئ إذا سمح لطهران "بإحداث الفوضى في المنطقة"
- Actor: الاتفاق النووي الإيراني مع القوى العالمية
- Transition/Shift: إن...يعني "يومًا سعيدًا"...لكن الاتفاق قد يثبت أنه سيئ إذا سمح لطهران "بإحداث الفوضى في المنطقة

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
where a certain group is described as being inferior, particularly in 
political contexts.
    
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
a certain action (manifesting in military practices) made by certain group, and detecting
the level of detail associated with such descriptions.
    
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