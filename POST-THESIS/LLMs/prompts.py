prompts_dict = {
    'Bachir_Gemayel_Election': {
        'modality': {
            'direct': '''
You are a social scientist applying discourse analysis over Lebanese newspapers.
Your task is, given an excerpt of text taken from a Lebanese newspaper, to identify whether a sentence contains any of the following:
* can, could, may, might, must, should, shall, would, will
If there is such a sentence, retrieve and return the sentence. For each sentence you retrieve, put it in a new line.
Text: {text}
Answer: 
        ''',
        'direct_fewshot': ''' 
You are a social scientist applying discourse analysis over Lebanese newspapers.
Your task is, given an excerpt of text taken from a Lebanese newspaper, to identify whether a sentence contains any of the following:
* can, could, may, might, must, should, shall, would, will
If there is such a sentence, retrieve and return the sentence. For each sentence you retrieve, put it in a new line.

Example: اسرائيل لا تزال العدو ولكن هل علينا ادانتها يوميا؟
(Israel is still the enemy but do we have to condemn it every day?)
Explanation: The use of the interrogative sentence 'do we have to condemn Israel…' which contains modality regarding the obligation to keep condemning Israel every time. It implies that the Lebanese people are requiring the person asking to repeat the same words on a regular basis.

Example: هناك افكار عديدة يجب جمعها ومقالرنتها واستخلاص ما يجب استخلاصه منها
(Most of the opinions should be obtained and compared and conclusions should be drawn in this regard)
Explanation:  Deontic modal verbs are introduced to emphasize on an opinion that is an uncertainty.

Text: {text}
Answer: 
        ''',

            'direct_context': '''
You are a social scientist applying discourse analysis over Lebanese newspapers.
Bachir Gemayel was the commander of Lebanese forces, which was the military wing of the Kataeb party during the Lebanese Civil War.
The Palestinian presence through the PLO as well as the Syrian presence was angering many Lebanese leftists, and the IDF had plans of rooting up the PLO threat to Israel.
Gemayel allied with Israel and his forces fought the PLO and the Syrian Army.
The context above generated a lot of controversy with differing stances towards Gemayel and the Kataeb Party, which was demonstrated through many Lebanese newspapers.
Your task is, given an excerpt of text taken from a Lebanese newspaper, to identify whether a sentence contains any of the following:
* can, could, may, might, must, should, shall, would, will
If there is such a sentence, retrieve and return the sentence. For each sentence you retrieve, put it in a new line.
Text: {text}
Answer: 
        ''',
            'direct_fewshot_context': '''
You are a social scientist applying discourse analysis over Lebanese newspapers.
Bachir Gemayel was the commander of Lebanese forces, which was the military wing of the Kataeb party during the Lebanese Civil War.
The Palestinian presence through the PLO as well as the Syrian presence was angering many Lebanese leftists, and the IDF had plans of rooting up the PLO threat to Israel.
Gemayel allied with Israel and his forces fought the PLO and the Syrian Army.
The context above generated a lot of controversy with differing stances towards Gemayel and the Kataeb Party, which was demonstrated through many Lebanese newspapers.
Your task is, given an excerpt of text taken from a Lebanese newspaper, to identify whether a sentence contains any of the following:
* can, could, may, might, must, should, shall, would, will
If there is such a sentence, retrieve and return the sentence. For each sentence you retrieve, put it in a new line.

Example: اسرائيل لا تزال العدو ولكن هل علينا ادانتها يوميا؟
(Israel is still the enemy but do we have to condemn it every day?)
Explanation: The use of the interrogative sentence 'do we have to condemn Israel…' which contains modality regarding the obligation to keep condemning Israel every time. It implies that the Lebanese people are requiring the person asking to repeat the same words on a regular basis.

Example: هناك افكار عديدة يجب جمعها ومقالرنتها واستخلاص ما يجب استخلاصه منها
(Most of the opinions should be obtained and compared and conclusions should be drawn in this regard)
Explanation: Deontic modal verbs are introduced to emphasize on an opinion that is an uncertainty.

Text: {text}
Answer: 
    '''
        },
        'active_passive_voice': {
            'direct': '''
You are a social scientist applying discourse analysis over Lebanese newspapers.
Your task is, given an excerpt of text extracted from a Lebanese newspaper, to identify sentences that contain either an active or a passive voice.
For each sentence you retrieve, return the following:
If the sentence contains a passive voice, output:
Voice: <Passive>, Sentence: <the extracted sentence>, Passive phrase(s): <the verbs or phrases that were used in the passive form>.
If the sentence contains an active voice, output:
Voice: <Active>, Sentence: <the extracted sentence>, Active Agent(s): <the agent(s) that was the active voice in the sentence>.
Text: {text}
Answer: 
            ''',
            'direct_fewshot': '''
You are a social scientist applying discourse analysis over Lebanese newspapers.
Your task is, given an excerpt of text extracted from a Lebanese newspaper, to identify sentences that contain either an active or a passive voice.
For each sentence you retrieve, return the following:
If the sentence contains a passive voice, output:
Voice: <Passive>, Sentence: <the extracted sentence>, Passive phrase(s): <the verbs or phrases that were used in the passive form>.
If the sentence contains an active voice, output:
Voice: <Active>, Sentence: <the extracted sentence>, Active Agent: <the agent(s) that was the active voice in the sentence>.

Example: منها ما يقراء بين السطور
Voice: <Passive>, Sentence: <منها ما يقراء بين السطور>, Passive phrase(s): <ما يقراء>
Explanation: the actor is unknown and it is unclear who hid the meaning between the lines

Example: مشيرا الى ان الحرب الاسرائءيلية التي شنت على لبنان (indicating that the Israeli war which was imposed on lebanon was not balanced)
Voice: <Passive>, Sentence: <مشيرا الى ان الحرب الاسرائيلية التي شنت على لبنان>, Passive phrase(s): <الحرب الاسرائيلية التي شنت على لبنان>
Explanation: The verb “was imposed” is a passive material process, where the passive voice is deleting an unspecified actor.

Example: وفهم من كلامه
Voice: <Passive>, Sentence: <وفهم من كلامه>, Passive Phrases: <وفهم من كلامه>
Explanation: (It was understandable from his words) This sentence represents a verbal passive process where the agent is unidentified and unspecified.

Example: اتخذوا من ارضنا ساحة عراك لهم وليس لنا وكانت حربا بالوساطة
Voice: <Passive>, Sentence: <اتخذوا من ارضنا ساحة عراك لهم وليس لنا وكانت حربا بالوساطة>, Passive phrase(s): <اتخذوا من ارضنا>
Explanation: (our territory was exploited as a battlefield for them and not for us. It was a proxy war). The verb “was exploited” is a material verb used in the passive form where the agent is unknown. The journalist does not mention the actor who exploited the territory of Lebanon as a battlefield. The agent is unidentified in this process. The verb “imposed” does not indicate who the actor is that “imposed” the war on Lebanon. It is a passive material process
           
Example: هناك افكار عديدة يجب جمعها ومقارنتها واستخلاص ما يجب استخلاصه منها
Voice: <Passive>, Sentence: <هناك افكار عديدة يجب جمعها ومقارنتها واستخلاص ما يجب استخلاصه منها>, Passive phrase(s): < يجب جمعها ومقارنتها, واستخلاص ما يجب استخلاصه منها>
Explanation: (Most of the opinions should be obtained and compared and conclusions should be drawn in this regard.) The verbs “should be obtained”, “compared and drawn” are verbs used in the passive form where the agent is missing. The usage of “should” indicates an obligation.
            
Example: يصعب جدا تأليف حكومة سواها
Voice: <Passive>, Sentence: <يصعب جدا تأليف حكومة سواها>, Passive phrase(s): <يصعب جدا>
Explanation: The verb in Arabic “it is too difficult” is a passive mental process, which omits the agent who might form the government. It has a negative meaning as it implies that it is complicated and difficult to form a government.

Example: انهم عوضوا على اللذين هدمت منازلهم اخيرا, وهؤلاء مر على تهجيرهم ثلاثون سنة ولم يعوض عليهم
Voice: <Passive>, Sentence: <انهم عوضوا على اللذين هدمت منازلهم اخيرا, وهؤلاء مر على تهجيرهم ثلاثون سنة ولم يعوض عليهم>, Passive Phrase(s): <ولم يعوض عليهم, عوضوا>
Explanation: (compensation had been paid to the people whose homes had recently been destroyed, but that those who have been homeless for thirty years have not yet been paid”) The verbs “had been paid”, “have not yet been paid” are uttered in a passive form where the actor is unknown. The second verb has a negative meaning as the damages have not been paid to the people of the mountain who had left their homes.

Example: هدد وزير الدفاع ليون بانيتا إيران بضربة عسكرية إذا طورت أسلحة نووية، لكن إسرائيل تقول تلك الوعود باستخدام القوة ليست كافية 
Voice: <Active>, Sentence: <هدد وزير الدفاع ليون بانيتا إيران بضربة عسكرية إذا طورت أسلحة نووية، لكن إسرائيل تقول تلك الوعود باستخدام القوة ليست كافية>, Active agent(s): <وزير الدفاع ليون بانيتا, إسرائيل>
Explanation: Defense Secretary Leon Panetta and Israel are strong entities that can take practical measurements against Iran if needed – through the use of an active voice which makes the subjects seem powerful.

Example: وحذرت المخابرات السرية
Voice: <Active>, Sentence: <وحذرت المخابرات السرية>, Active Agent(s): <المخابرات السرية>
Explanation: Secret Intelligence Service is presented in active voice as it is issuing a warning.

Example: قصفت القوات النيجيرية مخبأً نائياً لبوكو حرام فيما انضمت طائرات مقاتلة لقصف متجدد هجوم عسكري يهدف إلى طرد المجموعة
Voice: <Active>, Sentence: <قصفت القوات النيجيرية مخبأً نائياً لبوكو حرام فيما انضمت طائرات مقاتلة لقصف متجدد هجوم عسكري يهدف إلى طرد المجموعة>, Active Agent(s): <القوات النيجيرية>
Explanation: Nigerian troops are represented as strong entities is destructive and can flush out the Boko Hraam group.

Example: أعلن الجيش النيجيري حظر التجول لمدة 24 ساعة في عشرات الأحياء في البلاد مدينة شمال شرق البلاد وهي معقل لجماعة بوكو حرام المسلحة
Voice: <Active>, Sentence: <أعلن الجيش النيجيري حظر التجول لمدة 24 ساعة في عشرات الأحياء في البلاد مدينة شمال شرق البلاد وهي معقل لجماعة بوكو حرام المسلحة>, ACtive Agent(s): <الجيش النيجيري>
Explanation: Nigeria's military is represented as a strong and authoritarian entity because of its ability to "declare" a "curfew" in a city that is a stronghold of the armed group Boko Haram.

Example: وتنظر إسرائيل إلى التهديد الذي تمثله إيران المسلحة نوويا
Voice: <Active>, Sentence: <وتنظر إسرائيل إلى التهديد الذي تمثله إيران المسلحة نوويا>, Active Agent(s): <اسرائيل>
Explanation: Israel is represented as an authoritarian entity that is "looking over" the threats posed by "The nuclear-ly armed Iran".

Text: {text}
Answer: 
            ''',
            'direct_context': '''
You are a social scientist applying discourse analysis over Lebanese newspapers.
Bachir Gemayel was the commander of Lebanese forces, which was the military wing of the Kataeb party during the Lebanese Civil War.
The Palestinian presence through the PLO as well as the Syrian presence was angering many Lebanese leftists, and the IDF had plans of rooting up the PLO threat to Israel.
Gemayel allied with Israel and his forces fought the PLO and the Syrian Army.
The context above generated a lot of controversy with differing stances towards Gemayel and the Kataeb Party, which was demonstrated through many Lebanese newspapers.
Your task is, given an excerpt of text extracted from a Lebanese newspaper, to identify sentences that contain either an active or a passive voice.
For each sentence you retrieve, return the following:
If the sentence contains a passive voice, output:
Voice: <Passive>, Sentence: <the extracted sentence>, Passive phrase(s): <the verbs or phrases that were used in the passive form>.
If the sentence contains an active voice, output:
Voice: <Active>, Sentence: <the extracted sentence>, Active Agent(s): <the agent(s) that was the active voice in the sentence>.
Text: {text}
Answer: 
            ''',
            'direct_fewshot_context': '''
You are a social scientist applying discourse analysis over Lebanese newspapers.
Bachir Gemayel was the commander of Lebanese forces, which was the military wing of the Kataeb party during the Lebanese Civil War.
The Palestinian presence through the PLO as well as the Syrian presence was angering many Lebanese leftists, and the IDF had plans of rooting up the PLO threat to Israel.
Gemayel allied with Israel and his forces fought the PLO and the Syrian Army.
The context above generated a lot of controversy with differing stances towards Gemayel and the Kataeb Party, which was demonstrated through many Lebanese newspapers.
Your task is, given an excerpt of text extracted from a Lebanese newspaper, to identify sentences that contain either an active or a passive voice.
For each sentence you retrieve, return the following:
If the sentence contains a passive voice, output:
Voice: <Passive>, Sentence: <the extracted sentence>, Passive phrase(s): <the verbs or phrases that were used in the passive form>.
If the sentence contains an active voice, output:
Voice: <Active>, Sentence: <the extracted sentence>, Active Agent(s): <the agent(s) that was the active voice in the sentence>.

Example: منها ما يقراء بين السطور
Voice: <Passive>, Sentence: <منها ما يقراء بين السطور>, Passive phrase(s): <ما يقراء>
Explanation: the actor is unknown and it is unclear who hid the meaning between the lines

Example: مشيرا الى ان الحرب الاسرائءيلية التي شنت على لبنان (indicating that the Israeli war which was imposed on lebanon was not balanced)
Voice: <Passive>, Sentence: <مشيرا الى ان الحرب الاسرائيلية التي شنت على لبنان>, Passive phrase(s): <الحرب الاسرائيلية التي شنت على لبنان>
Explanation: The verb “was imposed” is a passive material process, where the passive voice is deleting an unspecified actor.

Example: وفهم من كلامه
Voice: <Passive>, Sentence: <وفهم من كلامه>, Passive Phrases: <وفهم من كلامه>
Explanation: (It was understandable from his words) This sentence represents a verbal passive process where the agent is unidentified and unspecified.

Example: اتخذوا من ارضنا ساحة عراك لهم وليس لنا وكانت حربا بالوساطة
Voice: <Passive>, Sentence: <اتخذوا من ارضنا ساحة عراك لهم وليس لنا وكانت حربا بالوساطة>, Passive phrase(s): <اتخذوا من ارضنا>
Explanation: (our territory was exploited as a battlefield for them and not for us. It was a proxy war). The verb “was exploited” is a material verb used in the passive form where the agent is unknown. The journalist does not mention the actor who exploited the territory of Lebanon as a battlefield. The agent is unidentified in this process. The verb “imposed” does not indicate who the actor is that “imposed” the war on Lebanon. It is a passive material process
           
Example: هناك افكار عديدة يجب جمعها ومقارنتها واستخلاص ما يجب استخلاصه منها
Voice: <Passive>, Sentence: <هناك افكار عديدة يجب جمعها ومقارنتها واستخلاص ما يجب استخلاصه منها>, Passive phrase(s): < يجب جمعها ومقارنتها, واستخلاص ما يجب استخلاصه منها>
Explanation: (Most of the opinions should be obtained and compared and conclusions should be drawn in this regard.) The verbs “should be obtained”, “compared and drawn” are verbs used in the passive form where the agent is missing. The usage of “should” indicates an obligation.
            
Example: يصعب جدا تأليف حكومة سواها
Voice: <Passive>, Sentence: <يصعب جدا تأليف حكومة سواها>, Passive phrase(s): <يصعب جدا>
Explanation: The verb in Arabic “it is too difficult” is a passive mental process, which omits the agent who might form the government. It has a negative meaning as it implies that it is complicated and difficult to form a government.

Example: انهم عوضوا على اللذين هدمت منازلهم اخيرا, وهؤلاء مر على تهجيرهم ثلاثون سنة ولم يعوض عليهم
Voice: <Passive>, Sentence: <انهم عوضوا على اللذين هدمت منازلهم اخيرا, وهؤلاء مر على تهجيرهم ثلاثون سنة ولم يعوض عليهم>, Passive Phrase(s): <ولم يعوض عليهم, عوضوا>
Explanation: (compensation had been paid to the people whose homes had recently been destroyed, but that those who have been homeless for thirty years have not yet been paid”) The verbs “had been paid”, “have not yet been paid” are uttered in a passive form where the actor is unknown. The second verb has a negative meaning as the damages have not been paid to the people of the mountain who had left their homes.

Example: هدد وزير الدفاع ليون بانيتا إيران بضربة عسكرية إذا طورت أسلحة نووية، لكن إسرائيل تقول تلك الوعود باستخدام القوة ليست كافية 
Voice: <Active>, Sentence: <هدد وزير الدفاع ليون بانيتا إيران بضربة عسكرية إذا طورت أسلحة نووية، لكن إسرائيل تقول تلك الوعود باستخدام القوة ليست كافية>, Active agent(s): <وزير الدفاع ليون بانيتا, إسرائيل>
Explanation: Defense Secretary Leon Panetta and Israel are strong entities that can take practical measurements against Iran if needed – through the use of an active voice which makes the subjects seem powerful.

Example: وحذرت المخابرات السرية
Voice: <Active>, Sentence: <وحذرت المخابرات السرية>, Active Agent(s): <المخابرات السرية>
Explanation: Secret Intelligence Service is presented in active voice as it is issuing a warning.

Example: قصفت القوات النيجيرية مخبأً نائياً لبوكو حرام فيما انضمت طائرات مقاتلة لقصف متجدد هجوم عسكري يهدف إلى طرد المجموعة
Voice: <Active>, Sentence: <قصفت القوات النيجيرية مخبأً نائياً لبوكو حرام فيما انضمت طائرات مقاتلة لقصف متجدد هجوم عسكري يهدف إلى طرد المجموعة>, Active Agent(s): <القوات النيجيرية>
Explanation: Nigerian troops are represented as strong entities is destructive and can flush out the Boko Hraam group.

Example: أعلن الجيش النيجيري حظر التجول لمدة 24 ساعة في عشرات الأحياء في البلاد مدينة شمال شرق البلاد وهي معقل لجماعة بوكو حرام المسلحة
Voice: <Active>, Sentence: <أعلن الجيش النيجيري حظر التجول لمدة 24 ساعة في عشرات الأحياء في البلاد مدينة شمال شرق البلاد وهي معقل لجماعة بوكو حرام المسلحة>, ACtive Agent(s): <الجيش النيجيري>
Explanation: Nigeria's military is represented as a strong and authoritarian entity because of its ability to "declare" a "curfew" in a city that is a stronghold of the armed group Boko Haram.

Example: وتنظر إسرائيل إلى التهديد الذي تمثله إيران المسلحة نوويا
Voice: <Active>, Sentence: <وتنظر إسرائيل إلى التهديد الذي تمثله إيران المسلحة نوويا>, Active Agent(s): <اسرائيل>
Explanation: Israel is represented as an authoritarian entity that is "looking over" the threats posed by "The nuclear-ly armed Iran".

Text: {text}
Answer: 
            '''
        },
        'nominalisation': {
            'direct': '''
You are a social scientist applying discourse analysis over Lebanese newspapers.
Your task is, given an excerpt of text extracted from a Lebanese newspaper, to extract the sentences were a process is converted into a noun, by exchanging a verb with a simple noun or phrase, reducing it were some meaning becomes missing.
For each sentence you retrieve, return the following:
Sentence: <the extracted sentence>, Reduction phrase(s): <the phrase(s) from the extracted sentence were a meaning is reduced>
Text: {text}
Answer: 
            ''',
            'direct_fewshot': '''
You are a social scientist applying discourse analysis over Lebanese newspapers.
Your task is, given an excerpt of text extracted from a Lebanese newspaper, to extract the sentences were a process is converted into a noun, by exchanging a verb with a simple noun or phrase, reducing it were some meaning becomes missing.
For each sentence you retrieve, return the following:
Sentence: <the extracted sentence>, Reduction phrase(s): <the phrase(s) from the extracted sentence were a meaning is reduced>

Example: أدان رئيس الوزراء الكندي حادث إطلاق النار على مسجد في كيبيك الليلة الماضية ووصفه بأنه عمل إرهابي، حيث أكدت خدمات الطوارئ مقتل ستة أشخاص في الهجوم. وقُتل المصلون بالرصاص وأصيب عدد أكبر
Sentence: <أدان رئيس الوزراء الكندي حادث إطلاق النار على مسجد في كيبيك الليلة الماضية ووصفه بأنه عمل إرهابي، حيث أكدت خدمات الطوارئ مقتل ستة أشخاص في الهجوم. وقُتل المصلون بالرصاص وأصيب عدد أكبر.> Reduction phrase(s): <حادث إطلاق النار, عمل>
Explanation:  “shootings” and “act” enable agent concealment, as they offer the writer the possibility of hiding the agent. compression of information in a single word or phrase leads to the depersonalization of the agent, which might in turn lead to a certain degree of ambiguity concerningwho did what to whom.

Example: وضربت أربعة انفجارات لندن قبل وقت قصير من الساعة التاسعة صباحا، ثلاثة منها في قطارات أنفاق لندن وواحد في حافلة ذات طابقين كانت تسير عبر بلومزبري. وقالت الشرطة إن عدد القتلى المؤكد هو 37 وهو في ارتفاع. وأصيب نحو 700 من الركاب والسائحين، نُقل نصفهم تقريباً إلى المستشفى بسيارات الإسعاف.  لقد كان أسوأ هجوم إرهابي في المملكة المتحدة وحمل جميع السمات المميزة لهجوم القاعدة، على غرار تفجيرات قطارات مدريد في مارس 2004. وأعلنت خلية مجهولة تابعة للقاعدة مسؤوليتها في ادعاء لا يمكن التحقق منه على موقع إسلامي على الإنترنت وأعلنت :"بريطانيا الآن تحترق بالخوف.".
Sentence: <وضربت أربعة انفجارات لندن قبل وقت قصير من الساعة التاسعة صباحا، ثلاثة منها في قطارات أنفاق لندن وواحد في حافلة ذات طابقين كانت تسير عبر بلومزبري. وقالت الشرطة إن عدد القتلى المؤكد هو 37 وهو في ارتفاع. وأصيب نحو 700 من الركاب والسائحين، نُقل نصفهم تقريباً إلى المستشفى بسيارات الإسعاف.>. Reduction phrase(s): <أربعة انفجارات>.
Explanation: “four blasts” could have been “X blasted London”. “blast” is transformed into an entity which then plays the role of the agent in the clause. It is, in otherwords, “the blasts” which is represented as carrying out the action of “hitting”, and is a representation of Muslims through passivization in British News Media Discourse.

Text: {text}
Answer: 
            ''',

            'direct_context': '''
You are a social scientist applying discourse analysis over Lebanese newspapers.
Bachir Gemayel was the commander of Lebanese forces, which was the military wing of the Kataeb party during the Lebanese Civil War.
The Palestinian presence through the PLO as well as the Syrian presence was angering many Lebanese leftists, and the IDF had plans of rooting up the PLO threat to Israel.
Gemayel allied with Israel and his forces fought the PLO and the Syrian Army.
The context above generated a lot of controversy with differing stances towards Gemayel and the Kataeb Party, which was demonstrated through many Lebanese newspapers.
Your task is, given an excerpt of text extracted from a Lebanese newspaper, to extract the sentences were a process is converted into a noun, by exchanging a verb with a simple noun or phrase, reducing it were some meaning becomes missing.
For each sentence you retrieve, return the following:
Sentence: <the extracted sentence>, Reduction phrase(s): <the phrase(s) from the extracted sentence were a meaning is reduced>
Text: {text}
Answer: 
            ''',
            'direct_fewshot_context': '''
You are a social scientist applying discourse analysis over Lebanese newspapers.
Bachir Gemayel was the commander of Lebanese forces, which was the military wing of the Kataeb party during the Lebanese Civil War.
The Palestinian presence through the PLO as well as the Syrian presence was angering many Lebanese leftists, and the IDF had plans of rooting up the PLO threat to Israel.
Gemayel allied with Israel and his forces fought the PLO and the Syrian Army.
The context above generated a lot of controversy with differing stances towards Gemayel and the Kataeb Party, which was demonstrated through many Lebanese newspapers.
Your task is, given an excerpt of text extracted from a Lebanese newspaper, to extract the sentences were a process is converted into a noun, by exchanging a verb with a simple noun or phrase, reducing it were some meaning becomes missing.
For each sentence you retrieve, return the following:
Sentence: <the extracted sentence>, Reduction phrase(s): <the phrase(s) from the extracted sentence were a meaning is reduced>

Example: أدان رئيس الوزراء الكندي حادث إطلاق النار على مسجد في كيبيك الليلة الماضية ووصفه بأنه عمل إرهابي، حيث أكدت خدمات الطوارئ مقتل ستة أشخاص في الهجوم. وقُتل المصلون بالرصاص وأصيب عدد أكبر
Sentence: <أدان رئيس الوزراء الكندي حادث إطلاق النار على مسجد في كيبيك الليلة الماضية ووصفه بأنه عمل إرهابي، حيث أكدت خدمات الطوارئ مقتل ستة أشخاص في الهجوم. وقُتل المصلون بالرصاص وأصيب عدد أكبر.> Reduction phrase(s): <حادث إطلاق النار, عمل>
Explanation:  “shootings” and “act” enable agent concealment, as they offer the writer the possibility of hiding the agent. compression of information in a single word or phrase leads to the depersonalization of the agent, which might in turn lead to a certain degree of ambiguity concerningwho did what to whom.

Example: وضربت أربعة انفجارات لندن قبل وقت قصير من الساعة التاسعة صباحا، ثلاثة منها في قطارات أنفاق لندن وواحد في حافلة ذات طابقين كانت تسير عبر بلومزبري. وقالت الشرطة إن عدد القتلى المؤكد هو 37 وهو في ارتفاع. وأصيب نحو 700 من الركاب والسائحين، نُقل نصفهم تقريباً إلى المستشفى بسيارات الإسعاف.  لقد كان أسوأ هجوم إرهابي في المملكة المتحدة وحمل جميع السمات المميزة لهجوم القاعدة، على غرار تفجيرات قطارات مدريد في مارس 2004. وأعلنت خلية مجهولة تابعة للقاعدة مسؤوليتها في ادعاء لا يمكن التحقق منه على موقع إسلامي على الإنترنت وأعلنت :"بريطانيا الآن تحترق بالخوف.".
Sentence: <وضربت أربعة انفجارات لندن قبل وقت قصير من الساعة التاسعة صباحا، ثلاثة منها في قطارات أنفاق لندن وواحد في حافلة ذات طابقين كانت تسير عبر بلومزبري. وقالت الشرطة إن عدد القتلى المؤكد هو 37 وهو في ارتفاع. وأصيب نحو 700 من الركاب والسائحين، نُقل نصفهم تقريباً إلى المستشفى بسيارات الإسعاف.>. Reduction phrase(s): <أربعة انفجارات>.
Explanation: “four blasts” could have been “X blasted London”. “blast” is transformed into an entity which then plays the role of the agent in the clause. It is, in otherwords, “the blasts” which is represented as carrying out the action of “hitting”, and is a representation of Muslims through passivization in British News Media Discourse.

Text: {text}
Answer: 
            '''
        },
    }
}
