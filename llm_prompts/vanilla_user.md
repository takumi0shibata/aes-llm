[Prompt]
{{prompt}}
(end of [Prompt])

[Note]
I have made an effort to remove personally identifying information from the essays using the Named Entity Recognizer (NER). The relevant entities are identified in the text and then replaced with a string such as "{PERSON}", "{ORGANIZATION}", "{LOCATION}", "{DATE}", "{TIME}", "{MONEY}", "{PERCENT}”, “{CAPS}” (any capitalized word) and “{NUM}” (any digits). Please do not penalize the essay because of the anonymizations.
(end of [Note])

[Essay]
{{essay}}
(end of [Essay])

Strictly follow the format below to give your answer. Other formats are NOT allowed.
Evaluation: <evaluation>insert evaluation here</evaluation>
Score: <score>insert score ({{minimum score value}} to {{maximum score value}}) here</score>