# WASSA-2022-Empathy-detection-and-Emotion-Classification

The broad goal of this task was to model an
empathy score, a distress score, and the type
of emotion associated with the person who had
reacted to the essay written in response to a
newspaper article.   
We have used the RoBERTa
model for training, and on top of it, five layers are added to finetune the transformer.  
We
also use a few machine learning techniques to
augment and upsample the data.  
Our system
achieves a Pearson Correlation Coefficient of
0.488 on Task 1 (Average of Empathy - 0.470
and Distress - 0.506) and Macro F1-score of
0.531 on Task 2.
