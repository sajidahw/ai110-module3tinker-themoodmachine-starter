# Model Card: Mood Machine

This model card is for the Mood Machine project, which includes **two** versions of a mood classifier:

1. A **rule based model** implemented in `mood_analyzer.py`
2. A **machine learning model** implemented in `ml_experiments.py` using scikit learn

You may complete this model card for whichever version you used, or compare both if you explored them.

## 1. Model Overview

**Model type:**  
Describe whether you used the rule based model, the ML model, or both.  
Example: “I used the rule based model only” or “I compared both models.”
I used both, but only updated the rule based model with new posts and labels as well as logic updates. The ML model was trained on the same data but not updated with new posts.

**Intended purpose:**  
What is this model trying to do?  
Example: classify short text messages as moods like positive, negative, neutral, or mixed.
It takes a user's sentence and classifies it as positive, negative, neutral, or mixed based off of recognizing words, phrases, and emojis that are associated with each mood.

**How it works (brief):**  
For the rule based version, describe the scoring rules you created.  
Scoring rules are based on the presence of positive and negative words, negation handling, and emoji interpretation. Each post is scored based on these factors, and thresholds are set to determine the final mood classification.
For the ML version, describe how training works at a high level (no math needed).



## 2. Data

**Dataset description:**  
Summarize how many posts are in `SAMPLE_POSTS` and how you added new ones.
There were originally 6 and I added 5 more samples to the dataset.

**Labeling process:**  
Explain how you chose labels for your new examples.  
Mention any posts that were hard to label or could have multiple valid labels.
I also added new words to the positive and negative word lists. For mixed phrases which could be ambiguous or sarcasm, I did not try to label them. I did add more words to the list depending on if I felt it tends to be explicitly recognized as positive or negative. I also added more emojis to the list of positive and negative emojis.

**Important characteristics of your dataset:**  
Examples you might include:  

- Contains slang or emojis  
- Includes sarcasm  
- Some posts express mixed feelings  
- Contains short or ambiguous messages

**Possible issues with the dataset:**  
Think about imbalance, ambiguity, or missing kinds of language. Seemed to run into ambiguity with sarcasm, and certain type of slang like "ain't".

## 3. How the Rule Based Model Works (if used)

**Your scoring rules:**  
Describe the modeling choices you made.  
Examples:  

- How positive and negative words affect score  
- Negation rules you added  
- Weighted words  
- Emoji handling  
- Threshold decisions for labels

**Strengths of this approach:**  
Where does it behave predictably or reasonably well?
It works pretty well if the post is clearly positive or negative, and it can handle some negation and emojis. It also works well for posts that are short and to the point.

**Weaknesses of this approach:**  
Where does it fail?  
Examples: sarcasm, subtlety, mixed moods, unfamiliar slang.
It cannot recognize sarcasm, unfamiliar slang, or mixed moods. It also struggles with longer posts that have multiple sentences and ideas.

## 4. How the ML Model Works (if used)

**Features used:**  
Describe the representation.  
Example: “Bag of words using CountVectorizer.”

**Training data:**  
State that the model trained on `SAMPLE_POSTS` and `TRUE_LABELS`.

**Training behavior:**  
Did you observe changes in accuracy when you added more examples or changed labels?

**Strengths and weaknesses:**  
Strengths might include learning patterns automatically.  
Weaknesses might include overfitting to the training data or picking up spurious cues.

## 5. Evaluation

**How you evaluated the model:**  
Both versions can be evaluated on the labeled posts in `dataset.py`.  
Describe what accuracy you observed.
I observed that the ML version had a higher accuracy than the rule based version, but both models struggled with certain types of posts, especially those that were sarcastic or contained mixed emotions or unknown slang.

**Examples of correct predictions:**  
Provide 2 or 3 examples and explain why they were correct.
You: i ain't sad
ML model: negative

You: no cap this week has been rough
ML model: negative

**Examples of incorrect predictions:**  
Provide 2 or 3 examples and explain why the model made a mistake.  
If you used both models, show how their failures differed.
You: i ain't sad
ML model: negative

You: i absolutely love getting stuck in traffic for two hours
Model: positive

You: pretty fly weather today
Model: neutral

## 6. Limitations

Describe the most important limitations.  
Examples:  

- The dataset is small  
- The model does not generalize to longer posts  
- It cannot detect sarcasm reliably  
- It depends heavily on the words you chose or labeled

## 7. Ethical Considerations

Discuss any potential impacts of using mood detection in real applications.  
Examples: 

- Misclassifying a message expressing distress  
- Misinterpreting mood for certain language communities  
- Privacy considerations if analyzing personal messages

## 8. Ideas for Improvement

List ways to improve either model.  
Possible directions:  

- Add more labeled data  
- Use TF IDF instead of CountVectorizer  
- Add better preprocessing for emojis or slang  
- Use a small neural network or transformer model  
- Improve the rule based scoring method  
- Add a real test set instead of training accuracy only
