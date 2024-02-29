import os
import openai

openai.api_key="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"
messages=[]

query_format="Explain about the scene in the novel {0} when {1} says {2} and explain why this explains {1}'s MBTI which is {3} in 3 sentence in one paragraph without '\n'.".format("The Picture of Dorian Gray","Basil","Well, perhaps you are right. And now good-by, Dorian. You have been the one person in my life of whom I have been really fond. I don't suppose I shall often see you again. You don't know what it cost me to tell you all that I have told you.", "I")

question_string=query_format

messages.append({"role":"user", "content": question_string})  
        
completion=openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=messages
)

chat_response = completion.choices[0].message.content
print(chat_response)
messages.append({"role":"assistant", "content":chat_response})

