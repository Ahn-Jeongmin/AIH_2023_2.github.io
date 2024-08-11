import pandas as pd
import openai

def find_most_similar_characters(data, user_mbti):
    similarities = []
    
    for character, info in data.items():
        similarity = 0
        for dimension in ['I%', 'S%', 'T%', 'J%']:
            user_value = user_mbti[dimension]
            character_value = info[dimension]
            similarity += abs(user_value - character_value)
        
        similarities.append((character, similarity))
    
    # Sort based on similarity
    similarities.sort(key=lambda x: x[1])
    # return top 3 similar character
    return similarities[:3]  


def GPT_ans(nn, cn, line, mbti):
    openai.api_key= #masked *******
    messages=[]


    query_format="Explain about the scene in the novel {0} when {1} says {2} and explain why this explains {1}'s MBTI which is {3} in 3 sentence.".format(nn,cn,line, cn, mbti)

    
        
    question_string=query_format

    messages.append({"role":"user", "content": question_string})  
        
    completion=openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
    )

    chat_response = completion.choices[0].message.content
    print(chat_response)
    messages.append({"role":"assistant", "content":chat_response})



#Importing Database    
csv_file_path = "C:\\Users\\jordi\\OneDrive\\바탕 화면\\23-2 연구\\data_set\\character_database\\character_info_db_final.csv"
df = pd.read_csv(csv_file_path)
data_dict = df.set_index('character').to_dict(orient='index')
#print(data_dict)

    

# User input
user_mbti = {}
user_mbti['I%'] = float(input("I값을 입력하세요: "))
user_mbti['S%'] = float(input("S값을 입력하세요: "))
user_mbti['T%'] = float(input("T값을 입력하세요: "))
user_mbti['J%'] = float(input("J값을 입력하세요: "))

# top 3 similar character data
similar_characters = find_most_similar_characters(data_dict, user_mbti)

if len(similar_characters) >= 1:
    character, similarity = similar_characters[0]
    character_info = data_dict[character]
    print("You are a \'", character,"\' type of person !")
    print("similarity with the character :", (similarity) , "%")
    print()
    print("<<Character Info>>")
    print(character_info['type'])
    print("-> I = ", character_info['I%']) if character_info['I%'] > 50 else print("-> E = ", 100-character_info['I%'])
    print("-> S = ", character_info['S%']) if character_info['S%'] > 50 else print("-> N = ", 100-character_info['S%'])
    print("-> T = ", character_info['T%']) if character_info['T%'] > 50 else print("-> F = ", 100-character_info['T%'])
    print("-> J = ", character_info['J%']) if character_info['J%'] > 50 else print("-> P = ", 100-character_info['J%'])

    print()
    print("You might most likely to sympathize with this 4 lines said by ",character, ":")
    print()
    print("1. ", character_info['EI_linelist1'])
    GPT_ans(character_info['novel'], character, character_info['EI_linelist1'], character_info['type1'])
    print()
    print("2. ", character_info['NS_linelist1'])
    GPT_ans(character_info['novel'], character, character_info['NS_linelist1'], character_info['type2'])
    print()
    print("3. ", character_info['TF_linelist1'])
    GPT_ans(character_info['novel'], character, character_info['TF_linelist1'], character_info['type3'])
    print()
    print("4. ", character_info['JP_linelist1'])
    GPT_ans(character_info['novel'], character, character_info['JP_linelist1'], character_info['type4'])
    print()
    print("--------------")
    print()

if len(similar_characters) >= 2:
    character, similarity = similar_characters[1]
    character_info = data_dict[character]
    print("You might also sympathize well with :", character, ", [", character_info['novel'],"] by ",character_info['author'])
    print("Similarity:", 100-similarity)
    print()
    
if len(similar_characters) >= 3:
    character, similarity = similar_characters[2]
    character_info = data_dict[character]
    print("You might also sympathize well with :", character, ", [", character_info['novel'],"] by ",character_info['author'])
    print("Similarity:", 100-similarity)


