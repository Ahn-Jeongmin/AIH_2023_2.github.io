import torch
import pandas as pd
from transformers import BertForSequenceClassification, BertTokenizer
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

#E and I classification
def EandI_classification(f_path):
    # BERT 모델 및 토크나이저 로드
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    model_path = 'oversample_ver_bert_EandI_model.pth'
    loaded_model = torch.load(model_path, map_location=torch.device('cpu'))

    # 평가 모드로 설정
    loaded_model.eval()
    df = pd.read_csv(f_path)
    total_lines = df['line'].count()
    line_values = df['line'].tolist()
    simul_num = 5
    indice_list = np.zeros(total_lines)  # 초기화된 리스트 생성
    EandI_sum = 0 

    # E와 I의 개수를 저장할 리스트 초기화
    E_counts = [0] * total_lines
    I_counts = [0] * total_lines

    for _ in range(simul_num):
        for i in range(total_lines):
            input_sentence = line_values[i]
            inputs = tokenizer(input_sentence, return_tensors="pt")
            with torch.no_grad():
                outputs = loaded_model(**inputs)
            logits = outputs.logits
            predicted_class = torch.argmax(logits, dim=1).item()
            EandI_sum += predicted_class
            indice_list[i] += predicted_class
            if predicted_class == 1:
                E_counts[i] += 1
            else:
                I_counts[i] += 1
    print(E_counts)
    print(I_counts)
    # 결과 시각화
    labels = ['E', 'I']
    counts = [sum(E_counts), sum(I_counts)]

    plt.bar(labels, counts, color=['skyblue', 'orange'])
    plt.xlabel('Personality Types')
    plt.ylabel('Count')
    plt.title('Count of E and I_The Great Gatsby')
    plt.show()

    print("E %: ", sum(E_counts)/(total_lines * simul_num)*100)
    print("I %: ", sum(I_counts)/(total_lines * simul_num)*100)
    final_val = EandI_sum / (total_lines * simul_num)

    for i in range(total_lines):
        indice_list[i] /= simul_num  # 평균값 계산

    if (final_val / simul_num) < 0.5:
        top_indices = indice_list.argsort()[-3:][::-1]  # 가장 큰 값의 인덱스 찾기
        for idx in top_indices:
            print(idx, indice_list[idx])
        return "I"
    else:
        for i in range(total_lines):
            indice_list[i] = abs(indice_list[i] - 1)
        top_indices = indice_list.argsort()[-3:][::-1]  # 가장 큰 값의 인덱스 찾기
        for idx in top_indices:
            print(idx, indice_list[idx])
        return "E"
    
  
#N and S classification
def NandS_classification(f_path):
    # BERT 모델 및 토크나이저 로드
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    model_path = 'oversample_ver_bert_NandS_model.pth'
    loaded_model = torch.load(model_path, map_location=torch.device('cpu'))

    # 평가 모드로 설정
    loaded_model.eval()
    df = pd.read_csv(f_path)
    total_lines = df['line'].count()
    line_values = df['line'].tolist()
    simul_num = 5
    indice_list = np.zeros(total_lines)  # 초기화된 리스트 생성
    NandS_sum = 0 

    N_counts = [0] * total_lines
    S_counts = [0] * total_lines

    for _ in range(simul_num):
        for i in range(total_lines):
            input_sentence = line_values[i]
            inputs = tokenizer(input_sentence, return_tensors="pt")
            with torch.no_grad():
                outputs = loaded_model(**inputs)
            logits = outputs.logits
            predicted_class = torch.argmax(logits, dim=1).item()
            NandS_sum += predicted_class
            indice_list[i] += predicted_class
            if predicted_class == 1:
                N_counts[i] += 1
            else:
                S_counts[i] += 1

    print(N_counts)
    print(S_counts)
    # 결과 시각화
    labels = ['N', 'S']
    counts = [sum(N_counts), sum(S_counts)]

    plt.bar(labels, counts, color=['skyblue', 'orange'])
    plt.xlabel('Personality Types_The Great Gatsby')
    plt.ylabel('Count')
    plt.title('Count of N and S')
    plt.show()

    print("N %: ", sum(N_counts)/(total_lines * simul_num)*100)
    print("S %: ", sum(S_counts)/(total_lines * simul_num)*100)

    final_val = NandS_sum / (total_lines * simul_num)

    for i in range(total_lines):
        indice_list[i] /= simul_num  # 평균값 계산

    if (final_val / simul_num) < 0.5:
        top_indices = indice_list.argsort()[-3:][::-1]  # 가장 큰 값의 인덱스 찾기
        for idx in top_indices:
            print(idx, indice_list[idx])
        return "S"
    else:
        for i in range(total_lines):
            indice_list[i] = abs(indice_list[i] - 1)
        top_indices = indice_list.argsort()[-3:][::-1]  # 가장 큰 값의 인덱스 찾기
        for idx in top_indices:
            print(idx, indice_list[idx])
        return "N"
    
    
#T and F classification
def TandF_classification(f_path):
    # BERT 모델 및 토크나이저 로드
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    model_path = 'bert_TandF_model.pth'
    loaded_model = torch.load(model_path, map_location=torch.device('cpu'))

    # 평가 모드로 설정
    loaded_model.eval()
    df = pd.read_csv(f_path)
    total_lines = df['line'].count()
    line_values = df['line'].tolist()
    simul_num = 5
    indice_list = np.zeros(total_lines)  # 초기화된 리스트 생성
    TandF_sum = 0 

    T_counts = [0] * total_lines
    F_counts = [0] * total_lines

    for _ in range(simul_num):
        for i in range(total_lines):
            input_sentence = line_values[i]
            inputs = tokenizer(input_sentence, return_tensors="pt")
            with torch.no_grad():
                outputs = loaded_model(**inputs)
            logits = outputs.logits
            predicted_class = torch.argmax(logits, dim=1).item()
            TandF_sum += predicted_class
            indice_list[i] += predicted_class
            if predicted_class == 1:
                T_counts[i] += 1
            else:
                F_counts[i] += 1

    print(T_counts)
    print(F_counts)
    # 결과 시각화
    labels = ['T', 'F']
    counts = [sum(T_counts), sum(F_counts)]

    plt.bar(labels, counts, color=['skyblue', 'orange'])
    plt.xlabel('Personality Types_The Great Gatsby')
    plt.ylabel('Count')
    plt.title('Count of T and F')
    plt.show()

    print("T %: ", sum(T_counts)/(total_lines * simul_num)*100)
    print("F %: ", sum(F_counts)/(total_lines * simul_num)*100)
    final_val = TandF_sum / (total_lines * simul_num)

    for i in range(total_lines):
        indice_list[i] /= simul_num  # 평균값 계산

    if (final_val / simul_num) < 0.5:
        top_indices = indice_list.argsort()[-3:][::-1]  # 가장 큰 값의 인덱스 찾기
        for idx in top_indices:
            print(idx, indice_list[idx])
        return "F"
    else:
        for i in range(total_lines):
            indice_list[i] = abs(indice_list[i] - 1)
        top_indices = indice_list.argsort()[-3:][::-1]  # 가장 큰 값의 인덱스 찾기
        for idx in top_indices:
            print(idx, indice_list[idx])
        return "T"

#J and P classification
def JandP_classification(f_path):
    # BERT 모델 및 토크나이저 로드
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    model_path = 'oversample_ver_bert_JandP_model.pth'
    loaded_model = torch.load(model_path, map_location=torch.device('cpu'))

    # 평가 모드로 설정, 20번 반복으로 아예 함수 내에 설정
    loaded_model.eval()
    df = pd.read_csv(f_path)
    total_lines = df['line'].count()
    line_values = df['line'].tolist()
    simul_num = 5
    indice_list = np.zeros(total_lines)  # 초기화된 리스트 생성
    JandP_sum = 0 

    J_counts = [0] * total_lines
    P_counts = [0] * total_lines

    for _ in range(simul_num):
        for i in range(total_lines):
            input_sentence = line_values[i]
            inputs = tokenizer(input_sentence, return_tensors="pt")
            with torch.no_grad():
                outputs = loaded_model(**inputs)
            logits = outputs.logits
            predicted_class = torch.argmax(logits, dim=1).item()
            JandP_sum += predicted_class
            indice_list[i] += predicted_class
            if predicted_class == 1:
                J_counts[i] += 1
            else:
                P_counts[i] += 1

    print(J_counts)
    print(P_counts)
    # 결과 시각화
    labels = ['J', 'P']
    counts = [sum(J_counts), sum(P_counts)]

    plt.bar(labels, counts, color=['skyblue', 'orange'])
    plt.xlabel('Personality Types_The Great Gatsby')
    plt.ylabel('Count')
    plt.title('Count of J and P')
    plt.show()

    print("J %: ", sum(J_counts)/(total_lines * simul_num)*100)
    print("P %: ", sum(P_counts)/(total_lines * simul_num)*100)
    final_val = JandP_sum / (total_lines * simul_num)

    for i in range(total_lines):
        indice_list[i] /= simul_num  # 평균값 계산

    if (final_val / simul_num) < 0.5:
        top_indices = indice_list.argsort()[-3:][::-1]  # 가장 큰 값의 인덱스 찾기
        for idx in top_indices:
            print(idx, indice_list[idx])
        return "P"
    else:
        for i in range(total_lines):
            indice_list[i] = abs(indice_list[i] - 1)
        top_indices = indice_list.argsort()[-3:][::-1]  # 가장 큰 값의 인덱스 찾기
        for idx in top_indices:
            print(idx, indice_list[idx])
        return "J"


#exe
path_list=[
    "C:\\Users\\jordi\\OneDrive\\바탕 화면\\23-2 연구\\data_set\\character_line\\The_Great_Gatsby\\TheGreatGatsby_Gatsby_INFJ.csv",
    "C:\\Users\\jordi\\OneDrive\\바탕 화면\\23-2 연구\\data_set\\character_line\\The_Great_Gatsby\\TheGreatGatsby_Daisy_ENFP.csv",
    "C:\\Users\\jordi\\OneDrive\\바탕 화면\\23-2 연구\\data_set\\character_line\\The_Great_Gatsby\\TheGreatGatsby_Nick_ISFJ.csv"
]

for i in path_list:
    f_path=i
    predicted_list=[]
    character_mbti=[]
    character_mbti.append(EandI_classification(f_path))
    character_mbti.append(NandS_classification(f_path))
    character_mbti.append(TandF_classification(f_path))
    character_mbti.append(JandP_classification(f_path))

    predicted_mbti=''.join(character_mbti)
    print(predicted_mbti)