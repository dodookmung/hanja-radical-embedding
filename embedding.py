import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
import pandas as pd
from tqdm import tqdm
import json



model_name = 'text-embedding-ada-002'


# .env 파일 로드
load_dotenv()
# LangChain OpenAI 임베딩 객체 생성
embeddings_model = OpenAIEmbeddings(
    model=model_name,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)
print(f'embeddings_model : {embeddings_model.model}')

# CSV 파일 불러오기
df = pd.read_csv("hanja.csv")  # 파일 경로를 여기에 입력

# 'hanja'와 'radical' 컬럼만 추출
hanjas = df['hanja'].dropna().unique()
radicals = df['radical'].dropna().unique()

# 중복 제거된 한자 및 부수 리스트
unique_items = list(set(hanjas.tolist() + radicals.tolist()))



# 임베딩 결과 저장 딕셔너리
embeddings = {}

# ✅ 배치 기반 임베딩 함수
def embed_texts_batch(text_list, embeddings={}, batch_size=100):
    # 중복 요청 방지: 이미 있는 건 건너뜀
    new_texts = [text for text in text_list if text not in embeddings]
    
    for i in tqdm(range(0, len(new_texts), batch_size), desc="임베딩 중"):
        batch = new_texts[i:i+batch_size]
        try:
            # LangChain을 통한 배치 임베딩 생성
            batch_embeddings = embeddings_model.embed_documents(batch)
            
            # 결과를 딕셔너리에 저장
            for text, embedding in zip(batch, batch_embeddings):
                embeddings[text] = embedding
                
        except Exception as e:
            print(f"배치 {i // batch_size + 1} 임베딩 실패: {e}")
   
    return embeddings

# 🧠 임베딩 수행
embeddings = embed_texts_batch(unique_items, embeddings)




# 결과 저장 (optional)
with open(f"{str(embeddings_model.model)}.json", "w", encoding="utf-8") as f:
    json.dump(embeddings, f, ensure_ascii=False, indent=2)

print("✅ 임베딩 완료 및 json 저장됨.")




import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import random


def calculate_roc_for_hanja_radical_pairs(df, embeddings):
    # 긍정적 쌍(실제 한자-부수 관계) 및 부정적 쌍(무작위 쌍) 생성
    similarities = []
    y_true = []
    positive_pairs = []
    
    # 실제 한자-부수 쌍 추출 (긍정적 케이스)
    valid_rows = df.dropna(subset=['hanja', 'radical'])
    
    for _, row in valid_rows.iterrows():
        hanja = row['hanja']
        radical = row['radical']
        
        # 임베딩이 있는지 확인
        if hanja in embeddings and radical in embeddings:
            # 코사인 유사도 계산 - sklearn 함수 사용
            hanja_vector = np.array(embeddings[hanja]).reshape(1, -1)
            radical_vector = np.array(embeddings[radical]).reshape(1, -1)
            
            similarity = cosine_similarity(hanja_vector, radical_vector)[0][0]
            
            similarities.append(similarity)
            y_true.append(1)  # 실제 쌍은 1로 레이블
            positive_pairs.append((hanja, radical))
    
    # 무작위 한자-부수 쌍 생성 (부정적 케이스)
    hanjas_with_embeddings = [h for h in hanjas if h in embeddings]
    radicals_with_embeddings = [r for r in radicals if r in embeddings]
    
    random.seed(42)
    
    negative_count = 0
    max_attempts = len(positive_pairs) * 5
    attempts = 0
    
    while negative_count < len(positive_pairs) and attempts < max_attempts:
        random_hanja = random.choice(hanjas_with_embeddings)
        random_radical = random.choice(radicals_with_embeddings)
        
        # 이미 실제 쌍인 경우 건너뛰기
        if (random_hanja, random_radical) not in positive_pairs:
            hanja_vector = np.array(embeddings[random_hanja]).reshape(1, -1)
            radical_vector = np.array(embeddings[random_radical]).reshape(1, -1)
            
            similarity = cosine_similarity(hanja_vector, radical_vector)[0][0]
            
            similarities.append(similarity)
            y_true.append(0)  # 무작위 쌍은 0으로 레이블
            negative_count += 1
        
        attempts += 1
    
    # ROC 커브 및 AUC 계산
    y_scores = np.array(similarities)
    y_true = np.array(y_true)
    
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    return fpr, tpr, roc_auc, thresholds



# ROC 커브 계산
fpr, tpr, roc_auc, thresholds = calculate_roc_for_hanja_radical_pairs(df, embeddings)

# ROC 커브 시각화
plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, 
         label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Embedding similarity of Hanja-radical relation ROC curve')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

# 최적 임계값 찾기
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
print(f"최적 임계값: {optimal_threshold:.4f}")
print(f"이 임계값에서의 TPR: {tpr[optimal_idx]:.4f}, FPR: {fpr[optimal_idx]:.4f}")
# AUC 값 출력
print(f"ROC 커브의 AUC 점수: {roc_auc:.4f}")



# 결과 데이터 백업
with open(f"ROC-AUC_{str(embeddings_model.model)}.txt", "w", encoding="utf-8") as f:
    f.write(str(roc_auc))