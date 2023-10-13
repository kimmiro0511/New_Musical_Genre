# New_Musical_Genre

본 연구는 경희대학교 "데이터분석캡스톤디자인" 수업에서 진행되었습니다.
진행상황 : khuhub - http://khuhub.khu.ac.kr/2018102098/data

[CMU]: http://www.cs.cmu.edu/~ark/personas/ "Go CMU Movie Summary Corpus"


## Overview
### Needs
현재 뮤지컬은 오리지널 / 라이선스 / 창작 뮤지컬로 장르가 나뉘어져 있습니다.   
이는 로맨스 / 판타지 / 에세이 등으로 나뉘어진 소설, SF / 코미디 / 공포 / 멜로 등으로 나누어져 있으며 영화와는 달리 장르를 보고 내용을 유추하기 어렵습니다. 공식 티켓 판매처(인터파크 티켓, 예스24 공연 등)에도 역시 관련 장르 구분이 되어있지 않은 것을 볼 수 있습니다.   
장르를 모르기 때문에 작품 선택에 제한이 있으며 접근성 역시 떨어집니다. 따라서 인터넷상으로 확인할 수 있는 작품 소개와 시놉시스, 대본, 넘버(노래)를 통해 작품의 특징적인 요소로 장르를 나누어 시각적으로 표현하고자 합니다.

### Goals
뮤지컬 작품의 핵심요소를 텍스트 분석을 통해 파악하여 문학적 장르(fantasy, history, romance, social, thriller)로 구분합니다. 또한, 분류된 장르를 시각화하는 방법에 대해 연구합니다.
학습 데이터는 약 42,306개로, [CMU Movie Summary Corpus][CMU]의 Detaset을 이용합니다. 적용 데이터는 https://broadwaymusicalhome.com/shows.htm 의 홈페이지에서 크롤링한 시놉시스와 줄거리를 기반으로 합니다.


도출된 장르 근접도를 바탕으로 시각화 결과 이미지를 통해 해당 뮤지컬이 갖는 장르 근접도 정보를 한 눈에 볼 수 있습니다다. 시각화 결과 이미지를 통해 한 뮤지컬이 갖는 복합 장르적 특성을 수 있습니다. 뮤지컬 속 다양한 장르의 존재성을 알림과 동시에 매니아 층이 아닌 일반 관객들의 뮤지컬에 대한 접근성을 높일 수 있습니다. 또한 소비자들이 선호하는 장르를 바탕으로 추천 시스템 등의 서비스 분야에도 다양하게 활용될 수 있습니다.

### 활용 도구

- 데이터 크롤링 : requests, BeautifulSoup 모듈을 이용해 뮤지컬 작품설명, 줄거리 등의 텍스트 크롤링
- 자연어 처리 :  NLTK 패키지를 통한 형태소 / 명사 단위로 단어 추출 및 단어 빈도 분석, LSTN
- 장르 분류 : 의사결정나무 모델, RNN, k-nearest neighbor
- 장르 검증 : k-fold 교차 검증
- 장르 시각화 : matplot, Seaborn 패키지를 통한 시각화

## Model
- 데이터 수집 : [CMU Movie Summary Corpus][CMU]의 Detaset  42,306개, 뮤지컬 줄거리 307개
- 데이터 전처리 : Detaset 토큰화, 불용어 처리 후 정수 인코딩
- 데이터 분석 :  lstm
- 데이터 검증 : k-fold 교차검증을 활용하여 장르별 정확도를 확인
- 데이터 시각화 : 분석된 뮤지컬 데이터를 장르 단위로 시각화한다.

## 과제 수행
### 데이터 수집
[ 장르 분류 모델 학습 데이터 ] 
 모델 학습을 위한 데이터는 뮤지컬 줄거리로는 양이 충분하지 못해 뮤지컬과 줄거리가 비슷한 영화 데이 터를 활용했습니다. http://www.cs.cmu.edu/~ark/personas/에서 영화 줄거리(영어) 데이터를 42,306개 수집했 습니다.  
 
 
* 수집한 영화 줄거리 태깅 일부 : 각 영화 줄거리에는 데이터 제공 사이트에서 미리 장르가 다양하게 태깅 된 상태였습니다.  
 
 
 
 미리 태깅되어 있는 장르를 [‘romance’, ’fantasy’ ,‘thriller’, ‘drama’, ’ history’ , ‘social’ , ‘#N/A’] 같이 6개의 장르로 재분류했습니다.  



 재분류 방법은 위 표처럼 직접 모든(364개) 장르를 확인하며 위 6가지 정해놓은 장르 중 가장 유사한 장르 를 선택했습니다. 장르 분류가 애매한 경우는 ‘#N/A’으로 처리했습니다. 이러한 방법을 364개의 장르를 재 태깅했습니다.  

재태깅된 장르를 기반으로 줄거리를 6가지 장르로 분류해서 장르별 csv파일로 추출했습니다.


[ 모델 테스트 데이터 ] 
 브로드웨이 뮤지컬 줄거리를 크롤링을 통해 수집했습니다. 크롤링한 웹사이트는 https://broadwaymusicalhome.com/shows.htm 입니다. 총 307건의 줄거리를 크롤링하였습니다. (Beautifulsoup 패키지를 사용했습니다) 

### 데이터 전처리

[ NLTK 라이브러리 활용 전처리 ] 
 장르가 NULL인 줄거리를 제외하고 약 35,000개의 줄거리 데이터 전처리를 진행했습니다. 이때단어 토큰화, 불용어 제거를 위해 NLTK 자연어처리 라이브러리를 사용했습니다.  
* 전처리 소스코드 일부 (자세한 내용은 주석처리했습니다.) 
 
 
  ```python
  from tqdm import tqdm
  all_vocab = {} 
  all_sentences = []
  stop_words = set(stopwords.words('english'))
  
  for i in tqdm(allplot):
      all_sentences = word_tokenize(str(i)) # 단어 토큰화를 수행합니다.
      result = []
      for word in all_sentences: 
          word = word.lower() # 모든 단어를 소문자화하여 단어의 개수를 줄입니다.
          if word not in stop_words: # 단어 토큰화 된 결과에 대해서 불용어를 제거합니다.
              if len(word) > 2: # 단어 길이가 2이하인 경우에 대하여 추가로 단어를 제거합니다.
                  result.append(word)
                  if word not in all_vocab:
                      all_vocab[word] = 0 
                  all_vocab[word] += 1
      all_sentences.append(result) 
    
  all_vocab_sorted = sorted(all_vocab.items(), key = lambda x:x[1], reverse = True)

  all_word_to_index = {}
  i=0
  for (word, frequency) in all_vocab_sorted :
      if frequency > 1 : # 빈도수가 적은 단어는 제외한다.
          i=i+1
          all_word_to_index[word] = i
  vocab_size = 15000 #상위 15000개 단어만 사용
  words_frequency = [w for w,c in all_word_to_index.items() if c >= vocab_size + 1] # 인덱스가 15000 초과인 단어 제거
  for w in words_frequency:
      del all_word_to_index[w] # 해당 단어에 대한 인덱스 정보를 삭제  
  all_word_to_index['OOV'] = len(all_word_to_index) + 1
  ```
[ 전체 줄거리(전체 : 로맨스, 판타지, 스릴러, 역사 줄거리) 단어 인덱스 부여 ] 
 단어 사용 빈도수가 적은 단어는 제외하기 위해서 빈도순으로 인덱스를 부여했습니다. 15,000순위 미만인 단어 정보는 삭제했습니다.  
  

 
[ 학습 데이터(= RM_train+TH_train+FN_train+HS_train ) 인코딩 ]  
 총 학습 데이터는 7000개로 ‘로맨스 : 스릴러 : 판타지 : 역사 = 2 : 2 : 2 : 1’의 비율로 맞추었습니다.  
* 인덱스 부여 소스코드 일부 (자세한 내용은 주석 처리했습니다.) 
### 모델 학습

다중 분류를 위해 순환 신경망 모델을 사용했습니다.  
* LSTM 모델 구현 소스코드 (자세한 설명은 주석 처리했습니다.) 

```python
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np


M_test=Mu_encoded
M_test= np.array(M_test)
max_len = 230
X_train = pad_sequences(X_train, maxlen=max_len)
X_test = pad_sequences(X_test, maxlen=max_len)

model = Sequential()
model.add(Embedding(15002, 120))
model.add(LSTM(128))
model.add(Dense(4, activation='softmax'))

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
mc = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

X_train = np.array(X_train)
Y_train = np.array(Y_train)
X_test = np.array(X_test)
Y_test = np.array(Y_test)

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=7, batch_size=64, callbacks=[es, mc])

```
    
 뮤지컬 데이터로 테스트 결과 많은 뮤지컬 작품이 로맨스로 분류된 것을 확인하였고, 스릴러가 가장 적은 비율로 분류된것을 확인했습니다.  
[ 실제 줄거리와 비교 ] 
* 오페라의 유령 뮤지컬 줄거리의 장르를 예측한 결과 아래와 같이 로맨스로 정확하게 분류되었습니다. 오 페라의 유령같이 장르 특징이 뚜렷한 줄거리의 경우는 항상 정확한 분류가 이루어지는 것을 확인했습니다. 

## 기대효과 및 활용방안 
공식 티켓 판매처(인터파크, 예스 24공연 등)에도 뮤지컬 장르구분은 존재하지 않았습니다. 이런 사이트에서 축적된 공연 데이터를 통해 더 정확한 장르 구분을 한다면 일반인들의 뮤지컬에 대한 관심과 접근성을 높일 수 있을것으로 기대합니다. 또한, 국내 창작 뮤지컬에 로맨스 뮤지컬 뿐만 아니라 다양한 장르가 존재함 을 알릴 수 있습니다. 또한, 분류된 결과를 데이터베이스에 구축하고 웹사이트를 구현하면 뮤지컬 장르 체계화에 도움이 될 것입니다. 

## 결론 및 제언 
이진 분류의 경우(예시- 로맨스 vs 판타지)에는 테스트이렇 데이터 정확도가 80%에 이르며 높은 정확도를 보였습니다. 하지만 다중 분류(4개)의 경우에 학습한 경우 정확도가 최대 63%까지만 이르며 모델의 성능을 높이지 못했습니다. 원인으로 첫째, 학습한 줄거리 데이터의 길이가 짧아 내용이 불충분한 경우가 있습니다. 둘째, 줄거리 길이는 충분하지만 분류를 위한 특성을 추출하기 어려운 단어가 많은 경우가 있었습니다.  
 
프로젝트 진행 초반에는 한글 줄거리로 국내 뮤지컬 장르를 분류하고자 했습니다. 하지만 국내 뮤지컬 줄 거리의 대부분은 홍보용으로 내용에 대한 설명이 부실한 경우가 많아 분석에 어려움이 있었습니다. 충분한 한국어 뮤지컬 줄거리 데이터를 구하지 못해 영어 자연어처리로 주제가 변경된 부분이 아쉽습니다

###  역할 분담 
양윤지 :  영화 데이터 수집&정리, 줄거리 데이터 장르 재태깅, Train,Test데이터 시각화&패딩, LSTM모델구현, 시각화 
 
김서영 :  뮤지컬 데이터 크롤링&정리, 장르 데이터 전처리, Train,Test데이터 인코딩, LSTM모델 구현 및 학습, 테스트 데이터 학습, 뮤지컬 데이터 분류
 
시연 영상 링크 : https://youtu.be/EKjjQ0tHM4s
