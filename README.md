# New_Musical_Genre

본 연구는 경희대학교 "데이터분석캡스톤디자인" 수업에서 진행되었습니다.

[CMU]: http://www.cs.cmu.edu/~ark/personas/ "Go CMU Movie Summary Corpus"


## Overview
### Needs
현재 뮤지컬은 오리지널 / 라이선스 / 창작 뮤지컬로 장르가 나뉘어져 있습니다.   
이는 로맨스 / 판타지 / 에세이 등으로 나뉘어진 소설, SF / 코미디 / 공포 / 멜로 등으로 나누어져 있으며 영화와는 달리 장르를 보고 내용을 유추하기 어렵습니다. 공식 티켓 판매처(인터파크 티켓, 예스24 공연 등)에도 역시 관련 장르 구분이 되어있지 않은 것을 볼 수 있습니다.   
장르를 모르기 때문에 작품 선택에 제한이 있으며 접근성 역시 떨어집니다. 따라서 인터넷상으로 확인할 수 있는 작품 소개와 시놉시스, 대본, 넘버(노래)를 통해 작품의 특징적인 요소로 장르를 나누어 시각적으로 표현하고자 합니다.

### Goals
뮤지컬 작품의 핵심요소를 텍스트 분석을 통해 파악하여 문학적 장르(fantasy, history, romance, social, thriller)로 구분한다. 또한, 분류된 장르를 시각화하는 방법에 대해 연구한다.
학습 데이터는 약 42,306개로, [CMU Movie Summary Corpus][CMU]의 Detaset을 이용합니다. 적용 데이터는 https://broadwaymusicalhome.com/shows.htm 의 홈페이지에서 크롤링한 시놉시스와 줄거리를 기반으로 합니다.


도출된 장르 근접도를 바탕으로 시각화 결과 이미지를 통해 해당 뮤지컬이 갖는 장르 근접도 정보를 한 눈에 볼 수 있다. 시각화 결과 이미지를 통해 한 뮤지컬이 갖는 복합 장르적 특성을 수 있습니다. 뮤지컬 속 다양한 장르의 존재성을 알림과 동시에 매니아 층이 아닌 일반 관객들의 뮤지컬에 대한 접근성을 높일 수 있습니다. 또한 소비자들이 선호하는 장르를 바탕으로 추천 시스템 등의 서비스 분야에도 다양하게 활용될 수 있습니다.

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

- ###데이터 
  ```from tqdm import tqdm
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
### 영화 데이터
- **영화 데이터 크롤링**<br>
    7가지의 감정을 잘 예측하였는지 확인하기 위한 test set으로서 영화 리뷰 데이터를 활용하였다. 
    그 이유는 영화 리뷰는 영화를 본 후 감상평을 적는 것이기 때문에 사람들의 7가지 감정이 잘 녹아들어 있을 거라 판단했기 때문이다.<br>
 
