# **틴플레이 댓글 욕설 검열 및 신고 서비스**

## **💡목차**

1. 개요
2. 데이터 수집
   1) 크롤링
   2) 데이터 세트
   3) 데이터 통합
3. 데이터 전처리
4. 사전 모델 학습
5. 사전 모델 평가
6. 댓글 작성/수정 및 신고 시 추가 학습
7. Django 프로젝트 상용화화
8. 트러블 슈팅 및 느낀점

## **📋 개요**

### 댓글 비속어어 검열 서비스  
이 서비스는 활동, 모임 홍보, 위시리스트 상세 페이지에서 댓글 작성 및 수정 시 욕설을 검열해주는 기능을 제공합니다.  

사용자가 댓글을 작성하거나 수정할 때, 입력된 내용은 `Count Vectorizer`를 사용하여 벡터화되며,  
`Multinomial Naive Bayes` 모델을 통해 텍스트가 분류됩니다.  

이 과정에서 댓글 내용이 욕설로 판별될 경우, 댓글 작성 및 수정을 차단하고 경고 메시지를 표시합니다.  

### 댓글 신고 및 처리 서비스  
이 서비스는 작성된 댓글을 다른 사용자가 신고할 수 있는 기능을 제공합니다.  

사용자가 댓글을 신고하면, 해당 댓글은 자동으로 삭제되며, 이를 통해 커뮤니티의 건전성을 유지할 수 있습니다.  

원래는 신고된 댓글을 관리자 페이지에서 관리자가 검토하여 처리해야 하지만,  
현재 신고 관리 페이지가 없기 때문에 사용자가 신고할 때 자동으로 처리가 이루어지도록 구성했습니다.  

### AI 기반 욕설 필터링 및 학습  
댓글 신고 서비스는 신고된 댓글의 내용을 `Count Vectorizer`와 `Multinomial Naive Bayes` 모델을 통해 분석하여 욕설(1)로 추가 학습합니다.  

이를 통해 모델이 지속적으로 학습하고, 더 정확하게 욕설을 필터링할 수 있도록 개선됩니다.  

## **📊 데이터 수집 (Data Collection)**

#### 1) 크롤링을 통한 수집
- 복붙

#### 2) 데이터세트를 통한 수집
- 데이터 세트 깃허브 주소: https://github.com/2runo/Curse-detection-data
- 댓글 내용과 타겟이 `|`로 구분되어 있는 데이터 세트로 read_csv 시 `sep='|'`을 사용
- <details>
    <summary>Click to see full code</summary>

        import pandas as pd

        reply_df = pd.read_csv('./datasets/dataset.txt', sep='|')
        reply_df
    
  </details>

![스크린샷 2024-05-23 154004](https://github.com/gyoungDeok-Seo/django_with_ai/assets/142222116/465922ab-02b4-4265-968d-8ba02c12ef64)

#### 3) 데이터 통합
- 복붙

## **📊 데이터 전처리 (Data Preprocessing)**

- 통합한 데이터 세트를 `Jupyter Notebook` 환경에서 `pandas` 라이브러리를 통해 불러옵니다.

- <details>
    <summary>Click to see full code</summary>
    
        import pandas as pd
    
        reply_df = pd.read_csv('./datasets/merge_comments_data.csv')
        reply_df

</details>

![스크린샷 2024-05-23 172955](https://github.com/gyoungDeok-Seo/django_with_ai/assets/142222116/e938a4e3-3465-4abc-a401-6274c8735f70)


- 댓글 내용의 비속어 예측 정확도를 향상시키기 위해, 간결한 불용어 목록을 사용하여 자연어 처리를 수행합니다.  

- 또한, 단어 사이에 특수문자를 사용하여 비속어를 숨기는 경우를 감지하기 위해 정규표현식을 통해 한글, 숫자, 알파벳(대/소문자)을  
  제외한 나머지 문자를 제거하는 함수를 정의합니다.

- <details>
    <summary>Click to see full code</summary>

        import re
        
        # 불용어 목록 선언
        korean_stopwords = set([
            '이', '그', '저', '것', '들', '의', '를', '은', '는', '에', '와', '과', '도', '으로', '까지', '부터', '다시', '번', '만', '할',  
            '한다', '그리고', '가', '이다', '다', '등', '합니다', '있습니다', '합니다', '있다', '수', '인', '여기', '저기', '거기', '의해',  
            '같은', '등', '이랑', '며', '이와', '서', '한', '그리고', '합니다', '때문에', '대로', '따라', '마다', '하나', '두', '세', '네',  
            '한', '하기', '등', '이며', '이며', '이와', '이런', '이렇게', '하지만', '때문에', '그리고', '입니다', '하지만', '그러나', '어떻게',  
            '그러면', '어떤', '그래서', '뿐만', '그런데', '더욱', '더군다나', '게다가', '하지만', '그래서', '그러므로', '그러니까', '따라서',  
            '그러나', '그리고', '이와'
        ])

        # 데이터 전처리 함수 정의
        def preprocess_text(text):
            # 특수 문자 제거
            text = re.sub(r'[^가-힣a-zA-Z0-9\s-]', '', text)
            
            # 연속된 공백 제거
            text = re.sub(r'\s+', ' ', text).strip()
            
            # 형태소 분석
            words = text.split()
            
            # 불용어 제거
            text = ' '.join([word for word in words if word not in korean_stopwords])
            return text

</details>

- 영어를 제외한 외국어, 특수문자를 제거하면서 빈 `Comment` feater를 갖고있는 행을 삭제합니다.

- <details>
    <summary>Click to see full code</summary>
    
        reply_df['Comment'] = reply_df['Comment'].apply(preprocess_text)
        reply_df['Comment'].replace('', pd.NA, inplace=True)
        reply_df.dropna(subset=['Comment'], inplace=True)
        reply_df

</details>

![스크린샷 2024-05-23 235832](https://github.com/gyoungDeok-Seo/django_with_ai/assets/142222116/fbf1b54c-c783-4d44-9d50-321895c31047)

- Target의 비중을 맞추기 위해 `value_counts`를 확인합니다.
- Target은 0(정상), 1(비속어)로 이진 분류에 해당합니다.

- <details>
    <summary>Click to see full code</summary>
    
        reply_df.Target.value_counts()

</details>

![스크린샷 2024-05-24 000040](https://github.com/gyoungDeok-Seo/django_with_ai/assets/142222116/a9e57a4e-4e21-4205-a90e-e0fd7fe4ff8c)

- 비중의 차이가 10% 이하 이기 때문에 언더샘플링을 진행합니다.

- <details>
    <summary>Click to see full code</summary>
    
        profanity = reply_df[reply_df['Target'] == 1].sample(8151, random_state=124)
        normal = reply_df[reply_df['Target'] == 0]
        reply_df = pd.concat([profanity, normal]).reset_index(drop=True)
  
</details>

## **📈 사전 모델 학습**

- 댓글 내용을 `CountVectorizer()`를 통해 백터화 한 후 `MultinomialNB()` 분류 모델을 사용해서 예측을 진행합니다.  

- `Pipeline()`을 사용하여 데이터 전처리와 모델 학습을 순차적으로 진행할 수 있도록 파이프라인을 구축합니다.

- 최종적으로, fit 메서드를 사용하여 파이프라인을 학습 데이터(X_train, y_train)로 학습시킵니다.  

- <details>
    <summary>Click to see full code</summary>
    
        from sklearn.model_selection import train_test_split
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.naive_bayes import MultinomialNB
        from sklearn.pipeline import Pipeline

        features, targets = reply_df.Comment, reply_df.Target

        X_train, X_test, y_train, y_test = train_test_split(features, targets, stratify=targets, test_size=0.2, random_state=124)

        m_nb_pipe = Pipeline([('count_vectorizer', CountVectorizer()), ('multinomial_NB', MultinomialNB())])

        m_nb_pipe.fit(X_train, y_train)

</details>

## **📈 사전 모델 평가**

- 학습한 모델을 평가하기 위한 오차 행렬을 시각화 해주는 함수를 선언합니다.

- <details>
    <summary>Click to see full code</summary>
    
        import matplotlib.pyplot as plt
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
        
        def get_evaluation(y_test, prediction, classifier=None, X_test=None):
            # 오차 행렬
            confusion = confusion_matrix(y_test, prediction)
            # 정확도
            accuracy = accuracy_score(y_test , prediction)
            # 정밀도
            precision = precision_score(y_test , prediction, average='micro')
            # 재현율
            recall = recall_score(y_test , prediction, average='micro')
            # F1 score
            f1 = f1_score(y_test, prediction, average='micro')
            
            print('오차 행렬')
            print(confusion)
            print('정확도: {0:.4f}, 정밀도: {1:.4f}, 재현율: {2:.4f}, F1: {3:.4f}'.format(accuracy, precision, recall, f1))
            print("#" * 80)
            
            if classifier is not None and  X_test is not None:
                fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12,4))
                titles_options = [("Confusion matrix", None), ("Normalized confusion matrix", "true")]
        
                for (title, normalize), ax in zip(titles_options, axes.flatten()):
                    disp = ConfusionMatrixDisplay.from_estimator(classifier, X_test, y_test, ax=ax, cmap=plt.cm.Blues, normalize=normalize)
                    disp.ax_.set_title(title)
                plt.show()
  
</details>

- 데스트 데이터(X_test)에 대한 예측을 진행하고 그 결과를 앞서 선언한 함수를 통해 시각화합니다.

- <details>
    <summary>Click to see full code</summary>
    
        prediction = m_nb_pipe.predict(X_test)
        get_evaluation(y_test, prediction, m_nb_pipe, X_test)

</details>

![스크린샷 2024-05-24 010052](https://github.com/gyoungDeok-Seo/django_with_ai/assets/142222116/f87e74ba-f8dc-4f86-ae46-ec966d78f739)

- 모든 점수가 0.71로 상당히 양호한 결과를 나타냈으나, 정상 댓글(0)을 비속어(1)로 잘못 예측하는 비율이 다소 높아 댓글 작성에 제약이 있을 수 있습니다.
- 이는 신고 처리 과정에서 추가 학습을 통해 성능을 향상시킬 수 있을 것으로 판단됩니다.

- 사전 학습이 끝난 모델을 `joblib`라이브러리를 통해 `.pkl`파일로 내보냅니다.

- <details>
    <summary>Click to see full code</summary>
    
        import pickle
        
        with open('reply_default_model.pkl', 'wb') as file:
            pickle.dump(m_nb_pipe, file)

</details>


## **📉 댓글 작성/수정 및 신고 시 추가 학습**




