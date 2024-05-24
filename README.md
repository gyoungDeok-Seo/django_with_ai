# **틴플레이 댓글 욕설 검열 및 신고 서비스**

## **💡목차**

1. 데이터 수집
   1) 크롤링
   2) 데이터 세트
   3) 데이터 통합
2. 데이터 전처리
3. 사전 모델 학습
4. 사전 모델 평가
5. 댓글 작성/수정 및 신고 시 추가 학습
6. 트러블 슈팅 및 느낀점

## **📊 데이터 수집 (Data Collection)**

#### 1) 크롤링을 통한 수집
- 크롤링을 통하여 댓글을 수집하고 비속어가 포함된 댓글과 포함되지 않은 댓글의 비중을 맞춰주기 위해 전체 댓글 데이터의 45%를 임의로 욕설을 추가합니다.

- <details>
    <summary>크롤링 로직</summary>

    ```
    if __name__ == '__main__':
        warnings.filterwarnings('ignore')

        if os.path.exists('comments.csv'):
            existing_df = pd.read_csv('comments.csv', encoding='utf-8-sig', index_col=0)
            comment_final = existing_df['Comment'].tolist()
        else:
            existing_df = pd.DataFrame()
            comment_final = []

        driver = webdriver.Chrome()
        driver.get("https://www.youtube.com/watch?v=t7w3k3pjZY4")
        driver.implicitly_wait(3)

        time.sleep(1.5)

        driver.execute_script("window.scrollTo(0, 800)")
        time.sleep(3)

        last_height = driver.execute_script("return document.documentElement.scrollHeight")

        while True:
            driver.execute_script("window.scrollTo(0, document.documentElement.scrollHeight);")
            time.sleep(2)

            new_height = driver.execute_script("return document.documentElement.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height

        time.sleep(1.5)

        html = driver.page_source
        soup = BeautifulSoup(html, 'html.parser')

        comment_list = soup.select("span.yt-core-attributed-string")

        for i in range(len(comment_list)):
            temp_comment = comment_list[i].text.replace('"', '')
            temp_comment = temp_comment.replace('\n', ' ')
            temp_comment = temp_comment.strip()
            comment_final.append(temp_comment)
        new_df = pd.DataFrame({'Comment': comment_final})

        combined_df = pd.concat([existing_df, new_df]).drop_duplicates().reset_index(drop=True)

        combined_df.to_csv('qweqwe.csv', index=True, encoding='utf-8-sig')

        driver.quit()
    ```

</details>

- <details>
    <summary>임의로 욕설을 추가하는 코드</summary>

    ```
    profanity_list = ['욕설 리스트']

    def add_random_profanity(comment):
        new_comment = ''
        for word in comment.split():
            new_comment += word + ' '
            if random.random() < 0.4:
                profanity = random.choice(profanity_list)
                if random.random() < 0.5:
                    new_comment += profanity + ' '
                else:
                    new_comment += profanity
        return new_comment.strip()

    co_df = pd.read_csv('comments.csv')

    co_df['Comment'] = co_df['Comment'].apply(add_random_profanity)
    ```

</details> 

- <details>
    <summary>추가한 욕설이 있으면 Target 칼럼에 1 없으면 0 값 넣어주기</summary>

    ```
        profanity_list = ['욕설 리스트']

        profanity_list = [word.replace(' ', '') for word in profanity_list]

        co_df['Profanity'] = co_df['Comment'].apply(lambda x: 1 if any(word in x for word in profanity_list) else 0)
    ```

</details> 

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
- 수집한 두 데이터 세트를 합치고 결과를 csv파일로 내보냅니다.

 - <details>
    <summary>두 데이터 세트를 합치는 코드</summary>

    ```
        df_combined = pd.concat([co_df, bw_df], ignore_index=True)
    ```

 </details>

- <details>
    <summary>결과를 CSV 파일로 내보내는 코드</summary>

    ```
        df_combined.to_csv('merge_comments_data.csv', index=False, encoding='utf-8-sig')
    ```

 </details>

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

### **댓글 작성/수정 추가 학습**

- 활동, 모임 홍보, 위시리스트의 각 댓글을 작성/수정 시 모두 같은 로직으로 추가 학습을 진행합니다.

- 해당 댓글의 내용을 통해 예측하고 그 결과가 욕설일 경우 불러올 파이프라인 모델의 `CountVectorizer()`에 전달하여 벡터화 합니다.

- 백터화 된 값을 파이프라인 모델의 `MultinomialNB()`에 전달하여 `partial_fit()`을 통해 추가 학습을 진행합니다.

- - 정답 Target(y)은 비속어(1)로 설정합니다.

- 먼저 같은 코드가 반복되지 않도록 로직을 모듈화하여 했습니다. 또한, 예측 결과가 비속어 일 경우 훈련용 데이터 테이블에 insert되도록 했습니다.

- <details>
    <summary>Click to see full code</summary>
    
      import os
      from pathlib import Path
      import joblib
      from ai.models import ReplyAi
      
      
      def check_the_comments(reply_content):
          result = 'success'
          
          model_file_path = os.path.join(Path(__file__).resolve().parent.parent.parent.parent, 'ai/ai/reply_default_model.pkl')
          model = joblib.load(model_file_path)
          X_train = [reply_content]
          prediction = model.predict(X_train)
      
          if prediction[0] == 1:
              # 추가 fit
              transformed_X_train = model.named_steps['count_vectorizer'].transform(X_train)
              model.named_steps['multinomial_NB'].partial_fit(transformed_X_train, prediction)
              joblib.dump(model, model_file_path)
      
              # insert
              ReplyAi.objects.create(comment=X_train[0], target=prediction[0])
              result = 'profanity'
      
          return result
  
</details>

- 3개의 페이지에 대한 post 요청(댓글 작성)에 응답하는 ActivityReplyAPI, ClubPrPostReplyAPI, ReplyWriteAPI의 post() 메소드 중, 추가 학습 관련 코드입니다.

- <details>
    <summary>Click to see full ActivityReplyAPI code</summary>
    
      def post(self, request):
         data = request.data
         data = {
            'reply_content': data['reply_content'],
            'activity_id': data['activity_id'],
            'member_id': data['member_id']
         }
         
         result = check_the_comments(data['reply_content'])
         
         if result == 'profanity':
            return Response(result)
         
         activity_reply = ActivityReply.objects.create(**data)

         ....

         return Response("success")

</details>

- <details>
    <summary>Click to see full ClubPrPostReplyAPI code</summary>
    
      def post(self, request):
         data = request.data
         
         data = {
            'reply_content': data['reply_content'],
            'club_post_id': data['club_post_id'],
            'member_id': request.session['member']['id']
         }
         
         result = check_the_comments(data['reply_content'])
         
         if result == 'profanity':
            return Response(result)
         
         post_reply = ClubPostReply.objects.create(**data)

         ....

         return Response("success")

</details>

- <details>
    <summary>Click to see full ReplyWriteAPI의 code</summary>
    
      def post(self, request):
         data = request.data
         data = {
            'reply_content': data['reply_content'],
            'wishlist_id': data['wishlist_id'],
            'member_id': request.session['member']['id']
         }
         
         result = check_the_comments(data['reply_content'])
         
         if result == 'profanity':
            return Response(result)
         
         WishlistReply.objects.create(**data)
         
         return Response('success')

</details>

- 3개의 페이지에 대한 patch 요청(댓글 수정)에 응답하는 ActivityReplyAPI, ClubPrPostReplyAPI, ReplyActionAPI의 patch() 메소드 중, 추가 학습 관련 코드입니다.

- <details>
    <summary>Click to see full ActivityReplyAPI code</summary>
    
      def patch(self, request):
         activity_id = request.data['activity_id']
         member_id = request.data['member_id']
         reply_content = request.data['reply_content']
         id = request.data['id']
         
         result = check_the_comments(reply_content)
         
         if result == 'profanity':
            return Response(result)
         
         activity_reply = ActivityReply.enabled_objects.get(id=id, activity_id=activity_id, member_id=member_id)
         
         activity_reply.reply_content = reply_content
         activity_reply.updated_date = timezone.now()
         activity_reply.save(update_fields=['reply_content', 'updated_date'])
         
         return Response("success")
  
</details>

- <details>
    <summary>Click to see full ClubPrPostReplyAPI code</summary>
    
      def patch(self, request):
         data = request.data
         reply_content = data['reply_content']
         reply_id = data['id']
         
         result = check_the_comments(reply_content)
         
         if result == 'profanity':
            return Response(result)
         
         # 전달 받은 댓글 id를 통해 수정할 댓글 조회
         club_post_reply = ClubPostReply.enabled_objects.get(id=reply_id)
         club_post_reply.reply_content = reply_content
         club_post_reply.updated_date = timezone.now()
         club_post_reply.save(update_fields=['reply_content', 'updated_date'])
         
         return Response("success")

</details>

- <details>
    <summary>Click to see full ReplyActionAPI의 code</summary>
    
      def patch(self, request, reply_id):
         data = request.data
         reply_content = data['reply_content']
         
         result = check_the_comments(data['reply_content'])
         
         if result == 'profanity':
            return Response(result)
         
         updated_date = timezone.now()
         
         reply = WishlistReply.objects.get(id=reply_id)
         reply.reply_content = reply_content
         reply.updated_date = updated_date
         
         reply.save(update_fields=['reply_content', 'updated_date'])
         
         return Response(reply_content)

</details>

- 욕설일 경우 alert을 통해 경고문을 보여줍니다.

![스크린샷 2024-05-24 020419](https://github.com/gyoungDeok-Seo/django_with_ai/assets/142222116/9c2756bd-55af-443c-ba05-221f63f1e9b4)

![스크린샷 2024-05-24 020509](https://github.com/gyoungDeok-Seo/django_with_ai/assets/142222116/15936155-c1d0-4ecc-b2bc-2ae0ae91aa9b)


### **댓글 신고 추가 학습**

- 활동, 모임 홍보, 위시리스트의 각 댓글을 신고 시 추가 학습을 진행합니다.

- 기존 데이터의 삭제를 위해 댓글의 id와 어떤 글의 댓글(reply_type)인지 request로 받고 그에 맞는 클래스에서 객체화 합니다.

- 해당 댓글의 내용을 불러온 파이프라인 모델의 `CountVectorizer()`에 전달하여 벡터화 합니다.

- 백터화 된 값을 파이프라인 모델의 `MultinomialNB()`에 전달하여 `partial_fit()`을 통해 추가 학습을 진행합니다.

- 이때 정답 Target(y)은 비속어(1)로 설정합니다.

- 훈련 테이블에 정보를 insert하고 객체를 통해 기존 댓글 테이블에서 데이터를 삭제합니다.

- 3개의 페이지에 대한 post 요청(댓글 신고)에 응답하는 ReportReplyAPI의 post() 메소드 중, 추가 학습 관련 코드입니다.

- <details>
    <summary>Click to see full ReportReplyAPI code</summary>
    
      def post(self, request):
         reply_id = request.data['reply_id']
         reply_type = request.data['reply_type']
         reply = None
         
         if reply_type == 'activity':
            reply = ActivityReply.objects.get(id=reply_id)
         
         elif reply_type == 'club_post':
            reply = ClubPostReply.objects.get(id=reply_id)
         
         else:
            reply = WishlistReply.objects.get(id=reply_id)
         
         # 모델소환
         model_file_path = os.path.join(Path(__file__).resolve().parent, 'ai/ai/reply_default_model.pkl')
         model = joblib.load(model_file_path)
         X_train = [reply.reply_content]
         
         # 추가 fit
         transformed_X_train = model.named_steps['count_vectorizer'].transform(X_train)
         model.named_steps['multinomial_NB'].partial_fit(transformed_X_train, [1])
         joblib.dump(model, model_file_path)
         
         # insert
         ReplyAi.objects.create(comment=X_train[0], target=1)
         reply.delete()
         
         return Response("profanity")

</details>

![스크린샷 2024-05-24 023009](https://github.com/gyoungDeok-Seo/django_with_ai/assets/142222116/cd82a118-0c70-460d-ac8e-a315443a75fd)

![스크린샷 2024-05-24 023109](https://github.com/gyoungDeok-Seo/django_with_ai/assets/142222116/6e3842db-ebaf-41cd-9704-f76d1c3e5892)


## **📉 트러블 슈팅 및 느낀점**

### 1) 사전 모델 평가에서 너무 낮은 점수

가. 문제 발생

- 목표를 평가 점수 0.7이상으로 선정하고 수집한 약 5,000개의 데이터를 가지고 훈련 후 평가를 실시 했을 때 0.62로 낮은 점수가 나오는 문제가 발생했습니다.

나. 원인 추론

- 데이터 전처리 과정을 실시하지 않아 발생한 문제라고 판단하여 정규식을 통한 특수문자 및 영어를 제외한 외국어를 제거하고 훈련을 실시했는데 오히려 0.58로 점수가 더 떨어지는 것을 확인했습니다.
- 연속되는 공백이 존제하는 데이터를 식별하여 이에 대한 공백 축소도 진행했으나 0.55로 평가 점수가 더 떨어지는 것을 확인했습니다. 
- 불용어(의미를 전달하지 않는 단어들을 가리키는 용어)를 제거하였음에도 동일한 점수가 나오는 것을 확인하고 전처리의 문제가 아니라고 판단했습니다.

다. 해결 방안

- 다른 팀의 댓글 서비스 담당자와 비교했을 때 똑같은 문제가 발생하는 것을 확인했고 데이터 수의 문제라 생각하고,  
  3명의 데이터 세트를 통합하여 데이터 수를 늘려서 평가 점수를 확인했을 때 0.68로 더 높게 나오는걸 확인할 수 있었습니다.
- 목표로 선정한 0.7까지 높이기 위해 원인 추론 과정에서 했던 전처리 과정을 추가해서 진행했습니다.

라. 결과 확인

- 평가 점수는 0.71로 확실히 성능이 향상되는 것을 확인할 수 있었습니다.

### 2) 느낀점

- **데이터 수집의 어려문과 중요성**: 처음 비속어 검열 및 신고 서비스를 담당하게 되었을 때,  
  데이터 수집부터 활용까지 간단하게 끝낼 수 있을 것이라 생각했습니다.  
  그러나 실제 데이터 수집 과정에서 비속어가 포함된 댓글을 찾는 것이 예상보다 어려웠습니다.  
  이는 대부분의 웹사이트는 이미 댓글 필터링 기술을 적용하고 있기 때문입니다.  
  유명 커뮤니티의 경우 댓글의 수위가 너무 강하거나 비속어보다는 비난이나 감정적인 내용이 많아 부적절하다고 판단하여,  
  결국 다른 사람이 만들어 놓은 데이터셋을 이용했습니다. 이를 통해 데이터 수집이 얼마나 어렵고 중요한지 확실히 경험할 수 있었습니다.

- **모델 성능과 데이터의 관계**: 수업에서는 평가 점수가 낮을 때 전처리나 모델을 바꾸는 방식으로 성능을 높이는 방법을 주로 배웠습니다.  
  그러나 이번 프로젝트에서 배운 내용을 중점으로 진행했지만 성능이 오히려 떨어지는 것을 보고,  
  데이터의 양과 질이 성능에 큰 영향을 미친다는 것을 배울 수 있었습니다.  
  특히, 비속어와 정상 댓글을 구분하는 데 있어 데이터의 다양성과 대표성이 중요함을 깨닫게 되었습니다.
