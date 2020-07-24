---
title: "추천시스템에 대하여"
subtitle: "Contents or Collaborative based filtering"
layout: post
author: "Minki"
# header-img: "img/seoulpark_series/seoul_park.jpg"
# header-mask: 0.6
header-style: text
catalog: true
tags:
  - 추천시스템
  - Recommendation System
  - Collaborative Filtering
  - Contents based Filtering
---

*잘못된 내용은 언제든지 밑의 댓글로 알려주세요!*

# 들어가기

이번 포스팅은 추천시스템 포스팅 시리즈의 첫번째 편입니다. 딥러닝에 빠삭한 서비스 기획자를 꿈꾸는 저에게 추천시스템은 꼭꼭 정복하고 싶은 분야입니다.
그래서 추천시스템은 별도의 카테고리를 만들어 포스팅을 지속적으로 게시할 예정입니다.

# 추천시스템의 종류

추천시스템은 크게 두 가지 갈래로 나눌 수 있습니다. 첫 번째 갈래는 Contents based Filtering이고 두 번째 갈래는 Collaborative Filtering입니다.
그럼 지금부터 하나하나씩 살펴보도록 하겠습니다.

## 1. Contents based filtering

먼저 Contents based filtering부터 살펴보겠습니다.

<img src="/img/recommendation/post1/contents_based_filtering.png" style="width: 800px;"/>

Contents based filtering은 한 문장으로 '비슷한 것들을 추천해주는 알고리즘'입니다. 그렇다면 어떻게 객체들끼리 비슷한지 혹은 비슷하지 않은지 파악할 수 있을까요?
비슷한 정도의 파악을 위해 contents based filtering은 데이터를 설명해주는 profile 데이터를 활용합니다. 이해를 위해 어떤 서비스의 회원가입 정보를 저희가 보고있다고 상상해봅시다.

<img src="/img/recommendation/post1/people.png" style="width: 800px;"/>

여기서 개인마다 차이는 있겠지만, 저는 개인적으로 취미, 나이, 반려동물을 고려했을 때 다음과 같이 결론을 내렸습니다.

* 여자 A와 남자 B는 서로 비슷한 취향을 가지고 있을 것이다.
* 여자 B와 남자 A는 서로 비슷한 취향을 가지고 있을 것이다.

그럼 이제 추천시스템을 만드는 일은 매우 간단합니다. 여자 A가 보거나 구매했던 컨텐츠를 남자 B에게 추천해주면되고, 이를 여자 B와 남자 A 사이에도 마찬가지로 적용하면 됩니다.
이처럼 profile데이터를 가지고 서로 비슷한 정도를 파악해 추천해주는 것을 Contents based filtering이라고 합니다.  


여기서 분류를 위해 위와 같이 유저 데이터를 사용한다면 User based recommendation이라 분류하고, 아이템 데이터를 사용한다면 Item based recommendation으로 분류합니다.

## 2. Collaborative filtering

Collaborative filtering은 Contents based filtering과 다르게 항목이 얼마나 비슷하냐가 아닌, 얼마나 비슷하게 행동을 하냐가 기준이 됩니다. 그리고 행동의 대표적인 예로는
'평점 매기기'가 있습니다.

<img src="/img/recommendation/post1/rate.png" style="width: 800px;"/>

사용자가 영화 추천앱인 왓챠를 실행시키면, 왓챠는 적극적으로 사용자에게 영화의 평점을 매겨줄것을 요청합니다. 그 이유는 사용자가 매기는 영화 평점이 추천시스템의 성능과 밀접한 관계가 있기 때문입니다.

CF_rate1        | CF_rate2
:-------------------------:|:-------------------------:
<img src="/img/recommendation/post1/CF_rate1.png" style="width: 400px;"/>  |  <img src="/img/recommendation/post1/CF_rate2.png" style="width: 400px;"/>

넷플릭스, 왓챠 등과 같이 기술력이 뛰어난 기업에서 사용하는 추천시스템 알고리즘은 알 수 없으나, 
일반적으로 유저가 평점을 매기면 추천시스템 데이터셋을 구축하기 위해 평점과 유저간 2차원 Matrix를 생성할 수 있습니다.  

그러나 위 왼쪽 그림처럼 모든 유저들이 모든 영화의 평점을 매겨주면 매우 이상적이겠지만, 현실은 오른쪽 그림처럼 평점이 비어있는 곳이 더 많은 테이블이 생성됩니다.
그렇기에 기업들은 비어있는 평점을 예측하여 사용자의 취향을 파악해야만 하고, 이때 사용하는 대표적인 알고리즘이 바로 Collaborative Filtering입니다.  

덧붙여 설명하자면 위의 오른쪽 그림처럼 값이 대부분 비어있는 매트릭스를 Sparce Matrix, 즉 희소행렬이라고 합니다. 희소행렬의 정확한 정의는 행렬의 값이 대부분
0인 행렬을 의미합니다.

<img src="/img/recommendation/post1/matrix.png" style="width: 800px;"/>

마지막으로 위 글을 하나의 그림으로 요약하면 다음과 같습니다.

<img src="/img/recommendation/post1/summary.png" style="width: 800px;"/>

일반적으로 Collaborative Filtering이 Contents based Filtering 보다 성능이 좋다고 알려져있고, Collaborative Flitering 내에서는
Implicit dataset을 활용하는 것이 Explicit dataset을 활용하는 것 보다 성능이 좋다고 알려져 있습니다.  

이번 포스팅으로 추천시스템에 대한 개괄적인 설명은 끝났습니다. 다음 포스팅부터는 알고리즘을 본격적으로 살펴보겠습니다.  

## *reference*
* https://yeomko.tistory.com/3
* https://lsjsj92.tistory.com/563?category=853217
* [towardsdatascience.com](https://towardsdatascience.com/how-to-build-from-scratch-a-content-based-movie-recommender-with-natural-language-processing-25ad400eb243)