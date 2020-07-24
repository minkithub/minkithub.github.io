---
title: "머신러닝 함수 및 코드 정리"
subtitle: "with tesor & keras & torch"
layout: post
author: "Minki"
# header-img: "img/seoulpark_series/seoul_park.jpg"
# header-mask: 0.6
header-style: text
catalog: true
tags:
  - Tensorflow
  - 모두의 딥러닝
  - 텐서플로우 함수
---

*잘못된 내용은 언제든지 밑의 댓글로 알려주세요!*

# 들어가기

딥러닝을 다시 공부하면서 tensorflow에서 사용되는 함수들을 참 많이 까먹었단 생각이 들었습니다. 앞으로 이 포스팅에 tensorflow 함수에 대한 내용을
지속적으로 업데이트 하겠습니다.

# tensorflow에는 어떤 함수가 있을까?

지금부터 tensorflow에는 어떤 함수가 있는지 하나하나 예제와 함께 살펴보겠습니다.

## 1. 변수 생성

tensorflow에서는 tensor라는 변수를 생성하는 다양한 함수들이 있습니다. 이 부분에서는 변수의 생성을 다뤄보겠습니다.

### 1.1 tf.constant

변하지않는 상수를 생성하는 가장 기본적인 함수입니다. 일반적인 python 문법에서는 상수라는 개념이 없지만, tensorflow에서는 학습을 위해 존재합니다.

```python
sess = tf.Session()
intro = tf.constant('hello world')
a = tf.constant(10)
print(sess.run(intro)) # 'hello world'
print(sess.run(a)) # 10
sess.run(tf.global_variables_initializer())
```

### 1.2 tf.random_normal

랜덤 가우시안 분포를 생성하는 함수입니다. 단독으로 사용되는 경우는 많지 않습니다.

```python
# 평균이 -1이고 표준편차가 4인 가우시안분포에서 랜덤샘플링하는 코드
norm = tf.random_normal([20000, 30000], seed = 1234, mean=-1, stddev=4)
sess = tf.Session()
res = sess.run(norm)
print(np.mean(res, axis = 0))
print(np.mean(res, axis = 1))
print('=================================')
print(np.std(res, axis=0))
print(np.std(res, axis=1))
print('=================================')
sess.run(tf.global_variables_initializer())
```

## *reference*
* [텐서플로우 기초정리](http://hero4earth.com/blog/learning/2018/01/15/tensor_flow_basics/)
* [텐서플로우 코리아](https://tensorflowkorea.gitbooks.io/tensorflow-kr/content/g3doc/api_docs/python/constant_op.html)