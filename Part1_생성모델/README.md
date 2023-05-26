# PART1 생성모델

### 1. VAE VS GAN

### VAE vs GAN

<img width="1622" alt="1" src="https://github.com/junyong1111/ComputerVision/assets/79856225/e3ea9026-09a3-46c7-8b27-5f53025336d8">

<img width="1623" alt="2" src="https://github.com/junyong1111/ComputerVision/assets/79856225/b150f145-4d8f-43f9-b910-9fc571989b16">

VAEs(Variational Autoencoders)

- reale data에 데이터를 압축하여 핵심만을 학습하여 그 핵심으로 다시 real data를 예측
    - 사람이 암기과목을 학습할 때와 비슷함
- 학습  부담이 적다
- 창의성은 없다
- 학습이 완료된 후 decoder부분만을 사용

GAN(generative Adversarial Networks)

- 생성 모델이 random noise로부터 data를 생성한 후 판별 모델이 real data와 생성 data를 비교
- 학습 부담이 크다(학습이 잘 안될 수 있음)
- real data뿐 아니라 창의적이 data가 나올 수 있음
- 학습이 완료된 후 generative 부분만을 사용

### **머신러닝 기초**

- 데이터로부터 학습능력을 가진 알고리즘
- **E : 데이터 셋, T : 기능, P : 가치측정(비용측정)**

**Task의 종류**

- **회귀**
    - 어떠한 함수를 만들었을 때 반환값이 실수(float)

**분류**

- 어떠한 함수를 만들었을 때 반환값이 범주형(int)

학습의 종류

- 지도학습
    - 입력과 정답이 한 쌍으로 이루어져 있음
    - $y= f(x) == P(y |x )$
- 비지도학습
    - 입력만 주고 정답은 주지 않음
    - $P(X)$
- semi-supervised learning

**머신러닝의 목표**

**머신러닝은 손실값을 최소로 하는 세타를 찾는 문제**

→ 즉 최솟값을 유발하는 입력값을 찾는 문제


### 2. 확률, 정보 이론

### Discrete V ariables and Probability Mass Function

**Probability mass function (PMF)**

- P(x = x) → P(x) : x가 일어날 확률

ex) P(x = 3) : P를 주사위를 던지는 행위라고 했을 때 주사위를 던져서 3이 나올 확률

- 0 ≤ P(x = x) ≤ 1  : 확률값은 0~1 사이에 존재
- Sum P(x) = 1 : 모든 확률값을 더하면 1

**Joint Probability(결합 확률)**

- 2개 이상의 사건이 동시에 일어나는 확률이며 그리고 라고 해석하면 된다.
- P(x = x, y= y) : y가 일어나고 x가 일어날 확률

ex)  P(x = x, y = y)

- P :  주사위와 동전을 던지는 행위
- x : 주사위를 던졌을 때 특정 x값이 나올 확률
- y : 동전을 던졌을 때 앞면 또는 뒷면이 나올 확률

주사위와 동전을 던졌을 때 동전의 앞면 또는 뒷면이 나오고 주사위의 특정값이 동시에 나올 확률

|  | x = 1  | x = 2 | x = 3  | x = 4 | x = 5 | x = 6 |
| --- | --- | --- | --- | --- | --- | --- |
| y = 앞면 | 1/12 | 1/12 | 1/12 | 1/12 | 1/12 | 1/12 |
| y = 뒷면 | 1/12 | 1/12 | 1/12 | 1/12 | 1/12 | 1/12 |

**Marginal Probability(주변 확률)**

- **여러 사건이 동시에 일어날 수 있을 때(결합 확률), 하나의 사건이 일어날 확률 분포**
- 하나의 변수에 대한 확률 분포를 나타내는 것이다. 예를 들어, 날씨와 판매량을 고려하는 경우, 날씨와 판매량이라는 두 개의 변수가 있다. 그러나, 이 중에서 날씨가 어떻든, 판매량의 분포를 알고 싶은 경우, 이때 날씨를 고려하지 않고 판매량의 분포를 따로 계산하는 것이 주변 확률이며 자신이 알고 싶은 쪽으로 나머지 확률분포를 밀어 버린다고 생각하면 된다.

**주사위에 대한 Marginal probability**

| x = 1  | x = 2 | x = 3  | x = 4 | x = 5 | x = 6 |
| --- | --- | --- | --- | --- | --- |
| 2/12 | 2/12 | 2/12 | 2/12 | 2/12 | 2/12 |

**Conditional Probability(조건부 확률)**

- **어떤 조건이 주어진 상태에서 특정 사건이 발생할 확률**
- P(y = y | x = x) = P(x = x, y = y) / P(x = x)
- **조건부 확률 = 결합확률 / 주변 확률**

|  | x = 1  | x = 2 | x = 3  | x = 4 | x = 5 | x = 6 |
| --- | --- | --- | --- | --- | --- | --- |
| y = 앞면 | 1/12 | 1/12 | 1/12 | 1/12 | 1/12 | 1/12 |
| y = 뒷면 | 1/12 | 1/12 | 1/12 | 1/12 | 1/12 | 1/12 |

| p(y = y | x = 3) |
| --- |
| (1/12) / (1/6) =1 / 2 |

**The Chain Rule of Conditional Probability(조건부 확률의 연쇄 법칙)**

**결합확률 = 조건부확률 * 주변확률**

- P(a, b, c) = P(a | b, c)p(b,c)
    - P(b,c) = P(b | c) * P(c)
        
        → P(a, b, c) = P (a | b, c) * P(b |c) * P(c)
        

**조건부확률 = 결합확률 / 주변확률**

- P(a, b | c) = P(a, b, c) / P(c)
    - P(a, b, c) = P(a, b, | c) * P(c)
        
        → P(a, b, c) = P(a, b | c) * P(c) = P (a | b, c) * P(b |c) * P(c)
        

**Independence and Conditional Incdependence**

**사건의 독립**

- 어떤 사건 A,B에 대해 두 사건이 독립(Independent)이면 두 사건은 서로 쪼개질 수 있다.
- P(B | A) = P(B), P(A | B) = P(A)
- 조건으로 주어진 사건 A(B)가 다른 사건 B(A)에 아무런 영향을 주지 못한다.
- 조건부확률 정의에 따라 P(A | B) = P(A ^ B) / P(B) = P(A)
- **P(A ^ B) = P(A) * P (B) 독립된 2사건의 결합확률은 두 사건의 곱과 같다.**

**조건부 독립**

- 두 사건 A와 B가 세 번째 사건 C의 조건을 알고 있을 때 서로 독립적이라는 확률적 개념
- C 사건이 주어졌을 때, A 사건과 B 사건이 서로 영향을 주지 않는다면, A와 B는 C에 대해 조건부 독립
- P(A∩B|C) = P(A|C) * P(B|C)

여기서 P(A∩B|C)는 C 사건이 주어졌을 때 A와 B가 모두 발생할 확률, P(A|C)는 C 사건이 주어졌을 때 A가 발생할 조건부 확률, P(B|C)는 C 사건이 주어졌을 때 B가 발생할 조건부 확률을 나타낸다. 따라서 이러한 식이 성립한다면 A와 B는 C에 대해 조건부 독립이라고 할 수 있다.

**Expectation, Variance, and Covariance(기댓값, 분산, 공분산)**

**Expectation(기댓값)**

- **확률 변수의 평균값을 의미**. 즉, 확률 변수가 가질 수 있는 모든 값에 대한 확률의 가중 평균을 계산
    
    <img width="487" alt="1" src="https://github.com/junyong1111/ComputerVision/assets/79856225/803ad265-fd53-4538-b1ca-8b66b64cc310">
    

어떠한 x를 확률모델 P에서 주사위를 던져 뽑는다 → 그 확률을 input으로 어떠한 함수 f(x)에 대한 기댓값

**주사위를 던져서 부루마블에서 어떤 기능을 하는지에 대한 기댓값**

어떠한 확률에 따라 작동하는 기능에 대한 기댓값은 그 확률에 대한 확률함수 * 그확률에 대한 기능의 총 합이다?!

**Variance(분산)**

- 확률 변수의 **분포가 얼마나 넓게 퍼져 있는지를 나타내는 값**으로, 확률 변수의 값이 **기댓값에서 얼마나 멀리 떨어져 있는지를 나타내는 값**.

**Covariance(공분산)**

- **두 확률 변수의 상관 관계를 나타내는 값**으로, 두 확률 변수가 서로 어떤 방향으로 변화하는지를 나타내는 값이다. 두 확률 변수가 양의 값을 가질 때는 같은 방향으로 변화하는 것을, 음의 값을 가질 때는 서로 반대 방향으로 변화하는 것을 나타냄

**Bayes’ Rule(베이지안 룰)**

**—# P(x, y) = P(x | y) * p(y) = P(y | x) * p(x)** 

- **결합확률 = 조건부확률 * 주변확률**

[https:www.youtube.com/watch?v=R13BD8qKeTg](http://www.youtube.com/watch?v=R13BD8qKeTg)

P(x | y) = P(y | x) * P(x) / P(y)

**Information Theory(정보 이론)**

**정보량 I(x)**

**I(x) = - logP(x) : 정보의 가치는 확률이 적을수록 정보량이 높다.**

<img width="427" alt="2" src="https://github.com/junyong1111/ComputerVision/assets/79856225/92d3f133-e52c-4bdb-89c6-23f121485c29">

**정보량에 대한 기댓값은 = 그 정보가 일어날 확률의 -log를 취한것과 같다**

<img width="1567" alt="3" src="https://github.com/junyong1111/ComputerVision/assets/79856225/a2a17f27-f47e-4225-8d34-fc099a23e3b0">

- **H는 엔트로피(entropy)를 나타냅니다. 엔트로피는 정보 이론에서 불확실성의 정도를 나타내는 지표**

기댓값에서 정의했던것과 마찬가지로 P라는 확률함수(주사위를 던지는 함수)에서 특정한 x값이 나와 그 x값을 input으로 동작하는 f(x)에 대한 기댓값은 —> Sum → P(x) * F(x) : 각각의 x값에 대한 확률 값 * 그 확률값에 대한 기능의 총합이다.

**Kullback-Leibler (KL) divergence(두 분포 사이의 차이를 구하는 방법)**

<img width="1060" alt="4" src="https://github.com/junyong1111/ComputerVision/assets/79856225/a8608b2b-6439-4cc3-b44d-69d30d01d972">
P = 실제 데이터셋

Q = 모델

- P를 기준으로 Xdata를 받을 때 P(x)와 Q(x)의 차이

실제 고양이 그림셋 P중 하나의 고양이 그림 x를 뽑았을 때 분포와 생성한 모델 Q가 고양이 그림 x를 Inputd으로 생성한 모델의 분포가 비슷하게 만드는 것이 목표

**Cross-entropy**

<img width="1578" alt="5" src="https://github.com/junyong1111/ComputerVision/assets/79856225/4c45a9bf-bf88-4d4d-a2ae-2deeaec2a5df">

크로스 엔트로피란 2개의 다른 분포에 대한 차이를 비교할 때 사용이 된다.

- 예측값의 확률 분포: Q(y)
- 실제값의 확률 분포: P(y)

이때, Q(y)는 모델이 예측한 클래스가 y일 확률이며, P(y)는 실제 클래스가 y일 확률이다.

**H(P, Q) = - Σ P(y) * log(Q(y))**

**실제 클래스가 고양이 인 데이터를  실제 고양이 데이터셋의 확률 분포에서 뽑아서 그 이미지를 다시 모델에 넣었을 때 생성되는 고양이 이미지 분포에 대한 기댓값**

크로스 엔트로피는 실제값과 예측값의 차이가 클수록 더 큰 값을 가지며, 두 분포가 동일할 경우에는 최소값을 가집니다. 따라서, 모델의 예측값과 실제값이 일치할수록 크로스 엔트로피는 작아지며, 일치하지 않을수록 커집니다.

**KL and Cross-entropy의 관계**

<img width="537" alt="6" src="https://github.com/junyong1111/ComputerVision/assets/79856225/a0437883-1ee5-40c7-b7f3-c44aee181063">

- **cross-entropy는 KL divergence와 entropy의 합으로 표현될 수 있습니다**
- **크로스엔트로피 = P에대한 엔트로피(불확실성) + P,Q 두 분포 사이의 차이**
- H(P) : dataset으로 정해지기 때문에 변수가 아니다.
- 따라서 H(P, Q) = Dkl(P || Q)와 같다.


### 3. Maximum Likelihood Estimation, AutoEncoder

<img width="1507" alt="1" src="https://github.com/junyong1111/ComputerVision/assets/79856225/e9994f16-22a6-4d3b-86ed-b6586fe99ad4">

**이세상 모든 사람의 얼굴을 파란색 원이라고 본다면 우리가 모델에 사용하는 샘플 데이터는 그 중 일부이다.**

<img width="1520" alt="2" src="https://github.com/junyong1111/ComputerVision/assets/79856225/3eee4f33-69da-40c2-a40d-0dc0ea50a492">

**dataset이 많을수록 전체 모집단을 대변할 수 있다.**

**Maximum Likelihood Estimation**

<img width="800" alt="3" src="https://github.com/junyong1111/ComputerVision/assets/79856225/ee1a0e6f-161a-4149-962b-b3d29a91e796">

𝕏 : 데이터셋

𝜽 : 모델 파마메터

- Pmodel(𝕏, 𝜽) → 데이터셋의 각각의 이미지들은 모두 독립적이다
- 사건이 독립적이라면 각각의 사건은 곱으로 표현될 수 있다(사건의 독립)
- Pmodel(x1, 𝜽1) * Pmodel(x2, 𝜽2) * …Pmodel(x70000, 𝜽7000)

**최고의 성능을 보여주는 모델 파리미터를 학습**

이 대 0 ~ 1사이에 값을 곱하면 불안정하다. 따라서 이 값들 -**∞ ~ 0의 값으로 매핑해주기 위해 log를 씌운다.**

뿐만 아니라 로그를 씌움으로 복잡한 계산을 쉽게 할 수 있다.

<img width="543" alt="4" src="https://github.com/junyong1111/ComputerVision/assets/79856225/ec8464ba-2f0b-467d-8b2b-1cc07e7fe254">

**Bayesian Statistics(베이지안 통계)**

- Prior : 사전 지식이며 𝜽에 대한 사전지식을 알고 잇음
- Maximum Likeligood * Prior

### AutoEncoder

**AutoEncoder(안정적)**

**x → (autoencoder) → x’**

**입력과 유사한 출력이 나오게 하는 모델을 만드는 것이 목표**

- 암기과목등을 공부하는 인간과 유사하게 학습한다.
- 배운 내용들을 최대한 요약하고 그 요약한 것을 기준으로 다시 학습했던것과 유사한 출력을 내뱉음

<img width="364" alt="5" src="https://github.com/junyong1111/ComputerVision/assets/79856225/7683c13f-c170-425b-b6a1-e743e31d0203">

**Encoder**

- input으로 들어온 x데이터에 대한 특징들을 압축하는 과정이다.
- 압축이 끝난 데이터 latent code(잠재 공간) z를 출력한다.
- Z = E𝜽(x)로 표현

<img width="356" alt="6" src="https://github.com/junyong1111/ComputerVision/assets/79856225/65ed57c1-7a0d-4c60-ad14-47fb12003f45">

**Decoder**

- Encoder에서 나온 latent code(잠재 공간) z값을 input으로 압축된 데이터를 복원하는 과정
- z에 대한 복원이 끝나면 input(x)와 똑같은 형태로 출력
- X’ = DΦ(z)로 표현

<img width="342" alt="7" src="https://github.com/junyong1111/ComputerVision/assets/79856225/868932ad-e2a6-4146-ac7b-d33007506c8a">

**End to End Learning**

- Auto Encoder는 self-supervised learning
- **input으로 들어온 x데이터 자체가 레이블값이기 때문**
- X’ = DΦ(E𝜽(x)) 합성함수 형태로 표현
    
    → x데이터로 𝜽, Φ를 학습
    
<img width="342" alt="8" src="https://github.com/junyong1111/ComputerVision/assets/79856225/e24aca95-6858-41a4-ac33-7da08c38c088">