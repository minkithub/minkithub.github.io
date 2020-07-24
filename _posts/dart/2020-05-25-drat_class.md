---
title: "Dart 기본 문법 정리 2편"
subtitle: "getter and setter"
layout: post
author: "Minki"
# header-img: "img/seoulpark_series/seoul_park.jpg"
# header-mask: 0.6
header-style: text
catalog: true
tags:
  - dart
  - flutter
---

*잘못된 내용은 언제든지 밑의 댓글로 알려주세요!*

# void 문법

dart에서 void는 1급 객체입니다. 따라서 void로 지정한 method가 변수로 취급될 수 있습니다.

## 1. void에 변수 넣기

```dart
void main(String name) {
    something(); //anything
}

void something(String name) {
    print('anything')
}
```

something이라는 method를 만들고 그 안에 지정된 값을 넣으면 main함수에서 지정된 값만을 호출 할 수 있다.

## 2. void에 옵션으로 변수 추가

void에 옵션과 기본값을 지정할 수 있습니다.

```dart
void main() {
    something(180.3, name : '홍길동', age : 25); // 180.3, 홍길동, 25, male
}

void something(num height, {String name, int age, String gender = 'male'}) {
    print(name);
    print(age);
    print(gender);
    print(height);
}
```

1. height :  필수값. main 내에 `height : 180.3`으로 넣으면 옵션 값이 아니니 오류가 납니다.
2. name, age : 옵션값. main 내에 값을 넣지 않으면 `NULL`값이 나옵니다.
3. gender : 옵션값. main 내에 값을 넣지 않으면 기본값인 `male`이 출력됩니다.

# class 문법

class문법은 기본적으로 react와 javascript와 비슷합니다. 기본적인 문법 코드를 보겠습니다.

## 1. class 생성

```dart
void main() {
  var person = Person('홍길동', 25, gender : 'male');
  print(person.name);
  print(person.age);
  print(person.gender);
}

class Person {
  String name;
  int age;
  String gender;
  
  // constructor 생성법 1
  Person(String name, int age, {String gender = 'Female'}) {
    this.name = name;
    this.age = age;
    this.gender = gender;
  }

  print('============================================')

  // constructor 생성법 2
  Person2(this.name, this.age, {this.gender = 'Female'})
}
```

## 2. class Getter and Setter 활용

Getter and Setter를 활용하면 class에 할당된 변수를 이용하여 다른 추가적인 변수를 만들 수 있습니다.

* `_name` 변수를 받아 `name`변수 생성
* `_age` 변수를 받아 `age`변수 생성

```dart
void main() {
  var person = Person('홍길동', 25);
  print(person._name); // 홍길동
  print(person.name); // 제 이름은 홍길동 입니다.
  print(person.age); // 제 나이는 25 입니다.
}

class Person {
  // private 변수를 만들어서 외부에서 접근하지 못하도록 한다.
  String _name;
  int _age;
  
  // class Person에서 선언해준 변수들이 this로 설정되어야 함.
  // 이후 this에 할당된 변수를 가지고 추가 변수 생성
  Person(this._name, this._age);
  String get name => '제 이름은 $_name 입니다.';
  String get age => '제 나이는 $_age 입니다.';

  print('============================================')

  // 일반적인 method의 형태는 다음과 같이 표현됨.
  String get name {
    return '제 이름은 $_name 입니다.';
  }
}
```

또한 java에는 없는 class표현에는 다음과 같은 것들이 있습니다.

```dart
void main() {
  var person = Person( );
  
  person.setName('홍길동');
  person.setAge(25);
  
  // 위의 두줄을 밑의 한줄로 바꿔 표현 할 수 있다.
  var person2 = Person()
    ..setName('홍길동')
    ..setAge(25);
}

class Person {
  // private 변수를 만들어서 외부에서 접근하지 못하도록 한다.
  String name;
  int age;
  
  void setName(String name) {
    this.name = name;
  }
  
  void setAge(int age) {
    this.age = age;
  }
}
```

java에서 Interface에 해당하는 것은 다음과 같이 구현할 수 있습니다.

```dart
// Person의 기능만 가져오고 싶은 경우
class Employee implements Person {
  @override
  int age;
  
  @override
  String name;
  
  @override
  void setAge(int age) {

  }
  
  @override
  void setName(String name) {
    
  }
}

// person의 기능 중 일부만 가져오고 싶은 경우
// implemets대신 with를 사용하면 된다.
class Employee with Person {
    @override
    void setName(String name) {
    
  }
}
```

# Future 활용

## 1. Future 활용

`await`와 `async`를 `Future method`와 함께 사용해서 네트워크 요청시간과 끝 사이에 3초간의 딜레이를 설정합니다. 이는 Flutter 앱을 만들 때 많이 사용됩니다.

```dart
void main() {
  networkRequest();
}

Future networkRequest() async {
  print('네트워크 요청 시작');
  await Future.delayed(Duration(seconds: 3));
  print('네트워크 요청 끝');
}
```

## *reference*
* 오준석의 생존코딩