---
layout: post
title:  "Conda 가상 환경으로 PyTorch 설치하기"
date:   2018-01-30
tags:
image:
comments: true
---

PyTorch 설치가 어려울 때, conda 가상 환경 안에 PyTorch를 설치하면 깔끔하게 설치될 때가 많습니다. 이 글은 conda 가상 환경으로 PyTorch를 설치하고 Jupyter의 kernel로 등록하는 방법을 소개합니다. TensorFlow도 같은 방법으로 설치할 수 있습니다.

# Windows

**새 가상 환경 만들기** 

```shell
$ conda create -y -n pytorch ipykernel
```

`pytorch` 대신 자신이 원하는 이름을 쓸 수 있습니다.

**가상 환경 안으로 들어가기**

```shell
$ activate pytorch
```

**PyTorch 설치하기**

```shell
(pytorch)$ conda install -y -c peterjc123 pytorch
```

**Jupyter에 새 kernel 등록하기** 

```shell
(pytorch)$ python -m ipykernel install --user --name pytorch --display-name "PyTorch"
```

`--display-name`은 Jupyter Notebook 위에서 표시될 kernel의 이름으로 `"PyTorch"` 대신 자신이 원하는 이름을 쓸 수 있습니다.

**가상 환경 빠져나오기**

```shell
(pytorch)$ deactivate
```

# MacOS / Linux

**새 가상 환경 만들기** 

```shell
$ conda create -y -n pytorch ipykernel
```

`pytorch` 대신 자신이 원하는 이름을 쓸 수 있습니다.

**가상 환경 안으로 들어가기**

```shell
$ source activate pytorch
```

**PyTorch 설치하기**

```shell
(pytorch)$ conda install -y pytorch torchvision -c pytorch
```

**Jupyter에 새 kernel 등록하기** 

```shell
(pytorch)$ python -m ipykernel install --user --name pytorch --display-name "PyTorch"
```

`--display-name`은 Jupyter Notebook 위에서 표시될 kernel의 이름으로 `"PyTorch"` 대신 자신이 원하는 이름을 쓸 수 있습니다.

**가상 환경 빠져나오기**

```shell
(pytorch)$ source deactivate
```

# Jupyter에 등록된 kernel 확인하기

이제 Jupyter Notebook에서 새 노트북을 만들 때 PyTorch kernel을 선택할 수 있습니다.

![Jupyter Notebook에 등록된 PyTorch kernel](https://files.slack.com/files-pri/T25783BPY-F901YJLAV/_______________.png?pub_secret=0974008d7a)

# 해설

#### 새 가상 환경 만들기에 대해서

Conda는 패키지 관리 프로그램인 동시에 가상 환경 관리 프로그램입니다. Conda로 기존 환경과 충돌이 없는 가상 환경을 만들고 관리할 수 있습니다.

```shell
$ conda create -y -n pytorch ipykernel
```

- `conda create` : 새 conda 환경을 만듭니다.
- `-y` : `--yes`의 줄임말입니다. 설치 승인을 생략하고 바로 설치합니다.
- `-n pytorch` : `--name pytorch`의 줄임말입니다. 환경 이름을 pytorch로 짓습니다. 환경 이름은 필수입니다. 이름은 원하는 대로 지을 수 있습니다.
- `ipykernel` : ipykernel이 설치되어 있는 가상 환경을 만듭니다. ipykernel 외에 다른 패키지 이름을 쓸 수 있습니다.

#### Jupyter에 새 kernel 등록하기에 대해서

```shell
(pytorch)$ python -m ipykernel install --user --name pytorch --display-name "PyTorch"
```

* `python -m ipykernel` : ipykernel [모듈을 파이썬 스크립트로 실행](https://www.python.org/dev/peps/pep-0338/)합니다.
* `—name pytorch` : Jupyter 내부적으로 쓰이는 kernel의 이름을 지정합니다. 같은 이름을 쓰면 덮어쓰기가 됩니다.
* `--display-name "PyTorch"` : Jupyter Notebook 위에서 사용자에게 보이는 kernel의 이름을 정합니다. 내부적으로 쓰이는 이름과 상관없이 띄어쓰기 등 특수문자도 포함하여 자유롭게 지을 수 있습니다. 

# 유용한 Conda 명령어

conda를 사용할 때 자주 사용하는 명령어들을 모아보았습니다.

## 가상 환경 만들기 옵션

**파이썬만 있는 최소한의 환경을 원할 때**

```shell
$ conda create --name myenv python
```

파이썬과 pip 등의 최소한의 패키지만 설치되어 있는 `myenv`라는 이름의 새 가상 환경을 만듭니다. ipykernel, numpy 등이 모두 없는 환경입니다. `myenv` 대신 원하는 이름을 쓸 수 있습니다.

**파이썬 버전을 지정하고 싶을 때**

```shell
$ conda create --name myenv python=2.7
```

Python2.7의 가상 환경을 만듭니다. 이처럼 원하는 파이썬 버전을 지정하여 가상 환경을 만들 수 있습니다. 위와 마찬가지로 최소한의 환경만 설치합니다.

**아나콘다 환경을 만들고 싶을 때**

```shell
$ conda create --name myenv anaconda
```

아나콘다 환경에는 numpy, pandas 등 수학/과학 관련 패키지들이 포함됩니다. ipykernel 역시 설치되어 있습니다.

**파이썬 버전을 지정하고 아나콘다 환경을 만들고 싶을 때**

```shell
$ conda create --name myenv python=2.7 anaconda
```

## 가상 환경 목록 보기

```shell
$ conda env list

pytorch
myenv
myenv2
...
```

## 가상 환경 삭제하기

```shell
$ conda env remove --name myenv
```

# 참고 자료

- [Installing the IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html)
- [Managing environments in Conda](https://conda.io/docs/user-guide/tasks/manage-environments.html)
- [Installing Pytorch on WIndows](https://github.com/peterjc123/pytorch-scripts)