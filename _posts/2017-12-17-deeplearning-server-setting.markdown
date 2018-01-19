---
layout: post
title:  "딥러닝용 서버 설치기"
date:   2017-12-17 18:10:00
image: /assets/article_images/2014-08-29-welcome-to-jekyll/desktop.JPG
comments: true
---
어제 딥러닝용 1080 Ti GPU 서버의 개발 환경 세팅을 마쳤습니다. 이 글에서는 서버 구매부터 Ubuntu 설치, NVIDIA driver, CUDA 및 cuDNN 설치, 그리고 Tensorflow와 PyTorch 설치까지 제가 개발 환경을 세팅한 방법을 정리했습니다.

# 하드웨어 구성

참고를 위해 제가 구성한 서버 하드웨어도 보여드리겠습니다.

괄호 안의 가격은 실제 구매한 가격이 아니라 참고를 위해 적어놓은 현금 최저가입니다.

- CPU : 인텔 코어X-시리즈 i7-6850K (브로드웰-E) (587,870원)
- 메인보드 : ASUS X99-A II STCOM (432,490원)
- 메모리 : 삼성전자 DDR4 16G PC4-19200 X 4 (185,850원)
- 그래픽카드 : GIGABYTE 지포스 GTX1080 Ti AORUS D5X 11GB (1,061,700원)
- 하드디스크 : WD 4TB BLUE WD40EZRZ (SATA3/5400/64M)  (135,450원)
- 케이스 : BRAVOTEC 스텔스 TX 블랙 파노라마 윈도우 (58,000원)
- 파워 : 써멀테이크 터프파워 그랜드 RGB 850W 골드 풀 모듈러 (177,450원)
- 쿨러/튜닝 : CORSAIR HYDRO SERIES H80i v2  (148,540원)
- 외장HDD : Toshiba CANVIO BASIC 2 1TB  (69,830원)
- 조립비 : 컴퓨터 프리미엄 조립 + 1년 전국 무상 방문출장AS (1대분) (35,000원)

총 상품 금액 : 3,604,000원 (배송비 9,000원 제외)

가격 비교는 다나와(http://www.danawa.com) > **온라인 견적**을 이용했습니다. 다나와에서 견적을 만든 후 **호환성 체크**를 한 뒤 **구매신청 등록**을 클릭하면 구매를 진행할 수 있습니다.

![다나와(www.danawa.com)](https://files.slack.com/files-pri/T25783BPY-F8FUBADFF/screenshot.png?pub_secret=4f69dcc170)

![다나와 호환성 체크와 구매신청](https://files.slack.com/files-pri/T25783BPY-F8F6LG064/danawa.png?pub_secret=3ac2f607cb)

주문을 완료한 뒤 배송이 오면 이제 개발 환경을 세팅할 차례입니다.

# Ubuntu 설치하기

Ubuntu 14.04를 설치합니다. 최신 LTS인 16.04는 GTX1080 Ti GPU가 달린 서버에서 바로 부팅 시 검은 화면만 뜨는 문제가 있습니다. 설정을 바꿔 이 문제를 우회할 수는 있지만 설치 과정에서 복잡성을 최소화하기 위해 14.04를 설치하는 방법을 택했습니다. 14.04는 언제든지 16.04로 업그레이드할 수 있습니다.

### Ubuntu 다운받기

Ubuntu 14.04로 검색한 뒤 나오는 페이지(http://releases.ubuntu.com/14.04/)에서 [64-bit PC (AMD64) desktop image](http://releases.ubuntu.com/14.04/ubuntu-14.04.5-desktop-amd64.iso)를 다운받습니다.



### 부팅 디스크 만들기

USB 하나를 부팅 디스크로 만들어야 합니다. 

> 다음 과정은 Mac 기준이며 윈도우의 경우 다른 방식이 필요합니다.

먼저, Disk Utility를 이용해서 USB를 포맷합니다.

* **Applications** > **Disk Utility**를 실행합니다.
* USB를 선택한 뒤 **Erase**를 클릭합니다.
* Format은 **MS-DOS (FAT)**으로 지정하고 Scheme은 디폴트 상태로 둔 뒤 **Erase**를 눌러 포맷을 진행합니다.

![Disk Utility에서 USB 포맷하기](https://files.slack.com/files-pri/T25783BPY-F8FR4GM0V/screenshot.png?pub_secret=d1d065ce2b)

* UNetbootin(https://unetbootin.github.io/)를 설치합니다.
* 설치한 UNetbootin에 다운받은 Ubuntu 14.04 파일(.iso)을 넣고 USB를 지정해주면 부팅 디스크를 만들어줍니다.

![UNetbootin으로 부팅 디스크 만들기](https://files.slack.com/files-pri/T25783BPY-F8GGX7F0W/screenshot.png?pub_secret=32447bd174)



이제 부팅 디스크가 만들어졌습니다. 이 USB를 서버에 꽂은 뒤 BIOS에서 부팅을 할 때 **Boot Menu**에서 USB에 해당하는 항목을 클릭한 뒤 **Install Ubuntu**를 선택한 뒤 나오는 안내에 따라 설정을 입력하면 Ubuntu 설치가 완료됩니다.



##### 트러블 슈팅

서버의 전원을 켰을 때 `CPU Fan speed error detected`라는 문구가 계속해서 떴습니다. 이는 제가 수냉식 쿨러를 사용했고 ASUS 메인보드에서 이슈가 있었기 때문이었습니다. BIOS 상에서 Fan 설정을 바꾸어서 해결했습니다. (참고 : [아수스 CPU Fan speed errer detected 해결하기](http://rgy0409.tistory.com/1165))

# NVIDIA driver 설치하기

Ubuntu에서 NVIDIA driver를 설치하는 가장 간단하고 권장되는 방법은 운영체제가 제공하는 설정 메뉴에서 GUI 그래픽 드라이버 설치 방법를 따르는 것입니다.

* Ubuntu에서 **System Settings** > **Software & Updates** > **Additional Drivers**를 클릭합니다.
* 잠시 기다린 뒤 표시되는 선택지 중에 **using NVIDIA binary driver**를 선택합니다.

![System Settings에서 NVIDIA driver 다운로드 하기](https://files.slack.com/files-pri/T25783BPY-F8GU1GBQF/driver.png?pub_secret=a6137dc6e3)

### 설치 확인하기

다음 명령어를 터미널 창(단축키 : ctrl + alt + t)에서 입력합니다. 이 때 GPU 정보가 올바르게 표시되면 설치가 정상적으로 완료된 것입니다. 첫번째 명령어는 텍스트로, 두번째 명령어는 그래픽으로 GPU 정보를 표시합니다.

`nvidia-smi`

`nvidia-settings`

![driver 설치 후 nvidia-smi가 정상적으로 표시된 화면](https://files.slack.com/files-pri/T25783BPY-F8F6NLDEC/smi.png?pub_secret=7ecd36a66c)

##### 참고 자료

CLI를 이용해 NVIDIA driver를 설치하고 싶다면 다음 자료를 참고합니다. 다만 두번째 링크에서도 밝히고 있듯이 이는 권장되는 방식은 아닙니다.

* [[우분투] nvidia 드라이버 설치](http://pythonkim.tistory.com/48)
* [Install Nvidia driver instead of nouveau](https://askubuntu.com/questions/481414/install-nvidia-driver-instead-of-nouveau)



# CUDA 설치하기

현재 CUDA 최신 버전은 9.1이지만 TensorFlow는 CUDA 8.0 설치를 요구하고 있습니다. 따라서 여기에서도 CUDA 8.0을 설치하겠습니다.

* CUDA 8.0 다운로드 페이지 (https://developer.nvidia.com/cuda-80-ga2-download-archive)에 들어갑니다.
* **Linux > x86_64 > Ubuntu > 14.04 > deb (local)**을 선택하고 [다운로드](https://developer.nvidia.com/compute/cuda/8.0/Prod2/local_installers/cuda-repo-ubuntu1404-8-0-local-ga2_8.0.61-1_amd64-deb)합니다.
* 다운로드 페이지에서 안내되는 것처럼 다음 명령어를 입력합니다.

```bash
sudo dpkg -i cuda-repo-ubuntu1404-8-0-local-ga2_8.0.61-1_amd64.deb
sudo apt-get update
sudo apt-get install cuda
sudo reboot
```

* bashrc 파일을 열고 맨 아래에 다음 환경변수를 추가합니다.

```shell
gedit ~/.bashrc
```

```shell
export CUDA_HOME=/usr/local/cuda-8.0
export PATH=/usr/local/cuda-8.0/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```

![CUDA 다운로드 페이지](https://files.slack.com/files-pri/T25783BPY-F8F5HGB3J/cuda.png?pub_secret=88515d4aa2)



##### 참고 자료

* [Ubuntu에 CUDA 설치](http://haanjack.github.io/cuda/2016-02-29-cuda-linux/)
* [Ubuntu 16.04 LTS에 tensorflow gpu버전 설치하기](https://seojmediview.com/2017/08/09/ubuntu-16-04-lts%ec%97%90-tensorflow-gpu%eb%b2%84%ec%a0%84-%ec%84%a4%ec%b9%98%ed%95%98%ea%b8%b0/)

# cuDNN 설치하기

TensorFlow가 cuDNN 6.0을 요구하므로 최신 버전인 7.0이 아닌 6.0 버전을 설치해보겠습니다.



* cuDNN 다운로드 페이지(https://developer.nvidia.com/cudnn)에 들어갑니다.
* **Download** 버튼을 클릭하면 회원 가입을 하라는 문구가 뜹니다. NVIDIA 회원 가입을 진행합니다.
* 로그인 후 다시 **Download** 버튼을 클릭하고 몇가지 질문에 답한 후 이용 약관에 동의하면 다운로드 가능한 버전들을 볼 수 있습니다.
* 이 중에서 **Download cuDNN v6.0 (April 27, 2017), for CUDA 8.0** 를 클릭하면 다운로드 리스트가 뜹니다.
* Linux용 버전인 [cuDNN v6.0 Library for Linux](https://developer.nvidia.com/compute/machine-learning/cudnn/secure/v6/prod/8.0_20170307/cudnn-8.0-linux-x64-v6.0-tgz) 링크를 클릭하면 다운로드가 진행됩니다.

![cuDNN 다운로드 페이지](https://files.slack.com/files-pri/T25783BPY-F8FQY896W/cdnn-annotated.png?pub_secret=f996fb45cc)

* 다음 명령어를 입력해서 다운받은 cuDNN 파일의 압축을 풀고 cuda 라이브러리에 복사합니다.

```shell
cd Downloads/

tar xvzf cudnn-8.0-linux-x64-v5.1.tgz
sudo cp cuda/include/cudnn.h /usr/local/cuda-8.0/include
sudo cp cuda/lib64/libcudnn* /usr/local/cuda-8.0/lib64
sudo chmod a+r /usr/local/cuda-8.0/include/cudnn.h /usr/local/cuda-8.0/lib64/libcudnn*
```

##### 참고 자료

* [Ubuntu 16.04 LTS에 tensorflow gpu버전 설치하기](https://seojmediview.com/2017/08/09/ubuntu-16-04-lts%ec%97%90-tensorflow-gpu%eb%b2%84%ec%a0%84-%ec%84%a4%ec%b9%98%ed%95%98%ea%b8%b0/)



# Anaconda 설치하기

Anaconda는 Python과 여러 라이브러리들을 설치하는 가장 편리하고 빠른 방법입니다. 현재 Anaconda를 다운받으면 Python 3.6으로 설치되지만, [TensorFlow 1.4 GPU 버전이 아직 Python 3.6을 지원하지 않기 때문에](https://github.com/tensorflow/tensorflow/issues/14182) Python 3.5 환경으로 맞춰줄 필요가 있습니다. 여기에서는 Python 3.5를 지원하는 Anaconda의 이전 버전을 설치하겠습니다.

또한, 설치된 Python을 모든 사용자가 사용할 수 있게 하기 위해 아래 과정부터 CLI 명령어는 root 계정에서 실행하였습니다.

```shell
sudo su
```



* Anaconda 아카이브 페이지(https://repo.continuum.io/archive/index.html)에 접속합니다.
* [Anaconda3-4.2.0-Linux-x86_64.sh](https://repo.continuum.io/archive/Anaconda3-4.2.0-Linux-x86_64.sh) 버전을 선택하여 다운로드합니다. Anaconda3 4.2 버전은 Python 3.5가 디폴트인 버전입니다.
* 다운로드 받은 파일이 담긴 폴더로 이동하여 bash로 해당 파일을 실행합니다.

```shell
bash Anaconda3-4.2.0-Linux-x86_64.sh
```

* (Optional) 모든 사용자가 설치된 Python에 접근할 수 있게 하기 위해 Anaconda 설치 중 설치 경로 설정에서 `/opt/conda`를 입력하였습니다. 이 과정을 생략하면 사용자의 home 디렉토리 아래에 Anaconda가 설치됩니다.
* PATH를 추가하겠느냐는 질문에 yes를 입력합니다.

### 설치 확인하기

`python`을 실행했을 때 Anaconda 4.2의 Python 3.5가 실행되면 정상적으로 설치된 것입니다.

```shell
$ python

Python 3.5.2 |Anaconda 4.2.0 (64-bit)| (default, Jul  2 2016, 17:53:06) 
[GCC 4.4.7 20120313 (Red Hat 4.4.7-1)] on linux
Type "help", "copyright", "credits" or "license" for more information.
```



# Tensorflow 설치하기

이 글을 작성하는 때에 TensorFlow의 최신 버전은 1.4입니다. 최신 버전인 TensorFlow 1.4 버전을 설치해 보겠습니다. TensorFlow에서는 가상 환경 위에 설치를 권장하고 있지만 여기에서는 사용의 편리함을 위해 디폴트 Python 위에 pip으로 바로 설치합니다.



* 다음 명령어로 libcupti-dev 라이브러리를 설치합니다. 이는 TensorFlow [공식 설치 가이드](https://www.tensorflow.org/install/install_linux)의 요구사항을 따른 것입니다. CUDA와 cuDNN은 이미 설치했으니 마지막 단계인 libcupti-dev 라이브러리만 남았습니다.

```shell
sudo apt-get install libcupti-dev
```

* pip으로 TensorFlow GPU 버전을 설치합니다. pip으로 설치하는 가이드는 [Installinng with native pip](https://www.tensorflow.org/install/install_linux#InstallingNativePip)에서 찾을 수 있고 설치 URL은 [URL of the TensorFlow Python package](https://www.tensorflow.org/install/install_linux#the_url_of_the_tensorflow_python_package)에서 Python 3.5 버전을 선택했습니다.

```shell
pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.4.0-cp35-cp35m-linux_x86_64.whl
```



### 설치 확인하기

Python을 실행한 뒤 TensorFlow를 import하고, Session을 열 때 GPU 정보가 표시되면 TensorFlow GPU 버전이 정상적으로 설치된 것입니다.

```python
>>> import tensorflow as tf
>>> tf.Session()
2017-12-17 11:56:09.494588: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1030] Found device 0 with properties: 
name: GeForce GTX 1080 Ti major: 6 minor: 1 memoryClockRate(GHz): 1.683
pciBusID: 0000:01:00.0
totalMemory: 10.91GiB freeMemory: 10.49GiB
2017-12-17 11:56:09.494619: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1120] Creating TensorFlow device (/device:GPU:0) -> (device: 0, name: GeForce GTX 1080 Ti, pci bus id: 0000:01:00.0, compute capability: 6.1)<tensorflow.python.client.session.Session object at 0x7f9c04d43668>
```



### PyTorch 설치하기

PyTorch 설치는 공식 사이트(www.pytorch.org)를 그대로 따르면 됩니다. conda로 Python 3.5, CUDA 8.0 버전을 설치하였습니다.

```shell
conda install pytorch torchvision -c pytorch
```



# 네트워크 연결하기

이제 서버 개발 환경 세팅은 완료되었으니, 이 서버를 다른 컴퓨터에서도 접속할 수 있게 만들어야 합니다. 이를 위해서 외부 포트를 내부 포트로 연결하는 포트 포워딩을 설정하고, Ubuntu 내에서 SSH를 외부에서 접근 가능하도록 설정합니다.

> 다음 과정은 iptime을 사용한다고 가정합니다. 다른 제품을 쓰는 경우 과정이 약간 다를 수 있습니다.



### 포트포워딩

* **192.168.0.1** 주소로 접속하여 iptime 관리 페이지로 이동합니다. 관리 페이지를 모를 때는 Ubuntu 내 Terminal에서 `ifconfig`를 입력하여 내부 IP 주소(inet addr)를 알아낸 뒤 마지막 자리를 1로 치환한 주소가 관리 페이지 주소일 가능성이 높습니다. 
* 로그인 후 **관리 도구 > 고급 설정 > NAT/라우터 관리 > 포트포워드 설정**로 이동합니다.
* 새 규칙을 추가합니다. **규칙 이름**과 **외부 포트**, **내부 포트**, 그리고 **내부 IP주소**를 설정해주면 됩니다.

![iptime 포트포워딩 설정](https://files.slack.com/files-pri/T25783BPY-F8FSAKX0U/iptime.png?pub_secret=6b0d0ea838)



### SSH 외부 접속 설정하기

SSH 설정 방법은 [How to Enable SSH on Ubuntu](https://thishosting.rocks/how-to-enable-ssh-on-ubuntu/)를 따랐습니다.



* OpenSSH를 설치합니다.

```shell
sudo apt-get install openssh-server -y
```

* 다음 명령어로 SSH 설정 파일을 엽니다.

```shell
sudo nano /etc/ssh/sshd_config
```

* Port 설정을 수정합니다.

다음과 같은 라인을

```shell
# Port 22
```

다음과 같이 변경합니다. 이 때 Port 번호는 포트포워드에서 외부 포트에 연결한 내부 포트 번호와 일치해야 합니다.

```shell
Port 1337
```

* SSH를 재시작합니다.

```shell
sudo service ssh restart
```



### 공인 IP 알아내기

외부에서 접근하려면 공인 IP(public IP)를 알아야 합니다. 이는 간단하게 구글에서 `my ip` 등으로 검색해서 나오는 사이트에서 알아낼 수 있습니다.



### 설정 확인하기

외부 컴퓨터에서 SSH로 접속에 성공하면 설정이 완료된 것입니다.

> 다음 명령어는 Mac이나 Linux에서 바로 실행 가능하지만, 일반적인 Windows에서 동작하지 않습니다.

```shell
ssh <공인 IP> -l <username> -p <포트 번호>
```

