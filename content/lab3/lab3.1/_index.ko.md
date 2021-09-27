---
title: 여러 개의 얼굴 인식하기 (심화)
weight: 50
pre: "<b>4-1. </b>"
---

### 인식하려는 얼굴을 위한 S3 생성

1. S3 버켓을 하나 생성합니다.
1. 인식하려는 얼굴 사진을 업로드 합니다.

### DeepLens 람다 함수 템플릿 구성

[AWS 딥렌즈 람다 함수 템플릿](https://docs.aws.amazon.com/deeplens/latest/dg/samples/deeplens_inference_function_template.zip) 을 다운 받은 후 압축해제를 합니다.

폴더 내에는 다음과 같은 파일들이 들어 있습니다.

1. lambda_function.py: 딥렌즈 내부에서 돌아가는 실제 함수입니다.
1. local_display.py: 결과 영상 송출을 위한 파일입니다.
1. greengrasssdk: AWS Greengrass 배포를 위한 sdk 입니다.

여러 개의 얼굴을 인식하기 위해서 [다음 코드](https://github.com/k2sebeom/DeepLens-Rekognition-Demo/blob/main/multi-face-compare.py) 를 이용할 것입니다. lambda_function.py 를 열고 내용을 모두 지운 후, 코드를 붙여넣기 합니다.

1. 60번째 줄의 {s3-bucket-name} 을 얼굴 사진이 들어 있는 버켓 이름으로 변경합니다.
```{python}
def get_resp(client, img_key, img_bytes):
    try:
        return {
            "key": img_key,
            "resp": client.compare_faces(
                        SourceImage={
                            'S3Object': {
                                'Bucket': '{s3 bucket name}',
                                'Name': img_key
                            }
                        },
                        TargetImage={
                            'Bytes': img_bytes
                        }
            )
        }
    except Exception:
        return None
```

2. 104번째 줄의 target_dict 를 업데이트합니다. key는 s3 오븐젝트 키를, value 는 각 사진에 해당하는 label 을 작성합니다.

```{python}
target_dict = {
    "list.jpeg": "Names",
    "of.png": "of",
    "s3_keys.jpeg": "faces"
}
```

3. 전체 폴더를 다시 zip 파일로 압축합니다. 


### 람다 함수 생성하기

1. [AWS Lambda 콘솔](https://console.aws.amazon.com/lambda) 에 들어갑니다. 
1. "함수 생성"을 클릭합니다.
1. 함수 이름을 작성합니다. 이때 함수 이름은 deeplens 로 시작해야 합니다. 예) deeplens_custom_face
1. 런타임은 Python 3.7 을 선택합니다.
1. 함수를 생성합니다.
1. 코드 소스 부분으로 이동한 후, "에서 업로드 -> .zip 파일" 을 클릭합니다.
1. 위에서 만든 zip 파일을 업로드합니다.
1. 페이지 위로 올라가서 "작업 -> 새 버전 발행" 을 클릭합니다.
1. 새로운 버전을 발행합니다.

### 딥렌즈 프로젝트 생성

1. [AWS DeepLens 콘솔](https://console.aws.amazon.com/deeplens) 에 들어갑니다. 
1. Projects 를 클릭합니다.
1. Create new project 클릭하고, Create a new blank project 를 클릭합니다.
1. 프로젝트 이름과 설명을 작성합니다.
1. Project content 에서 Add model 을 한 후, deeplens-face-detection 을 선택합니다. 앞의 과정에서 face-detection 샘플 프로젝트를 배포했다면, 모델이 있을 것입니다.
1. Project content 에서 Add function 을 한 후, 위에서 배포한 람다 함수를 선택합니다.
1. 디바이스로 배포합니다.


<p align="center">
© 2020 Amazon Web Services, Inc. 또는 자회사, All rights reserved.
</p>


